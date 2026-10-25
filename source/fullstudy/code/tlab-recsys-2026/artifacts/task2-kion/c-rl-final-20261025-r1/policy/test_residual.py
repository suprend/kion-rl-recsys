"""Stdlib-only contract/seam tests; no real arrays/models/GPU/provider jobs."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('residual_under_test', HERE / 'residual.py')
residual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(residual)
utility_spec = importlib.util.spec_from_file_location('utility_for_residual_test', HERE.parent / 'reward' / 'utility.py')
utility = importlib.util.module_from_spec(utility_spec)
utility_spec.loader.exec_module(utility)


class ResidualTests(unittest.TestCase):
    def test_module_descriptors_accept_string_or_pinned_object(self):
        utility_path = (HERE.parent / 'reward' / 'utility.py').resolve()
        request = dict(source_hashes={utility_path.as_posix(): residual.sha(utility_path)},
                       utility_module=utility_path.as_posix())
        self.assertEqual(residual.load_utility(request).CONTRACT, utility.CONTRACT)
        request['utility_module'] = dict(path=utility_path.as_posix(), sha256=residual.sha(utility_path))
        self.assertEqual(residual.load_utility(request).CONTRACT, utility.CONTRACT)
        request['utility_module']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            residual.load_utility(request)
        # Wrong frozen-policy descriptor fails before opening its path.
        with self.assertRaises(ValueError):
            utility.load_policy(dict(policy_module={'path': 'missing.py', 'sha256': '0' * 64}, source_hashes={}))

    def test_sampler_matches_frozen_order_with_duplicates(self):
        eligible = [0, 2, 5, 9]
        old_rng, new_rng = random.Random(20261024), random.Random(20261024)
        for _ in range(1000):
            expected = [old_rng.choice(eligible) for _ in range(16)]
            actual = residual.sample_positions(new_rng, eligible)
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), 16)

    def test_import_is_inert_and_no_monkeypatch(self):
        tree = ast.parse((HERE / 'residual.py').read_text(encoding='utf-8'))
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in statement.names] if isinstance(statement, ast.Import) else [statement.module]
                self.assertFalse(any(name.split('.')[0] in {'numpy', 'torch', 'catboost'} for name in names))
        assignments = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
        for assignment in assignments:
            targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
            self.assertFalse(any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                 and t.value.id in {'policy', 'utility'} for t in targets))

    def test_explicit_frozen_numeric_seams_and_gates_precede_optimizer(self):
        source = (HERE / 'residual.py').read_text(encoding='utf-8')
        for name in ('producer_layout', 'prepare_shaped_inputs', 'score_shaped', 'exact_loss',
                     'effective_advantage', 'distribution', 'precision', 'rank'):
            self.assertIn('policy.' + name + '(', source)
        self.assertNotIn('select_beta(', source)
        self.assertNotIn('frontier_output', source)
        self.assertLess(source.index('validate_prefit(request, utility)', source.index('def train(')), source.index('torch.optim.Adam('))
        self.assertLess(source.index('validate_parity_proof(request, manifest, policy)', source.index('def train(')), source.index('torch.optim.Adam('))
        self.assertLess(source.index("'Update0 top10 mismatch'"), source.index('torch.optim.Adam('))
        self.assertIn("lr=1e-5, weight_decay=0", source)
        self.assertIn('range(1, 1001)', source)

    def test_new_prereg_accepts_without_old_frontier_and_rejects_changed_fit(self):
        value = dict(contract=utility.PREREG, chosen_utility='rm-minus-frozen-item-mean', formula=utility.FORMULA,
                     coefficient=1., beta=.5, temperature=1., sampler_seed=20261024,
                     checkpoint_steps=[0, 250, 500, 1000], start_from='original-C', original_results_known=True,
                     one_additional_fit_only=True, primary_contrast='fixed-latest1000',
                     no_gpt_unfreeze=True, no_rm_fit=True, no_sft=True,
                     optimizer=dict(name='Adam', updates=1000, batch_size=16, lr=1e-5,
                                    grad_clip=1, weight_decay=0, alpha_ce=0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prereg.json'
            def request(data):
                path.write_text(json.dumps(data), encoding='utf-8')
                return dict(preregistration=path.as_posix(), preregistration_sha256=utility.sha(path))
            self.assertEqual(utility.validate_preregistration(request(value)), value)
            for key, bad in (('beta', .1), ('sampler_seed', 1), ('start_from', 'old-RL'), ('no_sft', False)):
                changed = dict(value, **{key: bad})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    utility.validate_preregistration(request(changed))
            changed = dict(value, optimizer=dict(value['optimizer'], updates=500))
            with self.assertRaises(ValueError):
                utility.validate_preregistration(request(changed))

    def test_checkpoint_namespace_and_contract_are_explicit(self):
        source = (HERE / 'residual.py').read_text(encoding='utf-8')
        self.assertNotEqual(residual.CONTRACT, 'kion-current-C1070-natural1024-H1-v1')
        self.assertIn("ROOT / 'policy' / 'residual-r1'", source)
        self.assertIn("actual_optimizer_updates=1000", source)
        self.assertIn("utility_formula=utility.FORMULA", source)
        self.assertIn("primary_contrast='fixed-latest1000'", source)

    def test_prefit_requires_both_bound_cohorts_and_same_prereg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = {}
            for cohort in ('train', 'screen'):
                path = root / cohort / 'utility.json'
                path.parent.mkdir()
                path.write_text(json.dumps(dict(contract=utility.CONTRACT, cohort=cohort,
                                                preregistration_sha256='p')), encoding='utf-8')
                pins[cohort] = dict(path=path.as_posix(), sha256=residual.sha(path))
            path = root / 'prefit.json'
            data = dict(contract=utility.CONTRACT, state='completed', both_cohorts_ready=True,
                        prefit_gate_passed=True, preregistration_sha256='p',
                        cohorts={'train': {'nondegenerate_prior': True}}, utility_manifests=pins)
            def request(data):
                path.write_text(json.dumps(data), encoding='utf-8')
                return dict(prefit_path=path.as_posix(), prefit_sha256=residual.sha(path),
                            preregistration_sha256='p', utility_dir=(root / 'train').as_posix())
            residual.validate_prefit(request(data), utility)
            changed = dict(data, utility_manifests={'train': pins['train']})
            with self.assertRaises(ValueError):
                residual.validate_prefit(request(changed), utility)
            changed = dict(data, cohorts={'train': {'nondegenerate_prior': False}})
            with self.assertRaises(ValueError):
                residual.validate_prefit(request(changed), utility)
            (root / 'screen' / 'utility.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                residual.validate_prefit(request(data), utility)


if __name__ == '__main__':
    unittest.main()
