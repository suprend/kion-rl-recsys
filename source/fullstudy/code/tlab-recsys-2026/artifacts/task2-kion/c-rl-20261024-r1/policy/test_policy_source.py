"""Stdlib-only policy contract tests; numerical smoke remains root-owned."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('policy_source_subject', HERE/'policy.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SourceTests(unittest.TestCase):
    def test_legacy_context_adapter_preserves_outer_authorization(self):
        calls, result = [], object()
        class FakeBank:
            def open_context(self, request, check):
                calls.append((request, check))
                return result
        check = lambda **fields: None
        for outer_stage in ('prepare_cache', 'evaluate_selected', 'rollout'):
            outer = dict(stage=outer_stage, allow_validation_graph_build=False,
                         validation_graph_cache='E:/unchanged/existing/validation-graph',
                         source_hashes={'old-bank.py':'original-pin'}, operation='root-signed-operation')
            before = dict(outer)
            self.assertIs(m.open_frozen_context(FakeBank(),outer,check),result)
            inner, received_check = calls[-1]
            self.assertIsNot(inner,outer)
            self.assertIs(received_check,check)
            self.assertEqual(outer,before)
            self.assertEqual(inner,dict(outer,stage='evaluate',allow_validation_graph_build=False))
            self.assertIs(inner['source_hashes'],outer['source_hashes'])

    def test_legacy_context_rejects_any_build_authorization(self):
        class ForbiddenBank:
            def open_context(self,*args):
                self.fail('Reader must not be opened')
        for request in ({'validation_graph_cache':'same/path','allow_validation_graph_build':True},
                        {'allow_validation_graph_build':False}):
            with self.assertRaises(ValueError):
                m.open_frozen_context(ForbiddenBank(),request,lambda **kw:None)

    def test_all_heavy_context_calls_use_adapter(self):
        tree = ast.parse((HERE/'policy.py').read_text(encoding='utf-8'))
        for name in ('prepare_cache','evaluate_selected','rollout'):
            function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
            calls = [node for node in ast.walk(function) if isinstance(node,ast.Call)]
            self.assertEqual(sum(isinstance(node.func,ast.Name) and node.func.id=='open_frozen_context' for node in calls),1)
            self.assertFalse(any(isinstance(node.func,ast.Attribute) and node.func.attr=='open_context' for node in calls))

    def test_no_numeric_import_side_effect(self):
        tree = ast.parse((HERE/'policy.py').read_text(encoding='utf-8'))
        imports = [node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(node) or 'numpy' in ast.unparse(node) for node in imports))

    def test_preregistration_actual_schema(self):
        value = json.loads((HERE.parent/'preregistration.json').read_text(encoding='utf-8'))
        normalized = m.normalized_preregistration(value)
        self.assertEqual((normalized['updates'],normalized['batch_size'],normalized['weight_decay']), (1000,16,0))
        self.assertEqual(normalized['learning_rate'],1e-5)
        self.assertEqual(normalized['gates']['KL_max'],.05)

    def test_beta_selection_and_tie(self):
        prereg = m.normalized_preregistration(json.loads((HERE.parent/'preregistration.json').read_text(encoding='utf-8')))
        baseline = dict(map10=.1,exact_KL=0.,unsupported_mass=.1,top1_max_share=.1,expected_effective_reward=0.)
        good = dict(baseline,expected_effective_reward=.01,exact_KL=.01)
        frontier = {'arms':{'C':baseline,'tilt-beta0.1':good,'tilt-beta0.5':good}}
        self.assertEqual(m.select_beta(prereg,frontier),.5)
        frontier['arms']['tilt-beta0.1'] = dict(good,expected_effective_reward=.02)
        self.assertEqual(m.select_beta(prereg,frontier),.1)
        frontier['arms']['tilt-beta0.1']['map10'] = .05
        self.assertEqual(m.select_beta(prereg,frontier),.5)

    def test_no_feasible_beta_conservative(self):
        prereg = m.normalized_preregistration(json.loads((HERE.parent/'preregistration.json').read_text(encoding='utf-8')))
        b = dict(map10=.1,exact_KL=0.,unsupported_mass=.1,top1_max_share=.1,expected_effective_reward=0.)
        bad = dict(b,exact_KL=1.)
        self.assertEqual(m.select_beta(prereg,{'arms':{'C':b,'tilt-beta0.1':bad,'tilt-beta0.5':bad}}),.5)

    def test_raw_id_ties(self):
        self.assertEqual(m.rank([3,4,5],[1.,1.,0.],[20,10,5]),[1,0,2])

    def test_date_group_scatter(self):
        batches = list(m.groups([5,7,2], {5:3,7:2,2:3}, maximum=1))
        self.assertEqual(batches,[[(1,7)],[(0,5)],[(2,2)]])

    def test_no_forced_target_candidate_api(self):
        tree = ast.parse((HERE/'policy.py').read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node,ast.Call)]
        self.assertFalse(any(isinstance(node.func,ast.Attribute) and node.func.attr == 'sample' for node in calls))

    def test_no_supervised_loss(self):
        source = (HERE/'policy.py').read_text(encoding='utf-8')
        self.assertNotIn('cross_entropy(',source)
        self.assertNotIn('AdamW(',source)

    def test_checkpoint_contract(self):
        self.assertEqual(m.STEPS,(0,250,500,1000))
        self.assertEqual(m.SOURCE,['current_detached_gpt_logit','log1p_strict7d_count'])

    def test_screen_clock_bound_to_selection(self):
        source = (HERE/'policy.py').read_text(encoding='utf-8')
        self.assertNotIn("origins = candidates.origins if split == 'train' else data['target_date']",source)
        self.assertIn("selection['states'][name]",source)


if __name__ == '__main__':
    unittest.main()
