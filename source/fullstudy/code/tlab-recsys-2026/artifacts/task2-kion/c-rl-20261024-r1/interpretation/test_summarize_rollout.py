"""Tiny generated-JSON tests only; never open actual rollout/catalog/model payloads."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name('summarize_rollout.py')
spec = importlib.util.spec_from_file_location('rollout_summary', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture():
    positions = list(range(16))
    document = {'contract': 'toy-policy', 'state': 'completed', 'five_step_training': False,
                'synthetic_prefix_only': True, 'RM_observed_s0_fixed': True,
                'cache_sha256': 'toy-cache', 'reward_sha256': 'toy-reward', 'results': [], 'RM_greedy_controls': []}
    for position in positions:
        control_tokens = [3, 5, 6, 7, 8] if position == 0 else [3, 4, 5, 6, 7]
        document['RM_greedy_controls'].append({'position': position, 'generated_ids': control_tokens,
            'generated_raw_ids': [token+100 for token in control_tokens], 'rewards': [.5]*5, 'supported': [True]*5})
        for head in mod.HEADS:
            for mode in mod.MODES:
                for mask in mod.MASKS:
                    tokens = [3, 3, 4, 5, 6] if mask == 'none' else [3, 4, 5, 6, 7]
                    steps = []
                    for j, token in enumerate(tokens):
                        ok = not (position == 0 and token == 4)
                        steps.append({'step': j+1, 'candidate_index': token-3, 'generated_token_id': token,
                            'generated_raw_id': token+100, 'generated_repeat': token in tokens[:j],
                            'original_seen': token == 3, 'selected_supported': ok, 'selected_reward': .5 if ok else None,
                            'selected_neutral_proxy': .3 if ok else 0., 'kl_nats': .1 if head == 'latest' else 0.,
                            'entropy_nats': 1., 'supported_mass': .8, 'unsupported_mass': .2,
                            'expected_supported_reward_conditional': .5, 'expected_neutral_proxy': .24})
                    partial = sum(.9**j*step['selected_reward'] for j, step in enumerate(steps) if step['selected_supported'])
                    result = {'contract': 'kion-c3-liveprefix-fixed-observed-rm-v1',
                        'scope': 'synthetic-id-slate-fixed-observed-rm', 'selection': mode, 'repeat_mask': mask,
                        'steps': 5, 'seed': 20261024+position, 'gamma': .9, 'temperature': 1.,
                        'rm_state': 'observed-s0-fixed', 'raw1068': 'observed-s0-fixed', 'source13': 'observed-s0-fixed',
                        'source12': 'live-prefix-gpt-logit', 'frozen_tensor_versions_unchanged': True,
                        'generated_token_ids': tokens, 'generated_raw_ids': [token+100 for token in tokens],
                        'per_step': steps, 'initial_reference_supported_baseline': .2,
                        'unique_generated': len(set(tokens)), 'generated_repeat_count': 5-len(set(tokens)),
                        'unsupported_selected': sum(not step['selected_supported'] for step in steps),
                        'fixed_neutral_proxy_return': sum(.9**j*step['selected_neutral_proxy'] for j, step in enumerate(steps)),
                        'fixed_completion_proxy_return': partial if all(step['selected_supported'] for step in steps) else None,
                        'supported_completion_partial_sum': partial}
                    document['results'].append({'position': position, 'head': head, 'selection': mode,
                                                'repeat_mask': mask, 'result': result})
    return document, positions


class RolloutSummaryTests(unittest.TestCase):
    def test_complete_grid_and_support_denominator(self):
        doc, positions = fixture()
        result = mod.summarize(doc, positions)
        self.assertEqual(result['trajectory_count'], 192)
        self.assertEqual(len(result['group_summaries']), 12)
        self.assertEqual(len(result['matched_C_comparisons']), 8)
        row = result['group_summaries'][0]
        self.assertEqual(row['mean_unique_generated'], 4)
        self.assertEqual(row['mean_generated_repeat_count'], 1)
        self.assertAlmostEqual(row['original_seen_fraction'], .4)
        self.assertEqual(row['fixed_completion_proxy_return_all_supported_only']['denominator'], 15)
        self.assertEqual(row['per_step'][0]['kl_nats']['denominator'], 16)
        self.assertEqual(result['RM_greedy_unique5_control']['mean_unique_generated'], 5)
        self.assertEqual(result['RM_greedy_unique5_control']['completion_proxy_return_all_supported_only']['denominator'], 16)

    def test_identical_repeated_slates_have_full_multiset_overlap(self):
        doc, positions = fixture()
        result = mod.summarize(doc, positions)
        comparison = result['matched_C_comparisons'][0]
        self.assertEqual(comparison['mean_multiset_slate_overlap_fraction'], 1.)
        self.assertEqual(comparison['exact_sequence_fraction'], 1.)
        row = result['group_summaries'][0]
        self.assertAlmostEqual(row['exposure']['slate_slot_hhi'], .4**2+3*.2**2)
        self.assertEqual(row['exposure']['top1_max_item_share'], 1.)

    def test_examples_use_explicit_first_three_before_outcomes(self):
        doc, positions = fixture()
        positions = list(reversed(positions))
        doc['results'].reverse()
        result = mod.summarize(doc, positions, titles={103: 'Fixture title'})
        self.assertEqual([example['position'] for example in result['examples']], [15, 14, 13])
        self.assertEqual(len(result['examples'][0]['slates']), 12)
        self.assertEqual(result['examples'][0]['slates'][0]['items'][0], {'raw_id': 103, 'title': 'Fixture title'})

    def test_partial_duplicate_and_unexpected_contexts_rejected(self):
        doc, positions = fixture()
        doc['results'].pop()
        with self.assertRaisesRegex(ValueError, 'Exactly192'):
            mod.summarize(doc, positions)
        doc, positions = fixture()
        doc['results'][-1] = copy.deepcopy(doc['results'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate/unexpected'):
            mod.summarize(doc, positions)
        doc, positions = fixture()
        doc['results'][0]['head'] = 'surprise'
        with self.assertRaisesRegex(ValueError, 'Duplicate/unexpected'):
            mod.summarize(doc, positions)

    def test_inner_binding_and_partial_steps_rejected(self):
        doc, positions = fixture()
        doc['results'][0]['result']['selection'] = 'categorical'
        with self.assertRaisesRegex(ValueError, 'Inner'):
            mod.summarize(doc, positions)
        doc, positions = fixture()
        doc['results'][0]['result']['per_step'].pop()
        with self.assertRaisesRegex(ValueError, 'Exactly5'):
            mod.summarize(doc, positions)

    def test_support_and_discounted_arithmetic_rejected(self):
        doc, positions = fixture()
        doc['results'][0]['result']['fixed_completion_proxy_return'] = 2.
        with self.assertRaisesRegex(ValueError, 'all selected'):
            mod.summarize(doc, positions)
        doc, positions = fixture()
        doc['results'][0]['result']['fixed_neutral_proxy_return'] += .01
        with self.assertRaisesRegex(ValueError, 'arithmetic'):
            mod.summarize(doc, positions)
        doc, positions = fixture()
        doc['results'][0]['result']['per_step'][0]['kl_nats'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            mod.summarize(doc, positions)

    def test_rm_control_unique_and_complete(self):
        doc, positions = fixture()
        doc['RM_greedy_controls'][0]['generated_ids'][1] = 3
        with self.assertRaisesRegex(ValueError, 'unique'):
            mod.summarize(doc, positions)
        doc, positions = fixture()
        doc['RM_greedy_controls'][-1]['position'] = 0
        with self.assertRaisesRegex(ValueError, 'Duplicate/unexpected RM'):
            mod.summarize(doc, positions)

    def test_fixed_RM_and_expected_neutral_bindings(self):
        doc, positions = fixture()
        result = doc['results'][4]['result']
        result['per_step'][0]['selected_reward'] = .6
        result['per_step'][0]['selected_neutral_proxy'] = .4
        result['fixed_neutral_proxy_return'] += .1
        result['supported_completion_partial_sum'] += .1
        with self.assertRaisesRegex(ValueError, 'Fixed observed-s0 RM'):
            mod.summarize(doc, positions)
        doc, positions = fixture()
        doc['results'][0]['result']['per_step'][0]['expected_neutral_proxy'] = .1
        with self.assertRaisesRegex(ValueError, 'Expected neutral proxy arithmetic'):
            mod.summarize(doc, positions)

    def test_stdlib_only_imports(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({'numpy', 'torch', 'catboost', 'shap', 'subprocess'} & {a.name for a in node.names})
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, ('numpy', 'torch', 'catboost', 'shap', 'subprocess'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
