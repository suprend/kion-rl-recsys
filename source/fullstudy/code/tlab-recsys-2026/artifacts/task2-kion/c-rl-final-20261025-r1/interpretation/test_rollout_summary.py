"""Pure JSON metadata fixtures; no real arrays, providers, NumPy or torch."""
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_final_rollout_summary_tests', HERE/'rollout_summary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture():
    document = dict(contract='kion-final3-liveprefix-grid-v1', state='completed', arms=list(summary.HEADS),
        positions=summary.POSITIONS[:], seed=20261024, five_step_training=False, synthetic_prefix_only=True,
        RM_observed_s0_fixed=True, clock='observed-s0-fixed', observed_history='observed-s0-fixed',
        results=[], RM_greedy_controls=[], original_val_rows=summary.ORIGINAL_ROWS[:], trajectory_count=192, step_count=960,
        cache_sha256='a'*64, reward_sha256='b'*64, utility_sha256='c'*64,
        source={'path':'fixture/rollout.py','sha256':'d'*64}, selection_pin={'path':'fixture/selection.json','sha256':'e'*64},
        preregistration={'path':'fixture/preregistration.json','sha256':'f'*64}, scientific_checkpoints={
            'C':{'path':'fixture/C/head.pt','sha256':'0'*64,'step':None},
            **{arm:dict(path='fixture/'+arm+'/latest.pt', sha256=str(j)*64, step=1000, updates=1000,
                       C_checkpoint_sha256='0'*64, training={'path':'fixture/'+arm+'/training.json','sha256':str(j+2)*64})
               for j,arm in enumerate(summary.HEADS[1:], start=1)}})
    for position in summary.POSITIONS:
        ids = [100+position*20+j for j in range(5)]
        raws = [token+1000 for token in ids]
        for arm in summary.HEADS:
            for mode in summary.MODES:
                for mask in summary.MASKS:
                    steps = []
                    for j, (token, raw) in enumerate(zip(ids, raws)):
                        steps.append(dict(step=j+1, generated_token_id=token, generated_raw_id=raw,
                            candidate_index=j, generated_repeat=False, original_seen=False, selected_supported=True,
                            producer_size=2, producer_slot=1, prefix_length=j+1, encoder_trim_width=max(3,j+1),
                            retrieval_projection='original-full-vocab-then-fixed-pool-gather',
                            initial_C_cache_parity=(dict(atol=.003, rtol=.001, baseline='fresh-live-C-on-identical-prefix',
                                score_tolerance_passed=True, rank_tolerance_passed=True,
                                fresh_reference_maxabs=0.,
                                top10_exact=True, max_abs_error=0.) if j == 0 else None),
                            q_l2_drift=0. if j == 0 else .1, base_rmse_drift=0. if j == 0 else .1,
                            cached_q_maxabs=.01, cached_base_maxabs=.01, selected_probability=.2, selected_base_logit=-.4,
                            kl_nats=0., entropy_nats=1., supported_mass=1., unsupported_mass=0.,
                            expected_supported_reward_conditional=.6, expected_neutral_proxy=0.,
                            expected_supported_residual_conditional=.3, expected_supported_prior_conditional=.3,
                            expected_neutral_residual_proxy=0., selected_reward=.6, selected_neutral_proxy=0.,
                            selected_residual=.3, selected_fit_prior=.3, selected_neutral_residual_proxy=0.))
                    total = math.fsum(.9**j*.6 for j in range(5))
                    residual = math.fsum(.9**j*.3 for j in range(5))
                    result = dict(contract='kion-final3-liveprefix-producer-shape-fixed-observed-rm-v1',
                        scope='synthetic-id-slate-fixed-observed-rm', selection=mode, repeat_mask=mask, steps=5,
                        seed=20261024+position, gamma=.9, temperature=1., rm_state='observed-s0-fixed',
                        raw1068='observed-s0-fixed', source13='observed-s0-fixed', source12='live-prefix-gpt-logit',
                        frozen_tensor_versions_unchanged=True, context_policy='BOS-plus-latest63-real-or-UNK-tokens',
                        encoder_peers='original-observed-s0-except-active-synthetic-prefix',
                        retrieval_projection='original-full-vocab-then-fixed-pool-gather', producer_size=2, producer_slot=1,
                        generated_token_ids=ids[:], generated_raw_ids=raws[:], per_step=steps,
                        initial_reference_supported_baseline=.6, initial_reference_supported_residual_baseline=.3,
                        unique_generated=5, generated_repeat_count=0, unsupported_selected=0,
                        fixed_neutral_proxy_return=0., supported_completion_partial_sum=total,
                        fixed_completion_proxy_return=total, fixed_neutral_residual_proxy_return=0.,
                        supported_residual_partial_sum=residual, fixed_residual_proxy_return=residual,
                        final_prefix=[1,*ids])
                    document['results'].append(dict(position=position, arm=arm, selection=mode, repeat_mask=mask, result=result))
        document['RM_greedy_controls'].append(dict(position=position, generated_ids=ids[:], generated_raw_ids=raws[:],
            rewards=[.6]*5, supported=[True]*5))
    return document


class SummaryTests(unittest.TestCase):
    def test_complete_crossproduct_summary_and_named_cases_are_prechosen(self):
        document = fixture()
        document['results'].reverse()
        result = summary.summarize(document, summary.POSITIONS, titles={6160: 'Fixture title'})
        self.assertEqual(result['trajectory_count'], 192)
        self.assertEqual(len(result['group_summaries']), 12)
        self.assertEqual(len(result['matched_C_comparisons']), 8)
        self.assertEqual([row['position'] for row in result['examples']], summary.POSITIONS[:3])
        self.assertTrue(all(len(row['slates']) == 12 for row in result['examples']))
        self.assertEqual(len(result['group_summaries'][0]['per_step']), 5)
        self.assertEqual(result['group_summaries'][0]['per_step'][0]['exposure']['recommendation_slots'], 16)

    def test_missing_duplicate_wrong_arm_and_old_schema_are_rejected(self):
        for change in (lambda d: d['results'].pop(),
                       lambda d: d['results'].__setitem__(0, copy.deepcopy(d['results'][1])),
                       lambda d: d['results'][0].__setitem__('arm', 'selected'),
                       lambda d: d.__setitem__('contract', 'old')):
            document = fixture()
            change(document)
            with self.assertRaises(ValueError):
                summary.validate_document(document, summary.POSITIONS)

    def test_nonfinite_seed_masks_and_support_arithmetic_are_rejected(self):
        mutations = [('kl_nats', math.nan), ('kl_nats', 9.), ('entropy_nats', -1.), ('entropy_nats', 100.), ('supported_mass', .8),
                     ('expected_neutral_proxy', .2), ('expected_neutral_residual_proxy', .2),
                     ('selected_fit_prior', .4), ('selected_probability', 0.), ('encoder_trim_width', 0),
                     ('initial_C_cache_parity', None)]
        for field, value in mutations:
            document = fixture()
            document['results'][0]['result']['per_step'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summary.validate_document(document, summary.POSITIONS)

    def test_provenance_missing_checkpoint_alias_and_counts_rejected(self):
        for field in ('source','scientific_checkpoints','selection_pin','preregistration','original_val_rows'):
            document = fixture()
            document.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                summary.validate_document(document, summary.POSITIONS)
        for field, value in (('trajectory_count',191), ('step_count',959), ('cache_sha256','broken')):
            document = fixture()
            document[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summary.validate_document(document, summary.POSITIONS)
        document = fixture()
        document['scientific_checkpoints']['RL-residual']['sha256'] = document['scientific_checkpoints']['RL-completion']['sha256']
        with self.assertRaises(ValueError):
            summary.validate_document(document, summary.POSITIONS)

    def test_selected_probability_cannot_exceed_supported_mass(self):
        document = fixture()
        step = document['results'][0]['result']['per_step'][0]
        step.update(supported_mass=.1, unsupported_mass=.9)
        with self.assertRaises(ValueError):
            summary.validate_document(document, summary.POSITIONS)

    def test_run_requires_named_catalog_and_writes_reviewable_artifacts(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix='rollout-fixture-') as temporary:
            root = Path(temporary)
            source = root/'rollout.json'
            document = fixture()
            source.write_text(json.dumps(document, allow_nan=False), encoding='utf-8')
            request = dict(rollout={'path':source.as_posix(),'sha256':summary.sha(source)},
                expected_positions=summary.POSITIONS[:], output=(root/'output').as_posix())
            with self.assertRaises(ValueError):
                summary.run(request)
            catalog = root/'items.csv'
            with catalog.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['item_id','title'])
                writer.writeheader()
                for control in document['RM_greedy_controls']:
                    for raw in control['generated_raw_ids']:
                        writer.writerow(dict(item_id=raw,title='Synthetic title '+str(raw)))
            request['items'] = {'path':catalog.as_posix(),'sha256':summary.sha(catalog)}
            result = summary.run(request)
            self.assertEqual(result['trajectory_count'],192)
            self.assertTrue((root/'output/rollout-summary.json').is_file())
            self.assertTrue((root/'output/REPORT.md').is_file())
            self.assertTrue(result['examples'][0]['slates'][0]['items'][0]['title'])
        document = fixture()
        document['results'][0]['result']['seed'] += 1
        with self.assertRaises(ValueError):
            summary.validate_document(document, summary.POSITIONS)

    def test_frozen_scope_prefix_and_returns_are_checked(self):
        for field, value in [('source12', 'cached'), ('encoder_peers', 'rebatched'), ('final_prefix', [1,2]),
                             ('fixed_neutral_residual_proxy_return', .1), ('fixed_completion_proxy_return', None)]:
            document = fixture()
            document['results'][0]['result'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summary.validate_document(document, summary.POSITIONS)

    def test_repeat_mask_and_h63_binding_checked(self):
        document = fixture()
        document['results'][0]['result']['per_step'][0]['original_seen'] = True
        with self.assertRaises(ValueError):
            summary.validate_document(document, summary.POSITIONS)
        document = fixture()
        document['results'][0]['result']['per_step'][1]['generated_repeat'] = True
        with self.assertRaises(ValueError):
            summary.validate_document(document, summary.POSITIONS)

    def test_old_positions_cannot_be_reordered_or_changed(self):
        with self.assertRaises(ValueError):
            summary.validate_document(fixture(), list(reversed(summary.POSITIONS)))


if __name__ == '__main__':
    unittest.main()
