"""Pure stdlib fake-render tests; never import NumPy/Matplotlib or real data."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('render_figures_v2.py')
spec = importlib.util.spec_from_file_location('native_figure_wrapper', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FAKE_PLOT = '''from pathlib import Path
import json
def validate(data):
    if data['candidate_width'] != 1024:
        raise ValueError('Wrong pool')
def render(data, output):
    output.mkdir(parents=True)
    result={'contract':'kion-scientific3-figures-v1','state':'completed','stage':data['stage'],'rows':data['rows'],'figures':[{'file':'fixture'}]}
    (output/'figures.json').write_text(json.dumps(result),encoding='utf-8')
    return result
'''


class NativeRenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.plot = self.root / 'interpretation' / 'plot_results_v3.py'
        self.plot.parent.mkdir()
        self.plot.write_text(FAKE_PLOT, encoding='utf-8', newline='\n')
        self.data = self.root / 'plot-data.json'
        self.value = {'contract': 'kion-scientific3-analysis-v1', 'state': 'completed', 'stage': 'screen',
                      'rows': 256, 'unique_users': 200, 'metric_denominator': 256, 'candidate_width': 1024}
        self.request = {'plot_module': {'path': self.plot.as_posix(), 'sha256': mod.sha(self.plot)},
                        'data': {}, 'source_hashes': {self.plot.as_posix(): mod.sha(self.plot)},
                        'input_hashes': {}, 'output': (self.root / 'reports' / 'figures-v2' / 'screen').as_posix()}
        self.refresh_data()
        self.events = []
        self.addCleanup(patch.stopall)
        patch.object(mod, 'ROOT', self.root).start()
        patch.object(mod, 'V3_PLOT_SOURCE', self.plot).start()
        # These are tiny stdlib objects, not imports or real numerical libraries.
        patch.dict(sys.modules, {'numpy': SimpleNamespace(__version__='fixture-numpy'),
                                 'matplotlib': SimpleNamespace(__version__='fixture-matplotlib')}).start()

    def refresh_data(self):
        self.data.write_text(json.dumps(self.value), encoding='utf-8')
        self.request['data'] = {'path': self.data.as_posix(), 'sha256': mod.sha(self.data)}
        self.request['input_hashes'] = {self.data.as_posix(): mod.sha(self.data)}

    def check(self, **fields):
        self.events.append(fields['stage'])

    def test_fake_render_records_actual_versions_pins_and_budget_order(self):
        result = mod.run(self.request, self.check)
        self.assertEqual(self.events, ['native-figure-input-integrity', 'native-figure-render-before', 'native-figure-render-after'])
        self.assertEqual(result['versions'], {'numpy': 'fixture-numpy', 'matplotlib': 'fixture-matplotlib'})
        self.assertEqual(result['data']['sha256'], mod.sha(self.data))
        self.assertEqual(result['plot_module']['sha256'], mod.sha(self.plot))
        self.assertEqual(result['figure_count'], 1)
        self.assertTrue(result['requires_visual_QA'])
        self.assertEqual(result['figures_index']['sha256'], mod.sha(Path(self.request['output']) / 'figures.json'))
        self.assertEqual(result['manifest']['sha256'], mod.sha(Path(self.request['output']) / 'native-render.json'))
        self.assertEqual(result['labels_artifact']['sha256'], mod.sha(Path(self.request['output']) / 'labels.json'))

    def test_full_cohort_has_separate_exact_directory(self):
        self.value.update(stage='full4096', rows=4096, metric_denominator=4096)
        self.refresh_data()
        with self.assertRaisesRegex(ValueError, 'cohort figure output'):
            mod.run(self.request, self.check)
        self.request['output'] = (self.root / 'reports' / 'figures-v2' / 'full').as_posix()
        self.assertEqual(mod.run(self.request, self.check)['stage'], 'full4096')

    def test_user_bindings_and_nonfinite_json_are_rejected_before_render(self):
        self.value['nested'] = {'user_id': [123]}
        self.refresh_data()
        with self.assertRaisesRegex(ValueError, 'Individual row/user'):
            mod.run(self.request, self.check)
        self.assertFalse(Path(self.request['output']).exists())
        del self.value['nested']
        self.value['invalid'] = float('nan')
        self.refresh_data()
        with self.assertRaisesRegex(ValueError, 'Nonfinite JSON'):
            mod.run(self.request, self.check)

    def test_source_and_input_pins_fail_closed(self):
        self.request['source_hashes'][self.plot.as_posix()] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'monitored binding'):
            mod.run(self.request, self.check)
        self.request['source_hashes'][self.plot.as_posix()] = mod.sha(self.plot)
        self.data.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'monitored binding'):
            mod.run(self.request, self.check)

    def test_nonroot_module_existing_or_external_output_and_missing_budget_fail(self):
        with self.assertRaisesRegex(ValueError, 'Budget callback'):
            mod.run(self.request)
        self.request['output'] = (self.root.parent / 'foreign').as_posix()
        with self.assertRaisesRegex(ValueError, 'root-local cohort'):
            mod.run(self.request, self.check)
        self.request['output'] = (self.root / 'reports' / 'figures-v2' / 'screen').as_posix()
        Path(self.request['output']).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'Fresh'):
            mod.run(self.request, self.check)

    def test_before_budget_failure_prevents_render(self):
        def budget(**fields):
            if fields['stage'] == 'native-figure-render-before':
                raise RuntimeError('fixture reserve')
        with self.assertRaisesRegex(RuntimeError, 'fixture reserve'):
            mod.run(self.request, budget)
        self.assertFalse(Path(self.request['output']).exists())

    def test_selected_title_join_preserves_exact_unicode_and_missing_ids(self):
        import csv
        items = self.root / 'items.csv'
        with items.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['item_id', 'ignored_private_fixture_field', 'title'])
            writer.writerow([1, 'not retained', 'Хрустальный'])
            writer.writerow([2, 'not retained', 'Название, с запятой\nи переносом'])
            writer.writerow([3, 'not retained', '   '])
            writer.writerow([999, 'not retained', 'Не выбранный фильм'])
        result = mod.title_lookup(items, {1, 2, 3, 4})
        self.assertEqual(result, {'1': 'Хрустальный', '2': 'Название, с запятой\nи переносом', '3': None, '4': None})
        with items.open('a', encoding='utf-8', newline='') as stream:
            csv.writer(stream).writerow([1, 'unused', 'Другое название'])
        with self.assertRaisesRegex(ValueError, 'Duplicate selected'):
            mod.title_lookup(items, {1})

    def test_title_enrichment_gets_separate_manifest_hash_and_source_pins(self):
        v2 = self.root / 'interpretation' / 'plot_results_v3.py'
        fake = FAKE_PLOT.replace('def render(data, output):', 'def render(data, output, labels=None):')
        fake += "\ndef needed_raw_ids(data):\n    return {1,2}\ndef format_label(raw_id,title):\n    return str(raw_id)+':'+str(title)\n"
        v2.write_text(fake, encoding='utf-8')
        patch.object(mod, 'V3_PLOT_SOURCE', v2).start()
        self.request['plot_module'] = {'path': v2.as_posix(), 'sha256': mod.sha(v2)}
        self.request['source_hashes'][v2.as_posix()] = mod.sha(v2)
        items = self.root / 'items.csv'
        items.write_text('item_id,title\n1,Хрустальный\n999,Не выбран\n', encoding='utf-8')
        self.request['items'] = {'path': items.as_posix(), 'sha256': mod.sha(items)}
        self.request['input_hashes'][items.as_posix()] = mod.sha(items)
        result = mod.run(self.request, self.check)
        self.assertEqual(result['data']['sha256'], mod.sha(self.data))
        enrichment = result['title_enrichment']
        self.assertEqual(enrichment['items']['sha256'], mod.sha(items))
        self.assertEqual([item['raw_id'] for item in enrichment['labels']], [1, 2])
        self.assertEqual([item['title'] for item in enrichment['labels']], ['Хрустальный', None])
        self.assertEqual([item['missing_title'] for item in enrichment['labels']], [False, True])
        self.assertEqual(len(enrichment['canonical_sha256']), 64)
        artifact = json.loads((Path(self.request['output']) / 'labels.json').read_text(encoding='utf-8'))
        self.assertEqual(artifact['title_enrichment'], enrichment)
        self.assertEqual(artifact['aggregate']['sha256'], mod.sha(self.data))
        self.assertEqual(artifact['aggregate_canonical_sha256'], result['aggregate_canonical_sha256'])

    def test_foreign_source_and_oversize_input_rejected_before_hash_or_render(self):
        foreign = self.root / 'foreign.py'
        foreign.write_text(FAKE_PLOT, encoding='utf-8')
        self.request['plot_module'] = {'path': foreign.as_posix(), 'sha256': mod.sha(foreign)}
        self.request['source_hashes'][foreign.as_posix()] = mod.sha(foreign)
        with self.assertRaisesRegex(ValueError, 'reviewed plot source'):
            mod.run(self.request, self.check)
        with patch.object(mod, 'MAX_JSON_BYTES', 1):
            with self.assertRaisesRegex(ValueError, 'compact input limit'):
                mod.run(self.request, self.check)

    def test_rollout_remains_owned_by_original_native_renderer(self):
        rollout = self.root / 'interpretation' / 'plot_rollout.py'
        fake = FAKE_PLOT.replace("'kion-scientific3-figures-v1'", "'kion-rollout-figures-v1'")
        fake = fake.replace("'stage':data['stage'],'rows':data['rows']", "'stage':'rollout','rows':data['position_count']")
        rollout.write_text(fake, encoding='utf-8')
        patch.object(mod, 'ROLLOUT_PLOT_SOURCE', rollout).start()
        self.request['plot_module'] = {'path': rollout.as_posix(), 'sha256': mod.sha(rollout)}
        self.request['source_hashes'][rollout.as_posix()] = mod.sha(rollout)
        self.value = {'contract': 'kion-final3-saved-liveprefix-summary-v1', 'state': 'completed',
                      'position_count': 16, 'trajectory_count': 192, 'candidate_width': 1024}
        self.refresh_data()
        self.request['output'] = (self.root / 'reports' / 'figures-v2' / 'rollout').as_posix()
        with self.assertRaisesRegex(ValueError, 'only screen/full'):
            mod.run(self.request, self.check)
        self.assertFalse(Path(self.request['output']).exists())

    def test_source_imports_only_stdlib_and_never_numerical(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(names & {'numpy', 'matplotlib', 'torch', 'subprocess', 'requests'})
        self.assertNotIn('sys.path', SOURCE.read_text(encoding='utf-8'))


V2_SOURCE = SOURCE.with_name('plot_results_v3.py')
v2spec = importlib.util.spec_from_file_location('title_adapter_fixture', V2_SOURCE)
v2 = importlib.util.module_from_spec(v2spec)
v2spec.loader.exec_module(v2)


class TitleAdapterTests(unittest.TestCase):
    @staticmethod
    def aggregate():
        exposures = {arm: {'exposure': {weighting: {'sorted_top10_exposure': [
            {'raw_id': 5, 'share': .7}, {'raw_id': 1, 'share': .3}]}
            for weighting in ('query_weighted', 'equal_user_weighted')}} for arm in v2.ARMS}
        return {'stage': 'full4096', 'summaries': exposures}

    def test_display_is_bounded_but_preserves_raw_id_and_missing_status(self):
        title = 'Очень длинное русское название фильма, которое не должно перекрывать соседние подписи на графике'
        label = v2.format_label(10440, title)
        self.assertTrue(label.startswith('10440 · '))
        self.assertLessEqual(len(label.splitlines()), 2)
        self.assertTrue(all(len(line) <= 34 for line in label.splitlines()))
        self.assertIn('…', label)
        self.assertEqual(v2.format_label(9, None), '9 · название отсутствует')

    def test_only_originally_plotted_ids_are_requested(self):
        self.assertEqual(v2.needed_raw_ids(self.aggregate()), {1, 5})

    def test_portable_labels_need_no_catalog_and_bind_original_data_and_title_hash(self):
        data = self.aggregate()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            data_path, labels_path = folder / 'data.json', folder / 'labels.json'
            data_path.write_text(json.dumps(data), encoding='utf-8')
            enrichment = {'enabled': True, 'items': {'path': 'unavailable/public/items.csv', 'sha256': 'a' * 64},
                          'requested_count': 2, 'found_title_count': 1,
                          'labels': [{'raw_id': 1, 'title': None, 'missing_title': True, 'display_label': 'unused'},
                                     {'raw_id': 5, 'title': 'Полное русское название', 'missing_title': False, 'display_label': 'unused'}]}
            enrichment['canonical_sha256'] = v2.canonical_sha(enrichment)
            artifact = {'contract': 'kion-plot-title-labels-v1', 'state': 'completed', 'stage': data['stage'],
                        'aggregate': {'sha256': hashlib.sha256(data_path.read_bytes()).hexdigest()},
                        'aggregate_canonical_sha256': v2.canonical_sha(data), 'title_enrichment': enrichment}
            labels_path.write_text(json.dumps(artifact), encoding='utf-8')
            self.assertEqual(v2.load_labels(labels_path, data, data_path), {'1': None, '5': 'Полное русское название'})
            artifact['title_enrichment']['labels'][1]['title'] = 'Подмена'
            labels_path.write_text(json.dumps(artifact), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'enrichment SHA'):
                v2.load_labels(labels_path, data, data_path)
            artifact['title_enrichment']['labels'][1]['title'] = 'Полное русское название'
            labels_path.write_text(json.dumps(artifact), encoding='utf-8')
            data_path.write_text(data_path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'byte SHA'):
                v2.load_labels(labels_path, data, data_path)

    def test_adapter_reuses_frozen_numerical_code_without_importing_plotting(self):
        self.assertEqual(hashlib.sha256(v2.BASE_SOURCE.read_bytes()).hexdigest(), v2.BASE_SHA256)
        tree = ast.parse(V2_SOURCE.read_text(encoding='utf-8'))
        names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(names & {'numpy', 'matplotlib', 'torch'})
        self.assertFalse(any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Subscript) and
                             any(isinstance(part, ast.Name) and part.id == 'data' for part in ast.walk(target)) for target in node.targets)
                             for node in ast.walk(tree)))
        self.assertIn('base.render(data, output)', V2_SOURCE.read_text(encoding='utf-8'))
        self.assertIn("parser.add_argument('--labels'", V2_SOURCE.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
