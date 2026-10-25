"""Stdlib-only synthetic package replay tests; fake rendering, no numerical imports."""
import hashlib
import ast
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('replay_portable_v2.py')
_spec = importlib.util.spec_from_file_location('packaged_replay', SOURCE)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
VERIFIER = SOURCE.parents[1] / 'reports/code/tools/verify_bundle.py'

FAKE_PLOT = '''from pathlib import Path
import hashlib,json
BASE_SOURCE=Path(__file__).with_name('plot_results.py')
DEPENDENCIES={BASE_SOURCE.as_posix():hashlib.sha256(BASE_SOURCE.read_bytes()).hexdigest()}
def validate(data):
    if data['candidate_width']!=1024:raise ValueError('Wrong candidate pool')
def needed_raw_ids(data):return {1}
def load_labels(path,data,input_path=None):return json.loads(Path(path).read_text(encoding='utf-8'))['labels']
def render(data,output,labels=None):
    output.mkdir(parents=True)
    result={'contract':'kion-scientific3-figures-v1','state':'completed','stage':data['stage'],'rows':data['rows'],
            'figures':[{'file':'synthetic','formats':['svg','png']} ]}
    (output/'synthetic.svg').write_text('<svg>'+labels['1']+'</svg>',encoding='utf-8')
    (output/'synthetic.png').write_bytes(b'invented fixture, not an actual PNG')
    (output/'figures.json').write_text(json.dumps(result),encoding='utf-8')
    return result
'''


class PackagedReplay(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.report = self.root / 'reports'
        self.code = self.report / 'code'
        self.source_dir = self.code / 'tlab-recsys-2026' / mod.SOURCE_PREFIX
        self.source_dir.mkdir(parents=True)
        self.verifier = self.code / 'tools/verify_bundle.py'
        self.verifier.parent.mkdir()
        self.verifier.write_bytes(VERIFIER.read_bytes())
        (self.source_dir / SOURCE.name).write_bytes(SOURCE.read_bytes())
        self.base = self.source_dir / 'plot_results.py'
        self.base.write_text('# Synthetic inert base\n', encoding='utf-8')
        self.plot = self.source_dir / 'plot_results_v3.py'
        self.plot.write_text(FAKE_PLOT, encoding='utf-8')
        fixture = self.code / 'fixtures/synthetic-ranking.json'
        fixture.parent.mkdir()
        fixture.write_text(json.dumps({'synthetic': True, 'target_ranks': [1,2,None],
            'expected': {'queries':3,'map10':.5,'recall10':2/3,'ndcg10':.5436432511904858}}), encoding='utf-8')
        self.evidence = self.report / 'evidence'
        self.evidence.mkdir()
        self.data = self.evidence / 'aggregate.json'
        self.labels = self.evidence / 'labels.json'
        self.value = {'contract':'kion-scientific3-analysis-v1','state':'completed','stage':'screen',
                      'rows':256,'candidate_width':1024}
        self.data.write_text(json.dumps(self.value), encoding='utf-8')
        self.labels.write_text(json.dumps({'labels':{'1':'Вымышленный фильм'}}), encoding='utf-8')
        self.manifest = self.report / 'code-manifest.json'
        self.request = {'report_root':self.report.as_posix(), 'source_hashes':{},'input_hashes':{},
                        'output':(self.report/'replay-figures/screen').as_posix(),
                        'supervised_updates':0,'rl_updates':0,'new_rm_fits':0,'final_test_reads':0}
        self.request['source_hashes'][SOURCE.resolve().as_posix()] = mod.sha(SOURCE)
        self.refresh_manifest()
        for field, path, binding in (('bundle_manifest',self.manifest,'input_hashes'),('verifier',self.verifier,'source_hashes'),
                                    ('plot_module',self.plot,'source_hashes'),('data',self.data,'input_hashes'),
                                    ('labels',self.labels,'input_hashes')):
            self.bind(field,path,binding)
        self.request['source_hashes'][self.base.as_posix()] = mod.sha(self.base)
        self.events = []
        self.addCleanup(patch.stopall)
        patch.object(mod,'ROOT',self.root).start()
        patch.dict(sys.modules, {'numpy':SimpleNamespace(__version__='synthetic-numpy'),
                                 'matplotlib':SimpleNamespace(__version__='synthetic-matplotlib')}).start()

    def bind(self, field, path, binding):
        self.request[field] = {'path':path.as_posix(),'sha256':mod.sha(path)}
        self.request[binding][path.as_posix()] = mod.sha(path)

    def refresh_manifest(self):
        files = [{'bundle_path':p.relative_to(self.report).as_posix(),'sha256':mod.sha(p),'bytes':p.stat().st_size}
                 for p in self.code.rglob('*') if p.is_file()]
        self.manifest.write_text(json.dumps({'contract':'kion-small-source-bundle-v1','files':files,
            'totals':{'files':len(files),'bytes':sum(row['bytes'] for row in files)}}), encoding='utf-8')
        if hasattr(self,'request') and 'bundle_manifest' in self.request:
            self.bind('bundle_manifest',self.manifest,'input_hashes')

    def check(self, **fields):
        self.events.append(fields['stage'])

    def test_replays_packaged_sources_with_unicode_titles_and_receipt(self):
        result = mod.run(self.request,self.check)
        self.assertEqual(self.events,['packaged-replay-validate','packaged-replay-render-before','packaged-replay-render-after'])
        self.assertEqual(result['titles_present'],1)
        self.assertEqual(result['versions']['numpy'],'synthetic-numpy')
        self.assertEqual(result['package_verification']['state'],'passed')
        self.assertEqual(len(result['files']),2)
        self.assertIn('Вымышленный фильм',(Path(result['output'])/'synthetic.svg').read_text(encoding='utf-8'))
        self.assertEqual(result['receipt']['sha256'],mod.sha(Path(result['output'])/'replay.json'))

    def test_rejects_corrupt_code_before_renderer_import(self):
        self.base.write_text('changed = True\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'mismatch'):
            mod.run(self.request,self.check)
        self.assertFalse(Path(self.request['output']).exists())

    def test_rejects_original_data_path_outside_package(self):
        outside = self.root/'original.json'
        outside.write_bytes(self.data.read_bytes())
        self.bind('data',outside,'input_hashes')
        with self.assertRaisesRegex(ValueError,'small packaged input'):
            mod.run(self.request,self.check)

    def test_requires_titles_and_exact_plotted_scope(self):
        for labels in ({'1':None},{'1':'Фильм','2':'Лишний фильм'},None):
            self.labels.write_text(json.dumps({'labels':labels}),encoding='utf-8')
            self.bind('labels',self.labels,'input_hashes')
            with self.subTest(labels=labels),self.assertRaisesRegex(ValueError,'titles are absent|Title map differs'):
                mod.run(self.request,self.check)
        self.assertFalse(Path(self.request['output']).exists())

    def test_forbids_user_records_nonfinite_json_and_ml_modules(self):
        self.value['nested']={'user_id':[123]}
        self.data.write_text(json.dumps(self.value),encoding='utf-8')
        self.bind('data',self.data,'input_hashes')
        with self.assertRaisesRegex(ValueError,'Individual user/query'):
            mod.run(self.request,self.check)
        del self.value['nested']
        self.value['invalid']=float('nan')
        self.data.write_text(json.dumps(self.value),encoding='utf-8')
        self.bind('data',self.data,'input_hashes')
        with self.assertRaisesRegex(ValueError,'Nonfinite JSON'):
            mod.run(self.request,self.check)
        with patch.dict(sys.modules,{'torch':SimpleNamespace()}),self.assertRaisesRegex(ValueError,'ML module present'):
            mod.run(self.request,self.check)

    def test_rejects_wrong_output_existing_output_and_missing_budget(self):
        with self.assertRaisesRegex(ValueError,'Budget callback'):
            mod.run(self.request)
        self.request['output']=(self.report/'figures/screen').as_posix()
        with self.assertRaisesRegex(ValueError,'Fresh native cohort'):
            mod.run(self.request,self.check)
        self.request['output']=(self.report/'replay-figures/screen').as_posix()
        Path(self.request['output']).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'Fresh native cohort'):
            mod.run(self.request,self.check)

    def test_pins_base_dependency_and_packaged_self(self):
        del self.request['source_hashes'][self.base.as_posix()]
        with self.assertRaisesRegex(ValueError,'monitored binding'):
            mod.run(self.request,self.check)
        self.request['source_hashes'][self.base.as_posix()]=mod.sha(self.base)
        self.request['source_hashes'][SOURCE.resolve().as_posix()]='0'*64
        with self.assertRaisesRegex(ValueError,'match packaged snapshot'):
            mod.run(self.request,self.check)

    def test_budget_stops_before_render_without_output(self):
        def budget(**fields):
            if fields['stage']=='packaged-replay-render-before':raise RuntimeError('synthetic budget stop')
        with self.assertRaisesRegex(RuntimeError,'budget stop'):
            mod.run(self.request,budget)
        self.assertFalse(Path(self.request['output']).exists())

    def stale_bytecode(self):
        original = self.plot.read_bytes()
        stat = self.plot.stat()
        alternate = original.replace(b"!=1024",b"!=9999")
        self.assertNotEqual(alternate,original)
        self.assertEqual(len(alternate),len(original))
        self.plot.write_bytes(alternate)
        os.utime(self.plot,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        py_compile.compile(str(self.plot),doraise=True)
        self.plot.write_bytes(original)
        os.utime(self.plot,ns=(stat.st_atime_ns,stat.st_mtime_ns))

    def test_rejects_valid_stale_bytecode_before_packaged_imports(self):
        self.stale_bytecode()
        self.assertEqual(self.request['plot_module']['sha256'],mod.sha(self.plot))
        with self.assertRaisesRegex(ValueError,'Bytecode cache'):
            mod.run(self.request,self.check)
        self.assertFalse(Path(self.request['output']).exists())

    def test_source_loader_executes_verified_bytes_even_with_stale_cache(self):
        self.stale_bytecode()
        loaded = mod.load(self.plot,mod.sha(self.plot),'_synthetic_exact_source_bytes')
        loaded.validate(self.value)

    def test_rejects_external_bytecode_cache_prefix(self):
        with patch.object(sys,'pycache_prefix',str(self.root/'external-cache')):
            with self.assertRaisesRegex(ValueError,'External bytecode cache prefix'):
                mod.run(self.request,self.check)
        self.assertFalse(Path(self.request['output']).exists())

    def test_rejects_case_variant_bytecode_cache_on_windows(self):
        cache = self.code/'__PYCACHE__'
        cache.mkdir()
        (cache/'synthetic.CPYTHON-312.PYC').write_bytes(b'invented bytecode placeholder')
        with self.assertRaisesRegex(ValueError,'Bytecode cache'):
            mod.run(self.request,self.check)


    def test_accepts_compact_title_artifact_under_new_figures_v2_only_with_pins(self):
        label_path = self.report/'figures-v2/screen/labels.json'
        label_path.parent.mkdir(parents=True)
        label_path.write_bytes(self.labels.read_bytes())
        self.bind('labels',label_path,'input_hashes')
        self.assertEqual(mod.run(self.request,self.check)['titles_present'],1)

    def test_all_integrity_and_cache_helpers_are_ast_identical_to_frozen_harness(self):
        old_source = SOURCE.with_name('replay_portable.py')
        self.assertEqual(mod.sha(old_source),'6f9dcd4c6c97c42f8ab33ae16648d0d510a58989f1edf9320c2dacd063d48cc4')
        old_tree = ast.parse(old_source.read_text(encoding='utf-8'))
        new_tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        old = {node.name:node for node in old_tree.body if isinstance(node,ast.FunctionDef)}
        new = {node.name:node for node in new_tree.body if isinstance(node,ast.FunctionDef)}
        for name in ('require','sha','reject_constant','read','canonical','aggregate_only','no_ml_modules','no_bytecode','pin','load'):
            self.assertEqual(ast.dump(old[name]),ast.dump(new[name]),name)
        class PresentationOnly(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value=='plot_results_v3.py':node.value='plot_results_v2.py'
                return node
            def visit_BoolOp(self,node):
                self.generic_visit(node)
                node.values=[value for value in node.values if ast.unparse(value)!="label_path.is_relative_to(report / 'figures-v2')"]
                return node
        normalized=PresentationOnly().visit(new['run'])
        self.assertEqual(ast.dump(old['run']),ast.dump(normalized))


if __name__=='__main__':
    unittest.main()
