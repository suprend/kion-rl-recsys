"""Tiny synthetic composition tests; never opens retained arrays or imports ML."""
import ast
from collections import OrderedDict
from contextlib import contextmanager
from datetime import date
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

import gpt_complete as c
from gpt_provider import ORIGINAL_PROVIDER_SHA256, FileBinding

SOURCE = Path(__file__).parent.parent / 'source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts/kion_complete_pack.py'


def pure_original():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_PROVIDER_SHA256
    tree = ast.parse(raw)
    selected = []
    constants = {'CONTRACT', 'FIT_CUTOFF', 'STATIC_MODULES', 'TEXT_FIELDS', 'NUMERIC_UNITS',
                 'NUMERIC_TRANSFORMS', 'SCOPED_READY_COMPONENTS', 'SCOPED_FAMILIES', 'FRANCHISE_ORDER_KINDS'}
    methods = {'contract', 'origin_keys', '_rows', '_origin', '_query', 'batch', '_external_bank',
               '_fill_personal_crosses', '_fill_batched_item_crosses', 'candidate_features',
               '_numeric_audit', '_transform_numeric', '_validate_model_range'}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {'_names', '_positive', 'numeric_plan', 'numeric_log_columns'}:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == 'CompleteProvider':
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
            selected.append(node)
    text = SimpleNamespace(FIELDS=('title', 'description', 'keywords'), SPECIALS={},
        _mapping=lambda m: list(m['item_id_to_token']), canonical_hash=c.canonical_hash,
        array_hash=lambda a: hashlib.sha256(a.tobytes()).hexdigest())
    ns = dict(np=np, date=date, copy=copy, text_module=text)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return SimpleNamespace(**ns, baseline=object(), local=SimpleNamespace(row_keys_hash=lambda d: 'rows'),
                           history_module=SimpleNamespace(_day=int))


class CompositionTests(unittest.TestCase):
    def fixture(self):
        original = pure_original()
        # Smaller admitted fixture census only; retained numeric/query methods are exact.
        original.SCOPED_READY_COMPONENTS = ()
        original.SCOPED_FAMILIES = ('local_base',)
        obj = type('Fixture', (c._Composition, original.CompleteProvider), {})()
        obj._original, obj._closed = original, False
        obj.io = SimpleNamespace(check=lambda: None, provenance=lambda *a: None)
        obj.root = 'D:/logical/package'
        obj.scoped, obj.partial = True, False
        obj.scope = {'validation': {'train_rows': 1, 'validation_rows': 1}, 'scope_id': 'fixture'}
        obj.required_families, obj.acceptance = ('local_base',), {'local_base': 'accepted_scoped'}
        obj._maps, obj._profiles = [], OrderedDict()
        obj._require_sources = lambda modules: None
        obj._pin = lambda root, p: c.logical_child(root, p['path'])
        obj.manifest = {'data': {'path': '../data', 'files': {n: {'path': '../data/'+n, 'sha256': 'x'} for n in ('metadata.json','train.npz','val.npz')}},
            'base_cache': {'path': '../base', 'manifest_sha256': 'base'}, 'row_keys_sha256': {'train':'rows','val':'rows'},
            'signature': 'signature', 'scope_decision': {'path':'scope.json'}}
        # Bound paths do not allow traversal, so use original absolute identities.
        obj.manifest['data']['path'] = 'D:/logical/data'
        obj.manifest['base_cache']['path'] = 'D:/logical/base'
        for name, pin in obj.manifest['data']['files'].items(): pin['path'] = 'D:/logical/data/'+name
        def init(config):
            for n in ('history','personal','audience','audience_core','people_dynamics','dynamic','related','graph'):
                setattr(obj, n, None)
            obj.runtime_provenance = {}
        obj._init_runtime = init
        obj._load_diagnostics = lambda d: setattr(obj, '_diagnostic_stores', {}) or {}
        cutoff = original.FIT_CUTOFF
        def split(day):
            return dict(x=np.zeros((1,64),np.int64), lengths=np.array([1]), y=np.array([3]),
                user_id=np.array([7]), target_date=np.array([day]), user_features=np.array([[2]],np.float32))
        base = {'feature_columns': {'item':['base'+str(i) for i in range(197)],'user':['u']}, 'outputs':{}}
        data = ({'context':64,'vocab_size':4,'item_id_to_token':{'1':3}}, split(cutoff-1),split(cutoff),np.zeros((4,197),np.float32),base)
        return obj, data

    def test_actual_load_and_public_protocol(self):
        obj, data = self.fixture()
        with patch.object(c, 'load_scoped_base_data', return_value=data) as loader:
            obj._load('D:/logical/data', False)
        self.assertEqual(loader.call_args.kwargs['manifest_sha256'], 'base')
        self.assertEqual(obj.item_features.shape, (4,197))
        self.assertEqual(obj.batch([0], 'train')[3].tolist(), [[2]])
        self.assertEqual(obj.candidate_features([0], np.array([[3,3]]), 'train').shape,(1,2,0))
        origins = obj.origin_keys('train')
        with self.assertRaises(ValueError): origins.setflags(write=True)
        contract = obj.contract()
        self.assertEqual(contract['status'], 'admitted_scoped')
        self.assertFalse(contract['full_package_admitted'])
        self.assertEqual(contract['preserved_original_item_prefix'],197)
        self.assertEqual(contract['numeric_transforms']['contract'],'kion-complete-numeric-v4')
        contract['feature_columns']['item'].clear()
        self.assertEqual(len(obj.feature_columns['item']),197)
        obj.close()
        with self.assertRaises(RuntimeError): obj.batch([0], 'train')

    def test_bound_external_bank_and_cancellation(self):
        obj, data = self.fixture()
        with patch.object(c, 'load_scoped_base_data', return_value=data): obj._load('D:/logical/data', False)
        bank = np.ones((4,1),np.float32)
        obj._matrix = lambda declaration: bank
        self.assertIs(obj._external_bank({'family':'external_interest','columns':['x'],'banks':{'2':{}}},2), bank)
        obj.io.check = lambda: (_ for _ in ()).throw(TimeoutError('deadline'))
        with self.assertRaises(TimeoutError): obj.origin_keys('train')
        obj.close()

    def test_context_lifetime_and_failure_cleanup(self):
        import gpt_sources
        log=[]
        @contextmanager
        def context(io, **kw):
            log.append('enter')
            try: yield object()
            finally: log.append('exit')
        obj=SimpleNamespace(close=lambda: log.append('close'))
        bindings=[FileBinding('D:/logical/source.py', SOURCE, 'a'*64, 'source')]
        with patch.object(gpt_sources,'pinned_runtime',context), patch.object(c,'_compose',return_value=obj):
            with c.CompleteProvider.load('D:/data','D:/features',bindings=bindings,check=lambda:None,scope_decision='D:/scope',manifest_sha256='a'*64) as value:
                self.assertIs(value,obj)
                self.assertEqual(log,['enter'])
        self.assertEqual(log,['enter','close','exit'])
        log.clear()
        with patch.object(gpt_sources,'pinned_runtime',context), patch.object(c,'_compose',side_effect=ValueError('failed')):
            with self.assertRaises(ValueError):
                with c.CompleteProvider.load('D:/data','D:/features',bindings=bindings,check=lambda:None,scope_decision='D:/scope',manifest_sha256='a'*64): pass
        self.assertEqual(log,['enter','exit'])

    def test_close_continues_after_failed_resource(self):
        obj,_=self.fixture()
        log=[]
        obj.related=SimpleNamespace(close=lambda: (_ for _ in ()).throw(ValueError('bad close')))
        obj._diagnostic_stores={'store':SimpleNamespace(close=lambda:log.append('closed'))}
        with self.assertRaises(ValueError): obj.close()
        self.assertEqual(log,['closed'])

    def test_static_transform_and_base_replacement_preserve_prefix(self):
        obj,data=self.fixture()
        meta,train,val,items,base=data
        items=np.column_stack([items,np.ones(4,np.float32)])
        items[:3,197]=0
        base['feature_columns']['item'].append('replace_me')
        base['outputs']={'item_features.npy':{'sha256':'base-array'}}
        base['schema_hash']='schema'
        obj._source=lambda name:'D:/source/'+name+'.py'
        obj.manifest['source_pins']=[{'path':'D:/source/kion_semantic_extensions.py','sha256':'source'}]
        obj.manifest['item_extensions']=[{'family':'semantic_extensions','columns':['signed'],
            'temporal_mode':'static_retrospective','transform':'signed_log1p','transform_columns':[0]}]
        replacement={'family':'semantic_extensions','mode':'replace_named_columns','columns':['replace_me'],
            'base_manifest_sha256':'base','base_item_features_sha256':'base-array','base_schema_hash':'schema',
            'transform':'identity','sha256':'replacement'}
        obj.manifest['item_replacements']=[replacement]
        obj.runtime=SimpleNamespace(load=lambda name:object())
        obj._matrix=lambda d: np.array([[0],[0],[0],[-3 if d.get('transform')=='signed_log1p' else 8]],np.float32)
        with patch.object(c,'load_scoped_base_data',return_value=(meta,train,val,items,base)):
            obj._load('D:/logical/data',False)
        np.testing.assert_array_equal(obj.item_features[:,:197],items[:,:197])
        self.assertEqual(obj.item_features[3,197],8)
        self.assertAlmostEqual(float(obj.item_features[3,198]),-np.log(4),places=6)
        self.assertEqual(obj.replacements[0]['replaced_values_sha256'],obj._original.text_module.array_hash(items[:,[197]]))
        obj.close()

    def test_compose_load_failure_closes_owned_arrays(self):
        original=pure_original()
        log=[]
        runtime=SimpleNamespace(load=lambda name:original,logical_source=lambda module:'D:/source.py')
        admitted=SimpleNamespace(manifest={},scope={},acceptance={},required_families=())
        def fail(obj,*args):
            obj._maps=[SimpleNamespace(close=lambda:log.append('closed'))]
            raise ValueError('late failure')
        with patch.object(c.AdmissionBackend,'from_sources',return_value=object()), patch.object(c,'admit_scoped',return_value=admitted), patch.object(c._Composition,'_load',fail):
            with self.assertRaisesRegex(ValueError,'late failure'):
                c._compose(object(),runtime,data_dir='D:/data',feature_dir='D:/features',scope_decision='D:/scope',manifest_sha256='a'*64)
        self.assertEqual(log,['closed'])

    def test_date_bank_units_coverage_candidate_scatter_and_transform(self):
        obj,data=self.fixture()
        cutoff=obj._original.FIT_CUTOFF
        units={'movielens_count':'raw_count'}
        banks={str(day):{'cutoff_exclusive':day,'columns':['movielens_count'],
            'numeric_units':units,'provenance':{}} for day in (cutoff-1,cutoff)}
        declaration={'family':'external_interest','clock':'strict_prefix','fit_cutoff':'2021-08-09',
            'columns':['movielens_count'],'numeric_units':units,'banks':banks}
        obj.manifest['date_features']=[declaration]
        seen=[]
        def matrix(d):
            seen.append(d['cutoff_exclusive'])
            return np.array([[0],[0],[0],[3]],np.float32)
        obj._matrix=matrix
        with patch.object(c,'load_scoped_base_data',return_value=data): obj._load('D:/logical/data',False)
        self.assertEqual(seen,[cutoff-1,cutoff])
        output=obj.candidate_features([0],np.array([[3,3,3]]),'train')
        np.testing.assert_allclose(output,np.full((1,3,1),np.log(4)),rtol=1e-6)
        self.assertEqual(obj.feature_columns['candidate'],['external_interest.movielens_count'])
        obj.close()
        obj,data=self.fixture()
        obj.manifest['date_features']=[dict(declaration,banks={str(cutoff):banks[str(cutoff)]})]
        obj._matrix=matrix
        with patch.object(c,'load_scoped_base_data',return_value=data), self.assertRaisesRegex(ValueError,'cover all'):
            obj._load('D:/logical/data',False)
        obj.close()

    def test_extension_replacement_exact_binding_and_preimage(self):
        obj,data=self.fixture()
        obj.base_manifest=data[-1]
        obj.runtime=SimpleNamespace(load=lambda name:object())
        ext={'path':'D:/extension.npy','family':'semantic_extensions','columns':['a'],
            'sha256':'array','shape':[4,1],'dtype':'float32','metadata_sha256':'meta',
            'vocabulary_sha256':'vocab','transform':'identity'}
        target={'path':ext['path'],'family':ext['family'],'namespace':ext['family'],
            'columns_sha256':c.canonical_hash(ext['columns']),
            **{k:ext[k] for k in ('sha256','shape','dtype','metadata_sha256','vocabulary_sha256','transform')}}
        values=np.array([[0],[0],[0],[9]],np.float32)
        receipt={'columns':['semantic_extensions.a'],'target_extension':target,
                 'replaced_values_sha256':obj._original.text_module.array_hash(values)}
        obj._matrix=lambda d:values
        obj._replacement(receipt,[ext])
        with self.assertRaisesRegex(ValueError,'preimage values'):
            obj._replacement(dict(receipt,replaced_values_sha256='0'*64),[ext])
        with self.assertRaisesRegex(ValueError,'identity/schema/preimage'):
            obj._replacement(dict(receipt,target_extension=dict(target,path='D:/other.npy')),[ext])


if __name__ == '__main__': unittest.main()
