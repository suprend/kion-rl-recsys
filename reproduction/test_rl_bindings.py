"""Tiny synthetic files only; no actual cache, model or campaign access."""
import copy
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import rl_bindings as b

try:
    import numpy as np
except ImportError:
    np = None


class Fixture:
    def __init__(self, root):
        self.root, self.entries = root, {}

    def add(self, name, value):
        file = self.root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.npy'):
            np.save(file, value, allow_pickle=False)
        else:
            file.write_text(json.dumps(value), encoding='utf-8')
        raw = file.read_bytes()
        logical = 'Z:/original/' + name
        self.entries[logical] = dict(logical=logical, physical=file.as_posix(),
            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), role='synthetic')
        result = dict(path=logical, sha256=self.entries[logical]['sha256'])
        if name.endswith('.npy'):
            result.update(dtype=str(value.dtype), shape=list(value.shape), bytes=len(raw))
        return result

    def numerical(self, bad_formula=False, bad_flags=False):
        state = self.add('cache/user.npy', np.array([7], dtype='int64'))
        self.cache = dict(contract=b.CONTRACT, state='completed', feature_schema_sha256=b.SCHEMA,
            source_columns=b.SOURCE, candidate_width=1024, no_target_injection=True, no_seen_filter=True,
            arrays={k:dict(state, path='user.npy') for k in ('user_ids','origins','x','lengths','candidate_raw_ids')})
        cache_pin = self.add('cache/cache.json', self.cache)
        reward = np.array([[.75, .25]], dtype='float32')
        support = np.array([[True, False]])
        files = {name:self.add('reward/'+name, value) for name,value in
            [('rewards.npy',reward),('support.npy',support),('fallback.npy', support if bad_flags else ~support)]}
        self.reward = dict(ready=True, contract='current-C-local-RM-r2-bridge-v1', cache=cache_pin,
            state_arrays={k:state for k in self.cache['arrays']}, files=files)
        reward_pin = self.add('reward/reward.json',self.reward)
        prior = np.array([[.5,.5]],dtype='float64')
        utility = np.where(support,reward.astype('float64')-prior,0.)
        if bad_formula:
            utility[0,1] = .01
        ufiles = {name:dict(self.add('utility/'+name, value),path=name) for name,value in
            [('utility.npy',utility),('prior.npy',prior)]}
        self.utility = dict(contract=b.UTILITY_CONTRACT,state='completed',formula=b.FORMULA,range=[-1,1],
            coefficient=1.,prior_archive=dict(sha256=b.PRIOR_SHA256),cache=cache_pin,parent_reward=reward_pin,
            support=files['support.npy'],files=ufiles,cohort='train',preregistration_sha256='a'*64)
        train = self.add('utility/utility.json',self.utility)
        screen = self.add('screen/utility.json',dict(self.utility,cohort='screen'))
        prefit = self.add('prefit.json',dict(contract=b.UTILITY_CONTRACT,state='completed',both_cohorts_ready=True,
            prefit_gate_passed=True,preregistration_sha256='a'*64,cohorts=dict(train=dict(nondegenerate_prior=True)),
            utility_manifests=dict(train=train,screen=screen)))
        self.request = dict(prefit_path=prefit['path'],prefit_sha256=prefit['sha256'],
            preregistration_sha256='a'*64,utility_dir='Z:/original/utility',reward_dir='Z:/original/reward')
        self.arrays = dict(candidate_ids=np.array([[1,2]]))
        return b.BindingResolver(self.entries)


class ResolverTests(unittest.TestCase):
    def test_metadata_size_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Fixture(Path(tmp))
            pin=fixture.add('large.json',{'padding':'x'*(8*1024*1024)})
            resolver=b.BindingResolver(fixture.entries)
            with self.assertRaisesRegex(ValueError,'8 MiB'):
                resolver.read_bound(pin['path'])

    def test_stdlib_imports_only(self):
        tree=ast.parse(Path(b.__file__).read_text(encoding='utf-8'))
        imports={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
        imports.update(alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names)
        self.assertLessEqual(imports, {'copy','hashlib','json','pathlib'})

    def test_two_roots_unchanged_metadata(self):
        value = {'parent':{'path':'Z:/original/untouched.json'},'extra':[1,2]}
        digests=[]
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                fixture=Fixture(Path(tmp)); pin=fixture.add('meta.json',value)
                resolver=b.BindingResolver(fixture.entries)
                self.assertEqual(resolver.read_bound(pin['path']),value)
                digests.append(resolver.sha_bound(pin['path']))
                with self.assertRaises(ValueError): resolver.resolve('Z:/unknown/file.json')
                with self.assertRaises(ValueError): resolver.resolve(Path(tmp)/'unregistered')
                fixture.entries[pin['path']]['sha256']='0'*64
                self.assertEqual(resolver.read_bound(pin['path']),value)
                (Path(tmp)/'meta.json').write_bytes(b'{}')
                with self.assertRaises(ValueError): resolver.read_bound(pin['path'])
        self.assertEqual(*digests)

    def test_duplicates_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Fixture(Path(tmp)); fixture.add('meta.json',{})
            entry=next(iter(fixture.entries.values()))
            for second in (entry,dict(entry,logical='Z:/other.json'),dict(entry,logical='Z:/original/../meta.json')):
                with self.assertRaises(ValueError): b.BindingResolver([entry,second])
            for name in ('../escape.npy','Z:/escape.npy','/escape.npy','sub\\escape.npy'):
                with self.assertRaises(ValueError): b.payload(Path(tmp),name)


@unittest.skipIf(np is None, 'Optional tiny-array NumPy tests')
class NumericalTests(unittest.TestCase):
    def test_header_range_cache_and_parent_corruption(self):
        mutations = (
            ('reward', lambda m: m['files']['rewards.npy'].update(dtype='float64')),
            ('reward', lambda m: m['files']['support.npy'].update(shape=[9,9])),
            ('reward', lambda m: m['cache'].update(sha256='0'*64)),
            ('reward', lambda m: m['state_arrays']['user_ids'].update(sha256='0'*64)),
            ('utility', lambda m: m['parent_reward'].update(sha256='0'*64)),
            ('utility', lambda m: m.update(coefficient=.5)),
        )
        for target, mutate in mutations:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                f=Fixture(Path(tmp)); f.numerical()
                metadata=copy.deepcopy(getattr(f,target)); mutate(metadata)
                f.add(target+'/'+target+'.json',metadata)
                with self.assertRaises(ValueError):
                    b.utility_arrays_bound('Z:/original/utility',f.arrays,f.cache,np,b.BindingResolver(f.entries))
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); resolver=f.numerical()
            # Byte corruption must fail before np.load can decode the payload.
            (Path(tmp)/'reward/rewards.npy').write_bytes(b'invalid')
            with self.assertRaises(ValueError):
                b.reward_arrays_bound('Z:/original/reward',f.arrays,f.cache,np,resolver)

    def test_full_valid_chain_and_prefit(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); resolver=f.numerical(); original=copy.deepcopy(f.utility)
            manifest, arrays=b.open_cache_bound('Z:/original/cache',np,resolver)
            self.assertEqual(manifest,f.cache)
            for array in arrays.values():
                array._mmap.close()
            utility,prior,support=b.utility_arrays_bound('Z:/original/utility',f.arrays,f.cache,np,resolver)
            self.assertTrue(np.array_equal(utility,np.array([[.25,0.]])))
            self.assertEqual(f.utility,original)
            b.validate_parent_reward_bound(f.utility,f.request,resolver)
            b.validate_prefit_bound(f.request,resolver)
            (Path(tmp)/'screen/utility.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError): b.validate_prefit_bound(f.request,resolver)

    def test_exact_formula_and_fallback(self):
        for kwargs in (dict(bad_formula=True),dict(bad_flags=True)):
            with tempfile.TemporaryDirectory() as tmp:
                f=Fixture(Path(tmp)); resolver=f.numerical(**kwargs)
                with self.assertRaises(ValueError):
                    b.utility_arrays_bound('Z:/original/utility',f.arrays,f.cache,np,resolver)

    def test_parent_identity_and_support_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); f.numerical()
            f.add('other/reward.json',f.reward)
            resolver=b.BindingResolver(f.entries)
            with self.assertRaises(ValueError):
                b.validate_parent_reward_bound(f.utility,dict(f.request,reward_dir='Z:/original/other'),resolver)
            changed=copy.deepcopy(f.utility); changed['support']['sha256']='0'*64
            f.add('utility/utility.json',changed)
            with self.assertRaises(ValueError):
                b.utility_arrays_bound('Z:/original/utility',f.arrays,f.cache,np,b.BindingResolver(f.entries))


if __name__ == '__main__':
    unittest.main()
