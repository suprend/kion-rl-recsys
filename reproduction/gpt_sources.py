"""Pinned retained-source execution, explicitly activated in a dedicated process.

This is an import boundary, NOT a sandbox. Keep the context open for every call
into retained code. Do not run unrelated import threads in that process. Third
party packages remain in sys.modules after exit; only owned source names are
removed. No source bytes, signed metadata, Path or open functions are rewritten.
"""
import ast
import builtins
import hashlib
from pathlib import Path
import sys
import threading
from types import MappingProxyType, ModuleType

from gpt_provider import SOURCE_LIMIT, ORIGINAL_PROVIDER_SHA256, FileBinding

RUNTIME_ROOT = 'D:/tlab-recsys-2026/artifacts/task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts'
_LOCK = threading.Lock()
# Exact original pins populated below; additional closure files are source-only.
RUNTIME_PINS = MappingProxyType({
    "hm_job": "a8ca33c389aad356ea2bd8b0a2ab43276a0af85029934888ebeca9f132c541a0",
    "kion_audience_fast": "b0d358ce30f4b7e49b3793f4aa4d2d659995a669e1d60189fd8ec96da3459be8",
    "kion_audience_features": "70777ef4301e3dcebcf69378ce879888f252f5829f53aee4b45f131fa36e98f1",
    "kion_basic_bundle_features": "25c5a80a2fba9de5ee76c67ac4bd6cd7bd437cf807cd42855baeba3b50f73ef8",
    "kion_complete_manifest": "1cc9ca82a712c0a4cbb55e2f8a651b40ac698793f50e58d35b51d98f2bfb9549",
    "kion_complete_model": "8d1be2352cd268481f4a1f7b764943f745e910695a1a270fd3d0b1c4e7ec24e7",
    "kion_complete_pack": "7cbe284dfbd9d133bd98e2d363a9ee8b8d8530ffe7b977cbe13c6146d0f4fe5e",
    "kion_complete_train": "16180fc30db79f6a5b5c0fc82080eb89658d4b466c65f3a96f66d313c85746d9",
    "kion_completion_diagnostics": "862e647f96f10ed0e2a1e8b5c78c80245fb48c9029937706e68e1a2d1962d552",
    "kion_completion_event_store": "8dd79dff657f4feaa2ef5d430278ec24fc2dc7c11aa713f3bc65b4f755a20ef5",
    "kion_conditioned_train": "b6c2c93bb8ac1475d00a1a20cd8d278da898468ae38a001cc5be4ec218478caf",
    "kion_cuda_batches": "efcdc050c2e3778731e6f1e0ad49bb21c0b5dc622ea22f1f7992ed8c21568483",
    "kion_entity_links": "c505adaa15c7af136dfbc9b4fe6372d24e83f034950684125c7e0231f8e50b11",
    "kion_external_facts": "240c4babdd9763e6c1a62e8fc78c987a456988fbe08677563fbaa15fb3ac222c",
    "kion_external_interest": "a1e0f9a0dbd005c9df28287c40aaf3e465813ce491f63391285336d650af809c",
    "kion_franchise_progress": "a954c884b98082093a351f23d4e508e91f4b73bb62b2c35999d45a11c9250ba0",
    "kion_full_train": "2b8d684e9f789eb7a51823e0b38df183dcd2f9f72942894a905fde1f1efd47ea",
    "kion_geography_time": "33c34f28f2294362505f366c2a484fbda8053d2af0e93009683b53815df14c3e",
    "kion_local_features": "e5f992093b53839d425ccd436dba0404862ad912b795a390d60ff6c9a2576de1",
    "kion_local_pack": "87412446a72033aae98897250cc0178084dd76b5818aee38a24fbb81a8e02aaf",
    "kion_narrative_relations": "e0d3978901a84db781543046252592a4e1db978a5e3e4853dc1acc9942360756",
    "kion_pack_dynamic": "7f53e94ddd5940514de8a132711ff5d782885983af9f0c62ae4c4515261bdd7a",
    "kion_pack_history": "ecdd0cd0e853d20578a7855acc8987c72c430925d1a30f7e26fee250000659dd",
    "kion_pack_people": "25c9b1f9ac9188f096f22085d63a5de694e55a757c9857da963854bb808e791b",
    "kion_pack_semantics": "74a59bad3bf89c5eb92624452ffac8e4775c18592823411a611d09d50fe3d7ff",
    "kion_pack_static": "5a52ef6ab610b0f3d091bd6566982bd0894f1d0c1ad891986dd27d3f8703e181",
    "kion_pack_train": "0f11d3c1bcc5165dfd9f35a7728b9400b679cd51c8bb444af04360643711546a",
    "kion_people_banks": "1a511e8f025bba1f6d0bdb16095fb833e7a407ffffa8e67561f161f42f73b070",
    "kion_people_dynamics": "6bf97bb98c8fdf0cb858db18066862d3a94cb107334bcb14c74e340f2943911c",
    "kion_people_dynamics_fast": "e6ad0d34174c44f50f4c27e088fab36a399e839a0ee96b77ec0df27e2bda2a91",
    "kion_people_embeddings": "473d4a097f3e39c50198c0a6cc15f6dd1ce0b283a37e73234d8def49c760c1ad",
    "kion_personal_candidates": "4769be6fe5de205a9cd793d511aa58a5c2b602a7e18c3690db44b4d7c119cff8",
    "kion_profession_relations": "bf6890604f9388ba49af92d25a42a8ec1e85c751f122e554dd4e424e4c3c3303",
    "kion_related_features": "0999efdccd9d682e4c1d0ce0393bdab12a3abdbc8b2e7fb14a4288a6dde8fae8",
    "kion_semantic_extensions": "6116813f219d83e00d1189dea9c5955b700db85312f462f34e61b2555c37a136",
    "kion_sex_features": "e524c68af1ad327c3e83959aa4503be17aabdc700beaa24793831d09a152a905",
    "kion_sports_settings": "b96e71bcff8e426652b057de5142147fe403dffe2e06c88f0dec4a6a2f3c9a2d",
    "kion_story_format": "56ad45fcab54823ea54253cdaf7b6c724b6b2d95d349ef2803fd217ae9db3c29",
    "kion_text_embeddings": "401272cd3dfe04eadde75f15f8a3d5baf8637e52973ed88bd07b86ca0c088d4a",
    "kion_train": "2066dcddcccabcd4c46971467287f2955cb06434bfa942247ac55f111e8d8d1b",
    "kion_user_diagnostics": "8b39da0a9515d986ef82649b37066d2778b5e45a7bdf6d6e8183f8041ef4e8da"
})


class SourceRuntime:
    """Low-level loader; production callers use pinned_runtime, not custom pins.

    ``pins`` maps finite module names to (logical identity, original SHA256).
    Construction is inert; entering verifies ALL source bytes before execution.
    ``external`` is explicit activation permission, not dependency version proof.
    """
    def __init__(self, io, pins, *, external=()):
        self.io = io
        self.pins = MappingProxyType(dict(pins))
        if not self.pins or len(self.pins) > 64 or any(
                not n.isidentifier() or not (n.startswith('kion_') or n == 'hm_job')
                for n in self.pins):
            raise ValueError('invalid finite retained module namespace')
        if len({v[0] for v in self.pins.values()}) != len(self.pins):
            raise ValueError('ambiguous source identities')
        self.external = frozenset(external)
        if self.external - {'numpy', 'torch', 'scipy', 'transformers'}:
            raise ValueError('unrecognized external activation')
        self.modules, self.sources = {}, {}
        self.active = False
        self.closed = False

    def __enter__(self):
        if self.closed or self.active or not _LOCK.acquire(blocking=False):
            raise RuntimeError('source runtime requires one fresh exclusive context')
        self.active = True
        try:
            collisions = set(self.pins).intersection(sys.modules)
            if collisions:
                raise RuntimeError('retained module collision: ' + ', '.join(sorted(collisions)))
            for name, (logical, pin) in self.pins.items():
                path = self.io.verify(logical, 'source', pin)
                with path.open('rb') as stream:
                    source = stream.read(SOURCE_LIMIT + 1)
                self.io.check()
                if len(source) > SOURCE_LIMIT or hashlib.sha256(source).hexdigest() != pin:
                    raise ValueError('source changed after verification')
                tree = ast.parse(source, filename=logical)
                for node in ast.walk(tree):
                    names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                             else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                    for imported in names:
                        root = imported.split('.')[0]
                        if (root.startswith('kion_') or root == 'hm_job') and root not in self.pins:
                            raise ValueError('incomplete retained source closure: ' + root)
                self.sources[name] = (source, path)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if not self.active:
            raise RuntimeError('retained runtime is closed')
        if level:
            raise ImportError('relative retained imports require an explicit adapter')
        root = name.split('.')[0]
        if root in self.pins:
            if name != root:
                raise ImportError('retained modules are not packages')
            return self.load(root)
        if root.startswith('kion_') or root in {'hm_job', 'scripts'}:
            raise ImportError('undeclared retained module: ' + name)
        if name == 'sys':
            proxy = ModuleType('sys')
            proxy.__dict__.update(vars(sys))
            proxy.path = list(sys.path)
            return proxy
        if name in {'importlib', 'importlib.metadata'}:
            proxy = ModuleType('importlib')
            proxy.import_module = self._import_module
            if name == 'importlib.metadata' or 'metadata' in (fromlist or ()):
                proxy.metadata = builtins.__import__('importlib.metadata', fromlist=('version',))
            if name == 'importlib' and set(fromlist or ()) - {'metadata', 'import_module'}:
                raise ImportError('unadmitted importlib interface')
            return proxy.metadata if name == 'importlib.metadata' and fromlist else proxy
        if root == 'importlib':
            raise ImportError('dynamic source loading requires explicit admitted adapter')
        if root not in sys.stdlib_module_names and root not in self.external:
            raise ImportError('external dependency not activated: ' + root)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def _import_module(self, name, package=None):
        if package is not None or not isinstance(name, str) or name not in self.pins:
            raise ImportError('dynamic import requires an exact admitted module name')
        return self.load(name)

    def load(self, name):
        if not self.active:
            raise RuntimeError('enter runtime context before loading')
        self.io.check()
        if name not in self.pins:
            raise ImportError('undeclared retained module: ' + name)
        if name in self.modules:
            if sys.modules.get(name) is not self.modules[name]:
                raise RuntimeError('retained module registry changed')
            return self.modules[name]
        if name in sys.modules:
            raise RuntimeError('retained module collision: ' + name)
        source, path = self.sources[name]
        module = ModuleType(name)
        module.__file__ = path.as_posix()
        module.__package__ = ''
        module.__loader__ = self
        module.__logical_source__ = self.pins[name][0]
        module.__source_sha256__ = self.pins[name][1]
        module.__builtins__ = dict(vars(builtins), __import__=self._import)
        self.modules[name] = sys.modules[name] = module
        try:
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
        except BaseException:
            self.modules.pop(name, None)
            if sys.modules.get(name) is module:
                del sys.modules[name]
            raise
        return module

    def logical_source(self, module):
        if self.modules.get(module.__name__) is not module:
            raise ValueError('module does not belong to this runtime')
        return self.pins[module.__name__][0]

    def logical_source_name(self, name):
        """Return verified logical provenance without executing a producer."""
        if not self.active:
            raise RuntimeError('source identity requires verified live context')
        return self.pins[name][0]

    def receipt(self):
        if not self.active:
            raise RuntimeError('receipt requires verified live context')
        return {'loader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'sources': {n: {'logical': v[0], 'sha256': v[1],
                               'physical': self.sources[n][1].as_posix()}
                            for n, v in self.pins.items()},
                'loaded': sorted(self.modules), 'external_activation': sorted(self.external)}

    def __exit__(self, *exc):
        if not self.active:
            return
        for name, module in self.modules.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]
        self.active = False
        self.closed = True
        self.sources.clear()
        _LOCK.release()


def pinned_runtime(io, *, external=()):
    """Future execution only; source closure is exact runtime-native-r5.

    Use external=('numpy', 'torch') for the original baseline's import closure.
    Torch cannot be omitted while importing the original kion_pack_train class
    definitions; no stub, transformed source, or eager import hides that fact.
    """
    if RUNTIME_PINS.get('kion_complete_pack') != ORIGINAL_PROVIDER_SHA256:
        raise ValueError('original provider identity mismatch')
    return SourceRuntime(io, {n: (RUNTIME_ROOT + '/' + n + '.py', pin)
                              for n, pin in RUNTIME_PINS.items()}, external=external)


def runtime_source_bindings(repo_root, *, existing=()):
    """Bind shipped source files explicitly; never fall back to historical paths.

    Returns all 41 bindings. Existing declarations of these identities must
    exactly match this explicit repo mapping, otherwise fail instead of silently
    replacing the caller's choice. Hashing happens only on runtime entry.
    """
    root = Path(repo_root)
    if not root.is_absolute():
        raise ValueError('repository root must be explicit and absolute')
    retained = root / 'source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts'
    rows = tuple(FileBinding(RUNTIME_ROOT + '/' + name + '.py',
                             (retained / (name + '.py')).as_posix(), pin, 'source')
                 for name, pin in RUNTIME_PINS.items())
    expected = {row.logical: row for row in rows}
    seen = set()
    for row in existing:
        if not isinstance(row, FileBinding) or row.logical in seen:
            raise ValueError('invalid or duplicate existing source binding')
        seen.add(row.logical)
        if row.logical in expected and row != expected[row.logical]:
            raise ValueError('explicit retained source binding conflicts with repository mapping')
    return rows
