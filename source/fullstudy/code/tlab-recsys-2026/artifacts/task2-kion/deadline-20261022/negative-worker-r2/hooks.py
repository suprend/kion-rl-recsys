"""Compose the reviewed single-pass identity hook with one sampler replacement."""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD_HOOKS = Path('D:/tlab-recsys-2026/artifacts/task2-kion/quality-recovery-20261021/uniform-control-r1/hooks.py')
OLD_SHA = '2dc7ba017dac6314a30ca4e2ff33ebd5d26602c4d99129f5535f0b5637465989'


def _load(path, digest, name):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('Pinned helper changed: ' + path.as_posix())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    return module


old = _load(OLD_HOOKS, OLD_SHA, '_retrieval128_reused_singlepass')
# The root worker verifies this local helper's request pin before hooks are loaded.
sampler_path = HERE / 'sampler.py'
sampler = _load(sampler_path, hashlib.sha256(sampler_path.read_bytes()).hexdigest(), '_retrieval128_sampler')


class SinglePass(old.SinglePass):
    def __init__(self, train, sidecar, snapshot):
        super().__init__(train, sidecar, snapshot)
        self.sampler = sampler.InstalledSampler(train)
        self.dynamic_cache = None
        self.cache_attempted = False
        self.evidence['runtime_performance'] = {'cpu_threads': 2, 'dynamic_cache': False}

    def install(self):
        self.sampler.install()
        try:
            super().install()
        except BaseException:
            self.sampler.restore()
            raise
        self.evidence['sampler_contract'] = sampler.CONTRACT
        self.evidence['sampler_specification'] = sampler.specification()
        self.evidence['sampler_audit'] = self.sampler.replacement.audit
        self.evidence['sampler_binding_verified'] = (
            self.originals['update'].__globals__['sample_training_candidates'] is self.sampler.replacement)

    def restore(self):
        try:
            super().restore()
        finally:
            try:
                self.sampler.restore()
            finally:
                if self.dynamic_cache is not None:
                    self.evidence['runtime_performance']['dynamic_cache_stats'] = self.dynamic_cache.stats()
                    self.dynamic_cache.uninstall()
                    self.dynamic_cache = None

    def update(self, *args, **kwargs):
        if not self.cache_attempted:
            self.cache_attempted = True
            # Only a pinned root control loader may attach this optional installer.
            installer = getattr(self.train, '_negative_dynamic_cache_install', None)
            if installer is not None:
                if not callable(installer) or len(args) != 6:
                    raise ValueError('Admitted DynamicIndex cache installer/call form differs')
                self.dynamic_cache = installer(args[1])
                self.evidence['runtime_performance']['dynamic_cache'] = True
        return super().update(*args, **kwargs)
