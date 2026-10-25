"""Opt-in, instance-local candidate acceleration. No provider/data loads on import."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'candidate-optim-20261022-r1'
PINS = {
    'provider': ('ready-pack-20261020/provider/runtime-native-r5/scripts/kion_complete_pack.py', '7cbe284dfbd9d133bd98e2d363a9ee8b8d8530ffe7b977cbe13c6146d0f4fe5e'),
    'personal_source': ('ready-pack-20261020/provider/runtime-native-r5/scripts/kion_personal_candidates.py', '4769be6fe5de205a9cd793d511aa58a5c2b602a7e18c3690db44b4d7c119cff8'),
    'personal_adapter': ('candidate-optim-20261022-r1/personal/personal_vector.py', 'bb85283c5741a79f21635d785c32f76f3c2958903380cab7dac8b677ba035096'),
    'related_adapter': ('candidate-optim-20261022-r1/related/related_direct_features.py', '15e311c091dc84180c4ef3650282e5a35eef52bc7c47333a54fd826162a444e4'),
}


def checked_path(key):
    relative, expected = PINS[key]
    path = ROOT / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Performance source pin changed: ' + key)
    return path


def _load(key):
    path = checked_path(key)
    name = '_night_performance_' + key
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Personal adapter imports only the already-loaded frozen personal module.
    spec.loader.exec_module(module)
    return module


class Handle:
    def __init__(self, provider, mode):
        self.provider, self.mode = provider, mode
        self.personal_method = self.related_handle = None
        self.installed = False

    def _check_owner(self):
        if self.personal_method is not None and self.provider.personal.__dict__.get('candidate_features') != self.personal_method:
            raise RuntimeError('Personal adapter ownership changed')
        if self.related_handle is not None and self.related_handle.installed:
            related = self.related_handle
            if related.index.__dict__.get('features') != related.method or related.index.__dict__.get('_direct_day') is not related.day:
                raise RuntimeError('Related adapter ownership changed')

    def uninstall(self):
        if not self.installed:
            return
        self._check_owner()
        if self.related_handle is not None:
            self.related_handle.uninstall()
        if self.personal_method is not None:
            del self.provider.personal.__dict__['candidate_features']
        self.installed = False

    def evidence(self):
        return {'contract': 'kion-night-performance-hook-v1', 'mode': self.mode,
                'installed': self.installed, 'candidate_columns': len(self.provider.feature_columns['candidate']),
                'pins': {key: {'path': (ROOT / value[0]).as_posix(), 'sha256': value[1]}
                         for key, value in PINS.items()},
                'scope': 'candidate arithmetic only; query store and caches keep their existing owner'}


def install(provider, *, mode='personal-related'):
    """Install on a disposable/admitted provider after its query-store hook.

    Modes: personal-related (default), personal, related. Baseline is obtained by
    omitting this hook. No factory, provider class, database, cache or source file
    is mutated. Caller owns resource admission and real-output parity evidence.
    """
    if mode not in ('personal-related', 'personal', 'related'):
        raise ValueError('Unknown performance mode')
    candidate_method = provider.candidate_features
    code = getattr(getattr(candidate_method, '__func__', None), '__code__', None)
    if code is None or Path(code.co_filename).resolve() != checked_path('provider').resolve():
        raise ValueError('Requires exact frozen CompleteProvider.candidate_features')
    columns = provider.feature_columns['candidate']
    if len(columns) != 1068 or len(set(columns)) != 1068:
        raise ValueError('Requires complete 1068-coordinate schema')
    handle = Handle(provider, mode)
    # Validate both mutable method boundaries before changing either.
    if 'personal' in mode:
        index = provider.personal
        function = getattr(getattr(index, 'candidate_features', None), '__func__', None)
        personal_code = getattr(function, '__code__', None)
        if (type(index).__name__ != 'PersonalCandidateIndex' or personal_code is None
                or Path(personal_code.co_filename).resolve() != checked_path('personal_source').resolve()
                or 'candidate_features' in index.__dict__ or len(index.columns) != 142):
            raise ValueError('Requires unwrapped pinned PersonalCandidateIndex')
        module = sys.modules.get(function.__module__)
        if module is None or Path(module.__file__).resolve() != checked_path('personal_source').resolve():
            raise ValueError('Personal module identity differs')
        personal = _load('personal_adapter')
    try:
        if 'related' in mode:
            related = _load('related_adapter')
            handle.related_handle = related.install_related_direct_features(provider)
        if 'personal' in mode:
            handle.personal_method = MethodType(personal.candidate_features, provider.personal)
            provider.personal.candidate_features = handle.personal_method
        handle.installed = True
        return handle
    except BaseException:
        if handle.related_handle is not None:
            handle.related_handle.uninstall()
        raise
