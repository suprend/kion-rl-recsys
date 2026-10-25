"""Read-only finite RL relocation port; contract kion-rl-bound-input-validators-v1.

Validator provenance: policy_v2 da7baf199b685cb467edd7d5eedaada151798ec40bbc23e9edfaee8e36340dba,
utility 4cffa8d876c84723326f035b825cafc520c4ad561d98e3851bfe38a5d3e1cd0f,
residual 781db1659305f2e6cc020f96cf52266201fa9f2ecfe21adde18bf6f16127b7f0.
This port has its own source identity. No global patches or metadata rewriting.
Imports are stdlib-only; callers explicitly supply a numerical backend.
"""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

ADAPTER_CONTRACT = 'kion-rl-bound-input-validators-v1'
CONTRACT = 'kion-current-C1070-natural1024-H1-v1'
SCHEMA = '71d90eee774d30db804d1ab32c4d82995d1320baf26aaf36443a6498fd85fc5d'
SOURCE = ['current_detached_gpt_logit', 'log1p_strict7d_count']
UTILITY_CONTRACT = 'kion-derived-completion-utility-v1'
FORMULA = 'r_RM(s,i)-m_fit(i)'
PRIOR_SHA256 = 'd5df55586a9df54b5e121d494a149764d956750ac3caaf2b4c1ada27a93243d4'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_physical(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts, 'Absolute physical path required')
    for part in (path, *path.parents):
        require(not part.is_symlink() and not getattr(part, 'is_junction', lambda: False)(),
                'Symlink/junction forbidden')
    return path.resolve()


class BindingResolver:
    """Snapshot train_rl.Bindings.entries or its exported list; construction only stats.

    Only exact registered logical names and physical files resolve. Reads verify
    original bytes before parsing. Directory mappings are never inferred.
    """
    def __init__(self, entries):
        entries = entries.values() if isinstance(entries, dict) else entries
        self._entries, self._physical = {}, {}
        for original in entries:
            entry = copy.deepcopy(original)
            logical = entry['logical']
            require(isinstance(logical, str) and '\\' not in logical and
                    (PureWindowsPath(logical).is_absolute() or PurePosixPath(logical).is_absolute()) and
                    '..' not in PurePosixPath(logical).parts and str(PurePosixPath(logical)) == logical,
                    'Canonical logical path required')
            require(logical.casefold() not in {k.casefold() for k in self._entries}, 'Duplicate logical binding')
            digest = entry['sha256']
            require(isinstance(digest, str) and len(digest) == 64 and
                    all(c in '0123456789abcdef' for c in digest), 'Invalid SHA256')
            file = safe_physical(entry['physical'])
            require(file.is_file() and isinstance(entry['bytes'], int) and
                    file.stat().st_size == entry['bytes'], 'Bound size mismatch')
            require(all(not file.samefile(old) for old in self._physical), 'Physical alias binding')
            entry['physical'] = file
            self._entries[logical], self._physical[file] = entry, logical

    def _entry(self, value):
        if str(value) in self._entries:
            return self._entries[str(value)]
        file = safe_physical(value)
        require(file in self._physical, 'Unknown binding: ' + str(value))
        return self._entries[self._physical[file]]

    def resolve(self, value):
        return safe_physical(self._entry(value)['physical'])

    def sha_bound(self, value):
        entry = self._entry(value)
        file = self.resolve(value)
        require(file.stat().st_size == entry['bytes'], 'Bound size changed')
        digest = hashlib.sha256()
        with file.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        result = digest.hexdigest()
        require(result == entry['sha256'], 'Bound SHA256 changed')
        return result

    def read_bound(self, value):
        entry = self._entry(value)
        file = self.resolve(value)
        limit = 8 * 1024 * 1024
        require(file.stat().st_size <= limit and entry['bytes'] <= limit, 'Metadata exceeds 8 MiB')
        with file.open('rb') as handle:
            data = handle.read(limit + 1)
        require(len(data) <= limit, 'Metadata exceeds 8 MiB')
        require(len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256'],
                'Bound metadata changed')
        return json.loads(data.decode('utf-8'))

    def root(self, value, filename):
        return self.resolve(str(PurePosixPath(str(value).replace('\\', '/')) / filename)).parent


def payload(root, relative):
    require(isinstance(relative, str) and '\\' not in relative and
            not PurePosixPath(relative).is_absolute() and not PureWindowsPath(relative).drive and
            '..' not in PurePosixPath(relative).parts, 'Relative contained payload required')
    file = safe_physical(root / relative)
    require(file.is_relative_to(root), 'Payload escape')
    return file


def open_cache_bound(path, np, bindings):
    root = bindings.root(path, 'cache.json')
    manifest = bindings.read_bound(root / 'cache.json')
    require(manifest.get('contract') == CONTRACT and manifest.get('state') == 'completed', 'Invalid cache contract')
    require(manifest.get('feature_schema_sha256') == SCHEMA and manifest.get('source_columns') == SOURCE
            and manifest.get('candidate_width') == 1024 and manifest.get('no_target_injection') is True
            and manifest.get('no_seen_filter') is True, 'Cache policy schema differs')
    arrays = {}
    for key, entry in manifest['arrays'].items():
        file = payload(root, entry['path'])
        require(file.is_relative_to(root.resolve()) and file.stat().st_size == entry['bytes'] and bindings.sha_bound(file) == entry['sha256'], 'Changed cache array: ' + key)
        value = np.load(file, mmap_mode='r', allow_pickle=False)
        require(str(value.dtype) == entry['dtype'] and list(value.shape) == entry['shape'], 'Changed NPY header')
        arrays[key] = value
    return manifest, arrays

def reward_arrays_bound(path, cache, manifest, np, bindings):
    """Bridge receipt pins exact observed state arrays and the completed cache manifest."""
    root = bindings.root(path, 'reward.json')
    metadata = bindings.read_bound(root / 'reward.json')
    require(metadata.get('ready') is True and metadata.get('contract') == 'current-C-local-RM-r2-bridge-v1', 'Incomplete reward bridge')
    cache_path = bindings.resolve(metadata['cache']['path'])
    require(bindings.sha_bound(cache_path) == metadata['cache']['sha256'] and bindings.read_bound(cache_path) == manifest, 'RM cache binding differs')
    for key in ('user_ids', 'origins', 'x', 'lengths', 'candidate_raw_ids'):
        entry = metadata['state_arrays'][key]
        require(bindings.sha_bound(entry['path']) == entry['sha256'] == manifest['arrays'][key]['sha256'], 'RM state binding differs: ' + key)
    values = []
    for filename in ('rewards.npy', 'support.npy', 'fallback.npy'):
        entry, file = metadata['files'][filename], payload(root, filename)
        require(bindings.sha_bound(file) == entry['sha256'], 'RM output changed')
        value = np.load(file, allow_pickle=False)
        require(str(value.dtype) == entry['dtype'] and list(value.shape) == entry['shape'], 'RM header differs')
        values.append(value)
    reward, support, fallback = values
    require(support.dtype == np.bool_ and fallback.dtype == np.bool_, 'RM support flags must be boolean')
    require(reward.shape == support.shape == fallback.shape == cache['candidate_ids'].shape, 'RM shape mismatch')
    require(bool(np.isfinite(reward[support]).all()) and bool(((reward[support] >= 0) & (reward[support] <= 1)).all()), 'Invalid supported calibrated rewards')
    require(not bool((support & fallback).any()), 'Fallback cannot be optimized')
    return reward, support, fallback

def utility_arrays_bound(path, cache, manifest, np, bindings):
    """Return (derived utility, observed-fit prior, unchanged support); never RM scores."""
    root = bindings.root(path, 'utility.json')
    meta = bindings.read_bound(root / 'utility.json')
    require(meta.get('contract') == UTILITY_CONTRACT and meta.get('state') == 'completed'
            and meta.get('formula') == FORMULA and meta.get('range') == [-1, 1]
            and meta.get('coefficient') == 1.0 and meta.get('prior_archive', {}).get('sha256') == PRIOR_SHA256,
            'Invalid separate derived utility contract')
    require(bindings.sha_bound(meta['cache']['path']) == meta['cache']['sha256'] and bindings.read_bound(meta['cache']['path']) == manifest,
            'Utility cache binding differs')
    parent = meta['parent_reward']
    require(bindings.sha_bound(parent['path']) == parent['sha256'], 'Parent reward changed')
    reward, support, _ = reward_arrays_bound(bindings.resolve(parent['path']).parent, cache, manifest, np, bindings)
    require(bindings.sha_bound(meta['support']['path']) == meta['support']['sha256'] == bindings.read_bound(parent['path'])['files']['support.npy']['sha256'],
            'Utility support changed')
    loaded = []
    for name in ('utility.npy', 'prior.npy'):
        entry = meta['files'][name]
        file = payload(root, entry['path'])
        require(file.is_relative_to(root) and bindings.sha_bound(file) == entry['sha256'], 'Derived utility payload changed')
        value = np.load(file, allow_pickle=False)
        require(str(value.dtype) == entry['dtype'] == 'float64' and list(value.shape) == entry['shape']
                and value.shape == support.shape and bool(np.isfinite(value).all()), 'Invalid utility header/finite values')
        loaded.append(value)
    utility, prior = loaded
    require(bool(((prior >= 0) & (prior <= 1)).all()) and bool(((utility >= -1) & (utility <= 1)).all()),
            'Derived utility/prior range failed')
    require(bool(np.array_equal(utility, np.where(support, np.asarray(reward, dtype=np.float64) - prior, 0.))),
            'Derived formula or unsupported neutrality differs')
    return utility, prior, support

def validate_prefit_bound(request, bindings):
    require(bindings.sha_bound(request['prefit_path']) == request['prefit_sha256'], 'Prefit diagnostics pin changed')
    receipt = bindings.read_bound(request['prefit_path'])
    require(receipt.get('contract') == UTILITY_CONTRACT and receipt.get('state') == 'completed'
            and receipt.get('both_cohorts_ready') is True and receipt.get('prefit_gate_passed') is True
            and receipt.get('preregistration_sha256') == request['preregistration_sha256']
            and receipt['cohorts']['train'].get('nondegenerate_prior') is True, 'Complete nondegenerate two-cohort prefit required')
    require(set(receipt['utility_manifests']) == {'train', 'screen'}, 'Both utility manifests required')
    for cohort, pin in receipt['utility_manifests'].items():
        require(bindings.sha_bound(pin['path']) == pin['sha256'], 'Prefit utility manifest changed')
        meta = bindings.read_bound(pin['path'])
        require(meta.get('contract') == UTILITY_CONTRACT and meta.get('cohort') == cohort
                and meta.get('preregistration_sha256') == request['preregistration_sha256'], 'Prefit cohort binding differs')
    train = receipt['utility_manifests']['train']
    require(bindings.resolve(train['path']) == bindings.root(request['utility_dir'], 'utility.json') / 'utility.json', 'Wrong train utility directory')
    return receipt

def validate_parent_reward_bound(metadata, request, bindings):
    parent = metadata['parent_reward']
    require(bindings.sha_bound(parent['path']) == parent['sha256'], 'Parent reward changed')
    require(bindings.resolve(parent['path']) == bindings.root(request['reward_dir'], 'reward.json') / 'reward.json',
            'Wrong parent reward identity')
