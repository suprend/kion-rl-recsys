"""Prepare an immutable historical-cache RL replay inventory; does not execute RL.

Only stdlib metadata reads and payload stats occur. Numerical validation, resource
admission, checkpoint loading and the relocated training port remain separate work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
LOGICAL = 'D:/tlab-recsys-2026/artifacts/task2-kion/'
SOURCE_BASE = 'source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/'
CONTRACT = 'kion-historical-cache-rl-preparation-v1'
SOURCE_PINS = {
    'source/matched-kl/policy/policy.py': 'cf6d6c29fa8c1d763109d53d1be68ade479c9a2a1c01767fc55030ba1d67e7e5',
    SOURCE_BASE + 'c-rl-20261024-r1/policy/policy_v2.py': 'da7baf199b685cb467edd7d5eedaada151798ec40bbc23e9edfaee8e36340dba',
    SOURCE_BASE + 'c-rl-final-20261025-r1/reward/utility.py': '4cffa8d876c84723326f035b825cafc520c4ad561d98e3851bfe38a5d3e1cd0f',
    SOURCE_BASE + 'c-rl-final-20261025-r1/policy/residual.py': '781db1659305f2e6cc020f96cf52266201fa9f2ecfe21adde18bf6f16127b7f0',
    SOURCE_BASE + 'three-head-262k-20261024-r1/heads.py': '795cdfe87f08079cadf067dc9c58811f7bcdacf9d1ebf2dbcb6c019dd68d6799',
}
METADATA = {
    'cache': ('c-rl-20261024-r1/policy/train-cache-r1/cache.json', 'e76c365d1245355d5314e383ba53499800eb3c099fa0af7de106cced5fad15e5'),
    'reward': ('c-rl-20261024-r1/reward/train512-r1/reward.json', 'e8047d12f8009bb422cc4956261a355dc10e05f42d73351ef76ccd5f7bc9ce05'),
    'utility': ('c-rl-final-20261025-r1/reward/residual-r1/train/utility.json', 'e888cb47586b8ea2a4e5323a1051401e949f602ec36e43d36254684e80a320a5'),
    'prefit': ('c-rl-final-20261025-r1/reward/residual-r1/prefit.json', 'f1431dd1092a5d7ef96e2ed89d4c2889067cd3558fa0f478f29eff73825c3d97'),
    'screen': ('c-rl-final-20261025-r1/reward/residual-r1/screen/utility.json', 'b3d4d9a000d953b207566d695066f6210155a345d683a58a1edf48e27765d709'),
    'preregistration': ('c-rl-final-20261025-r1/protocols/preregistration.json', '78c1568c35349394032333696ffce758b9ec1b7aea47edb9eb7023368e7ba59c'),
    'proof': ('c-rl-20261024-r1/policy/repair-proof-r1.json', 'dbdc6622bd620c15cbab77e2b12db4a241bb6e9577270fc0578b5da768d1e32b'),
    'heads': ('three-head-262k-20261024-r1/heads/training.json', '693ee6ae7df73af60f8b4fdaf0996a926c507219a89246a8d5a954d0d93941ad'),
}
BANK_SHA = '7eb271a5c5c44f33c2b22dfe26ce120bbd7016aeeba67b6fef817ad3e183ff5a'
BETA_SHA = 'be54df7bff4aabfa729480fdd3140661b976f532bcd418ad12d05cf1f192e13f'
SCIENCE = dict(seeds=[20261024, 20261025, 20261026],
    utilities={'completion': .5, 'residual': .16274030138555867},
    rl_updates=1000, checkpoint_steps=[0, 250, 500, 1000], endpoint='latest1000',
    start_from='original-C', optimizer='Adam', learning_rate=1e-5, weight_decay=0,
    grad_clip=1, batch_size=16, sampler='python-random-choice-with-replacement',
    precision='cuda-fp16', grad_scaler_initial=1024, temperature=1,
    supervised_updates=0, new_rm_fits=0, final_test_reads=0, cpu_threads=4,
    reserve_gib=2, action_normalization=1024, unsupported_advantage=0,
    loss_normalization='one-logical-B16-mean', utility_centering='supported-C-weighted')


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def safe_path(value):
    path = Path(value).absolute()
    require('..' not in path.parts, 'Traversal is forbidden')
    for part in (path, *path.parents):
        require(not part.is_symlink() and not (hasattr(part, 'is_junction') and part.is_junction()),
                'Symlink/junction is forbidden: ' + str(part))
    return path.resolve()


def child(root, relative):
    require(isinstance(relative, str) and bool(relative), 'Relative filename required')
    require('\\' not in relative and ':' not in relative and
            not PurePosixPath(relative).is_absolute() and not PureWindowsPath(relative).drive and
            '..' not in PurePosixPath(relative).parts, 'Unsafe relative filename')
    root = safe_path(root)
    path = safe_path(root / relative)
    require(path != root and path.is_relative_to(root), 'Relative input escaped root')
    return path


def pinned_bytes(path, expected):
    path = safe_path(path)
    require(path.is_file() and path.stat().st_size <= 8 << 20, 'Missing/oversized bounded input')
    with path.open('rb') as stream:
        data = stream.read((8 << 20) + 1)
    require(len(data) <= 8 << 20 and hashlib.sha256(data).hexdigest() == expected,
            'Pinned bytes changed: ' + str(path))
    return data


def pinned_json(path, expected):
    return json.loads(pinned_bytes(path, expected).decode('utf-8'))


class Bindings:
    """Exact finite lookup; provenance descriptors are never rewritten.

    No prefix substitution or fallback to a historical host filesystem is allowed.
    Aliased physical files and duplicate logical keys are rejected.
    """
    def __init__(self):
        self.entries = {}

    def add(self, logical, physical, sha256, role, expected_bytes=None):
        require(isinstance(logical, str) and PureWindowsPath(logical).is_absolute() and
                '\\' not in logical and '..' not in PurePosixPath(logical).parts,
                'Canonical historical absolute logical path required')
        require(logical not in self.entries, 'Duplicate logical binding')
        require(isinstance(sha256, str) and len(sha256) == 64 and
                all(c in '0123456789abcdef' for c in sha256), 'Invalid digest')
        physical = safe_path(physical)
        require(physical.is_file() and physical.stat().st_size > 0, 'Missing/empty input: ' + str(physical))
        for entry in self.entries.values():
            require(not physical.samefile(entry['physical']), 'Duplicate physical binding')
        size = physical.stat().st_size
        require(expected_bytes is None or size == expected_bytes, 'Payload size mismatch')
        self.entries[logical] = dict(logical=logical, physical=physical.as_posix(),
            sha256=sha256, bytes=size, role=role, content_verified=False)
        return physical

    def resolve(self, logical):
        require(logical in self.entries, 'Unknown logical path: ' + str(logical))
        return safe_path(self.entries[logical]['physical'])

    def descriptor(self, descriptor, expected_logical):
        require(descriptor.get('path') == expected_logical, 'Unexpected descriptor identity')
        require(descriptor.get('sha256') == self.entries[expected_logical]['sha256'],
                'Descriptor digest differs')
        return self.resolve(expected_logical)

    def read_json(self, logical):
        entry = self.entries[logical]
        result = pinned_json(self.resolve(logical), entry['sha256'])
        entry['content_verified'] = True
        return result


def validate_science(science):
    require(canonical_hash(science) == canonical_hash(SCIENCE), 'Fixed science configuration differs')


def validate_metadata(meta, bindings):
    """Metadata identity seams only; numerical array predicates are NOT validated."""
    cache, reward, utility, prefit = (meta[x] for x in ('cache', 'reward', 'utility', 'prefit'))
    logical = {key: LOGICAL + entry[0] for key, entry in METADATA.items()}
    for document in (reward, utility):
        bindings.descriptor(document['cache'], logical['cache'])
    bindings.descriptor(utility['parent_reward'], logical['reward'])
    for name, descriptor in reward['state_arrays'].items():
        expected = str(PurePosixPath(logical['cache']).parent / cache['arrays'][name]['path'])
        bindings.descriptor(descriptor, expected)
        require(descriptor['sha256'] == cache['arrays'][name]['sha256'], 'State/cache digest mismatch')
    support = str(PurePosixPath(logical['reward']).parent / 'support.npy')
    bindings.descriptor(utility['support'], support)
    require(utility['support']['sha256'] == reward['files']['support.npy']['sha256'], 'Parent support differs')
    require(prefit.get('both_cohorts_ready') is True and prefit.get('prefit_gate_passed') is True and
            prefit.get('state') == 'completed' and prefit['cohorts']['train']['nondegenerate_prior'] is True,
            'Incomplete prefit gate')
    require(set(prefit['utility_manifests']) == {'train', 'screen'}, 'Both prefit cohorts required')
    for cohort, key in (('train', 'utility'), ('screen', 'screen')):
        bindings.descriptor(prefit['utility_manifests'][cohort], logical[key])
        require(meta[key]['cohort'] == cohort and meta[key]['contract'] == prefit['contract'] and
                meta[key]['preregistration_sha256'] == METADATA['preregistration'][1] == prefit['preregistration_sha256'],
                'Prefit contract/cohort/preregistration differs')
    require(meta['heads']['bank_manifest_sha256'] == BANK_SHA and
            meta['heads']['teacher'] == meta['bank']['teacher'] == cache['teacher'], 'Head/bank teacher provenance differs')
    require(meta['heads']['feature_schema_sha256'] == cache['feature_schema_sha256'], 'Head/cache schema differs')
    require(cache['rows'] == 512 and cache['candidate_width'] == 1024 and cache['precision'] == 'cuda-fp16'
            and cache['cohort'] == 'train', 'Fixed train512 schema differs')


def plan(artifact_root, bank, output, root=ROOT, science=None):
    science = json.loads(json.dumps(SCIENCE if science is None else science))
    validate_science(science)
    root, artifacts, bank, output = map(safe_path, (root, artifact_root, bank, output))
    require(not output.exists() and output.parent.is_dir(), 'Fresh output with existing parent required')
    for tree in (root, artifacts, bank.parent):
        require(output != tree and not output.is_relative_to(tree) and not tree.is_relative_to(output),
                'Output/input overlap')
    sources = {}
    for relative, digest in SOURCE_PINS.items():
        path = child(root, relative)
        pinned_bytes(path, digest)
        sources[path.as_posix()] = digest
    beta = child(root, 'evidence/matched-kl/protocols/frozen-beta.json')
    require(pinned_json(beta, BETA_SHA)['beta'] == SCIENCE['utilities']['residual'], 'Frozen beta differs')
    bindings = Bindings()
    meta = {}
    for key, (relative, digest) in METADATA.items():
        logical = LOGICAL + relative
        bindings.add(logical, child(artifacts, relative), digest, 'metadata:' + key)
        meta[key] = bindings.read_json(logical)
    # bank's historical location is an opaque provenance key, not a host fallback.
    bank_logical = 'C:/Users/Михаил/.cache/codex-kion/three-head-262k-20261024-r1/train/bank.json'
    bindings.add(bank_logical, bank, BANK_SHA, 'metadata:bank')
    meta['bank'] = bindings.read_json(bank_logical)
    for key, field in (('cache', 'arrays'), ('reward', 'files'), ('utility', 'files')):
        relative = METADATA[key][0]
        for name, descriptor in meta[key][field].items():
            filename = descriptor.get('path', name)
            physical = child(child(artifacts, relative).parent, filename)
            logical = str(PurePosixPath(LOGICAL + relative).parent / filename)
            bindings.add(logical, physical, descriptor['sha256'], 'payload:' + key + ':' + name, descriptor.get('bytes'))
    for name in ('A', 'B', 'C'):
        relative = 'three-head-262k-20261024-r1/heads/' + name + '/head.pt'
        bindings.add(LOGICAL + relative, child(artifacts, relative), meta['heads']['checkpoint_sha256'][name], 'checkpoint:' + name)
    validate_metadata(meta, bindings)
    result = dict(contract=CONTRACT, mode='metadata-only-plan', execution_supported=False,
        scope='head-training preparation conditional on immutable historical train512 artifacts',
        output=output.as_posix(), scientific_configuration=science,
        proposed_fits=[dict(utility=u, beta=b, seed=s) for u, b in science['utilities'].items() for s in science['seeds']],
        source_pins=sources, beta_evidence=dict(path=beta.as_posix(), sha256=BETA_SHA),
        adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bindings=list(bindings.entries.values()), payload_content_verified=False,
        numerical_validation_performed=False, runtime_admitted=False, training_launched=False,
        raw_reproduction_complete=False,
        remaining=['resolved numerical cache/reward/utility validators', 'head-loading and parity-proof port',
                   'exclusive fit ledger, fresh deadline and resource monitor',
                   'full payload hash admission and CUDA FP16 all512 exact parity before optimizer',
                   'versioned training/checkpoint/diagnostic orchestration and independent review'])
    result['plan_sha256'] = canonical_hash(result)
    return result


def plan_destination(destination, request):
    destination = safe_path(destination)
    future_output = safe_path(request['output'])
    # The saved plan must not occupy the future directory or turn a required
    # input ancestor into a file. No filesystem mutation happens before this.
    protected = [future_output, *[safe_path(e['physical']) for e in request['bindings']],
                 *[safe_path(p) for p in request['source_pins']],
                 safe_path(request['beta_evidence']['path'])]
    for path in protected:
        require(destination != path and not destination.is_relative_to(path) and
                not path.is_relative_to(destination), 'Plan destination overlaps output/input')
    require(destination.parent.is_dir(), 'Plan parent must already exist')
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact-root', 'bank', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--plan-output', type=Path)
    args = vars(parser.parse_args(argv))
    destination = args.pop('plan_output')
    try:
        result = plan(**args)
        data = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if destination:
            destination = plan_destination(destination, result)
            with destination.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(data)
        print(data, end='')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'train_rl preparation: {exc}\n')


if __name__ == '__main__':
    main()
