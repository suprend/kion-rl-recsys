"""Fresh uninterrupted E1 -> E2, original H5 schedule; inert without execute flag.

The preserved run/update/evaluate/checkpoint functions own all numerical work.
This adapter owns input identity, finite admission, scope I/O and supervision.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import ModuleType

from gpt_inputs import identity, sha, read_small, json_bytes
from gpt_provider import BoundIO, FileBinding
from gpt_sources import RUNTIME_ROOT, RUNTIME_PINS, runtime_source_bindings
from input_inventory import checked_path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'D:/tlab-recsys-2026/artifacts/task2-kion'
SAMPLER = BASE + '/deadline-20261022/negative-worker-r2/sampler.py'
SAMPLER_SHA = '71d8747bfb080bc6ea82059c28b899e5c1bd266649deb9943de80de431933c44'
SCOPE_SHA = 'a674637029b96f564366fdda6e9c433a1cf3dab09721d191ad5a276b622cf962'
CONTRACT = 'portable-fresh-uninterrupted-E1-E2-H5-v1'
PROVIDER_MANIFEST_SHA = 'a8e6e78334c1ba1ca5519a09785f0fbd26972a1142d500e3403afb07c2a748bd'
FIXED = dict(provider='kion_complete_pack:load_training_provider', device='cuda',
    schedule_total_epochs=5, batch_size=128, accumulation=1, hidden_size=256,
    layers=4, heads=4, bottleneck=32, person_dim=32, seed=2026, patience=3,
    bucket_size=8192, checkpoint_updates=1000, log_updates=100, cpu_threads=2,
    lr=.0001, weight_decay=.01, warmup_ratio=.05, hard_negatives=0,
    uniform_negatives=3, auxiliary_weight=1., candidate_chunk_size=64,
    retrieval_topk=128, training_order='origin-grouped-v1', scoring_mode='rerank',
    integration_mode='scoped_production', text_variant='all',
    people_variants=['actor-mean', 'director-mean', 'actor-top3-concat'],
    allow_partial_integration=False, resume=False, resume_from='latest',
    max_updates=0, stop_after_validation=2)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_config(config):
    c = config['contract']
    for key in ('hidden_size', 'layers', 'heads', 'bottleneck', 'person_dim', 'seed',
                'text_variant', 'people_variants'):
        require(c['architecture'].get(key) == FIXED[key], 'architecture differs: ' + key)
    require(c['architecture'].get('context_length') == 64, 'context64 required')
    require({k: len(c['architecture']['feature_columns'][k]) for k in
             ('item', 'user', 'candidate')} == dict(item=5736, user=3252, candidate=1068),
            'full feature widths required')
    schedule = c['training_schedule']
    require(schedule['eligible_count'] == 3477437 and schedule['updates_per_epoch'] == 27235
            and schedule['mode'] == 'origin-grouped-v1' and schedule['microbatch'] == 128
            and schedule['accumulation'] == 1 and schedule['seed'] == 2026,
            'full origin-grouped H5 schedule required')
    for key, value in c['training'].items():
        require(key in FIXED and type(value) is type(FIXED[key]) and value == FIXED[key],
                'fixed training configuration differs: ' + key)
    for key in ('schedule_total_epochs', 'batch_size', 'accumulation', 'seed',
                'hard_negatives', 'uniform_negatives', 'auxiliary_weight', 'training_order'):
        require(c['training'].get(key) == FIXED[key], 'missing training configuration: ' + key)
    require(config['runtime']['cpu_threads'] == 2, 'uninterrupted fresh run uses CPU2')


def build_plan(request):
    """Bounded JSON/source reads and stat only; no provider or numerical imports."""
    keys = {'contract', 'data', 'feature_dir', 'scope_decision', 'manifest_sha256',
            'bindings', 'expected_config', 'expected_config_sha256', 'output',
            'max_seconds', 'min_free_disk_gib', 'max_rss_gib'}
    require(set(request) == keys and request['contract'] == CONTRACT, 'exact fresh request schema required')
    for key in ('data', 'feature_dir', 'scope_decision'):
        identity(request[key])
    sha(request['manifest_sha256'])
    for key, low, high in [('max_seconds', 1, 604800), ('min_free_disk_gib', 1, 4096), ('max_rss_gib', 1, 12)]:
        value = request[key]
        require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
                'invalid resource limit: ' + key)
    require(type(request['bindings']) is list and 1 <= len(request['bindings']) <= 8192,
            'finite exact file bindings required')
    rows = [FileBinding(**r) for r in request['bindings']]
    sources = runtime_source_bindings(ROOT, existing=rows)
    present = {row.logical for row in rows}
    rows += [row for row in sources if row.logical not in present]
    io = BoundIO(rows, lambda: None)
    manifest = io.bindings[request['feature_dir'].rstrip('/') + '/manifest.json']
    require(manifest.role == 'metadata' and manifest.sha256 == request['manifest_sha256'],
            'exact complete manifest binding required')
    output = checked_path(Path(request['output']))
    require(output.is_absolute() and not output.exists(), 'fresh absolute output required')
    protected = [ROOT / 'source', Path(request['expected_config'])] + [Path(r.physical) for r in rows]
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
            'output overlaps a protected input')
    raw = read_small(Path(request['expected_config']))
    require(hashlib.sha256(raw).hexdigest() == sha(request['expected_config_sha256']), 'expected config pin mismatch')
    config = json_bytes(raw)
    validate_config(config)
    sampler_path = ROOT / 'source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/deadline-20261022/negative-worker-r2/sampler.py'
    require(hashlib.sha256(read_small(sampler_path, 1 << 20)).hexdigest() == SAMPLER_SHA, 'sampler source changed')
    inventory = []
    for row in rows:
        path = Path(row.physical)
        inventory.append(dict(logical=row.logical, physical=row.physical, role=row.role,
                              sha256=row.sha256, available=path.is_file()))
    require(io.bindings[request['scope_decision']].sha256 == SCOPE_SHA, 'scope pin differs')
    return dict(mode='plan', training_executed=False, numerical_equivalence='untested',
                provider_admission='not-executed; transitive-closure-not-established',
                lineage=CONTRACT, historical_recovery=False, request=request,
                fixed=FIXED, expected_config=config, bindings=[vars(r) for r in rows],
                requirements=inventory, blocked_input=[r['logical'] for r in inventory if not r['available']],
                updates_per_epoch=27235, stop_updates=54470, schedule_denominator=136175,
                sampler_source=sampler_path.as_posix(), sampler_sha256=SAMPLER_SHA,
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def prepare_request(*, inventory_path, inventory_sha256, bindings_path, bindings_sha256,
                    expected_config, expected_config_sha256, output, request_path,
                    max_seconds=604800, min_free_disk_gib=10, max_rss_gib=12):
    """Compile a fresh request from explicitly byte-pinned finite inventories.

    Inventory presence is not scientific admission. The exact original provider
    manifest pin is additionally fixed here; runtime verifies all actual inputs.
    """
    def pinned_json(path, pin):
        data = read_small(Path(path))
        require(hashlib.sha256(data).hexdigest() == sha(pin), 'preparation input pin differs')
        return json_bytes(data)
    inventory = pinned_json(inventory_path, inventory_sha256)
    bindings = pinned_json(bindings_path, bindings_sha256)
    require(inventory.get('schema_version') == 1 and inventory.get('payloads_hashed') is False,
            'bounded transitive inventory schema required')
    entries = inventory.get('entries')
    require(type(entries) is list and 1 <= len(entries) <= 16384
            and type(bindings) is dict and 1 <= len(bindings) <= 8192, 'finite inventory/bindings required')
    rows = {}
    for entry in entries:
        logical = identity(entry['logical'])
        digest = sha(entry['sha256'])
        require(entry.get('kind') in ('metadata', 'source', 'payload'), 'unknown inventory kind')
        require(logical in bindings and entry.get('physical') == bindings[logical],
                'inventory physical path differs from pinned bindings')
        role = ('source' if logical.endswith('.py') else
                'metadata' if logical.endswith('.json') else 'payload')
        row = FileBinding(logical, bindings[logical], digest, role)
        require(logical not in rows or rows[logical] == row, 'conflicting inventory declarations')
        rows[logical] = row
    providers = [e for e in entries if e.get('sha256') == PROVIDER_MANIFEST_SHA
                 and e['logical'].endswith('/manifest.json')]
    require(len({e['logical'] for e in providers}) == 1, 'exact original complete manifest required')
    feature_dir = providers[0]['logical'].rsplit('/', 1)[0]
    config = pinned_json(expected_config, expected_config_sha256)
    scope = config['contract']['integration']['scope_decision']
    require(scope['sha256'] == SCOPE_SHA and scope['path'] in rows, 'expected config scope binding missing')
    # Original manifest metadata identifies the split root. Read it only after
    # checking its independent scientific byte pin, never inspect split arrays.
    manifest_row = rows[providers[0]['logical']]
    manifest = pinned_json(manifest_row.physical, PROVIDER_MANIFEST_SHA)
    data_identity = manifest['data']['path']
    identity(data_identity)
    for source in runtime_source_bindings(ROOT):
        if source.logical in rows:
            require(rows[source.logical].sha256 == source.sha256, 'retained source inventory pin differs')
        rows[source.logical] = source
    request = dict(contract=CONTRACT, data=data_identity, feature_dir=feature_dir,
        scope_decision=scope['path'], manifest_sha256=PROVIDER_MANIFEST_SHA,
        bindings=[vars(r) for r in rows.values()], expected_config=Path(expected_config).as_posix(),
        expected_config_sha256=expected_config_sha256, output=Path(output).as_posix(),
        max_seconds=max_seconds, min_free_disk_gib=min_free_disk_gib, max_rss_gib=max_rss_gib)
    plan = build_plan(request)
    destination = checked_path(Path(request_path))
    receipt_path = destination.with_suffix('.preparation.json')
    require(destination.is_absolute() and not destination.exists() and not receipt_path.exists(),
            'new absolute request and preparation receipt paths required')
    protected = [ROOT / 'source', Path(output), Path(expected_config), Path(inventory_path), Path(bindings_path)]
    protected += [Path(row.physical) for row in rows.values()]
    require(all(not dest.is_relative_to(p) and not p.is_relative_to(dest)
                for dest in (destination, receipt_path) for p in protected),
            'request publication overlaps inputs or training output')
    payload = (json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    receipt = dict(state='request-prepared-not-admitted', training_executed=False,
        request_sha256=hashlib.sha256(payload).hexdigest(),
        inventory=dict(path=Path(inventory_path).as_posix(), sha256=inventory_sha256),
        bindings=dict(path=Path(bindings_path).as_posix(), sha256=bindings_sha256),
        provider_manifest_sha256=PROVIDER_MANIFEST_SHA,
        expected_config_sha256=expected_config_sha256, adapter_sha256=plan['adapter_sha256'],
        blocked_input=plan['blocked_input'], provider_admission=plan['provider_admission'])
    with destination.open('xb') as stream:
        stream.write(payload)
    with receipt_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return receipt


def rss_bytes():
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Memory()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError('cannot inspect worker RSS')
        return counters.WorkingSetSize
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


class Guard:
    def __init__(self, request):
        self.request, self.started = request, time.monotonic()
        self.output = Path(request['output'])
        self.deadline = datetime.now(timezone.utc) + timedelta(seconds=request['max_seconds'])
        self.calls, self.peak_rss = 0, 0

    def __call__(self):
        self.calls += 1
        if (time.monotonic() - self.started >= self.request['max_seconds']
                or datetime.now(timezone.utc) >= self.deadline):
            raise TimeoutError('fresh training deadline exhausted')
        if (self.output / 'stop-request.json').exists():
            raise InterruptedError('fresh training cancelled')
        probe = self.output
        while not probe.exists():
            probe = probe.parent
        require(shutil.disk_usage(probe).free >= self.request['min_free_disk_gib'] * 2**30,
                'free disk reserve exhausted')
        self.peak_rss = max(self.peak_rss, rss_bytes())
        require(self.peak_rss <= self.request['max_rss_gib'] * 2**30, 'worker RSS cap exhausted')


def scoped_integration(provider, args):
    """Original scoped checks, with bound read and logical comparison separated."""
    require(args.integration_mode == 'scoped_production' and not args.allow_partial_integration,
            'scoped full-row training only')
    logical = identity(str(args.scope_decision).replace('\\', '/'))
    decision = provider.io.read_json(logical, expected=SCOPE_SHA)
    declaration = provider.contract()
    pin = dict(path=logical, sha256=SCOPE_SHA)
    active = decision['active_families']
    acceptance = declaration.get('family_acceptance', {})
    require(decision.get('contract') == 'kion-user-approved-ready-scope-v1'
        and decision.get('scope_id') == 'ready-pack-20261020-r1'
        and len(active) == len(set(active)) == 8
        and declaration.get('status') == 'admitted_scoped'
        and declaration.get('integration_mode') == 'scoped_production'
        and declaration.get('scope_id') == 'ready-pack-20261020-r1'
        and declaration.get('scope_decision') == pin and declaration.get('active_families') == active
        and set(acceptance) == set(active) and all(v == 'accepted_scoped' for v in acceptance.values())
        and set(declaration.get('materialized_families', [])) == set(active)
        and declaration.get('full_package_admitted') is False
        and declaration.get('full_row_fit_admitted') is True, 'scope/family admission differs')
    return dict(mode='scoped_production', status='admitted_scoped', scope_id='ready-pack-20261020-r1',
                scope_decision=pin, active_families=active, full_package_admitted=False, full_row_fit_admitted=True)


def run_bound(train, sampler, provider, args, expected, check):
    """Explicit narrow callbacks; unchanged numerical run and candidate sampler."""
    originals = {n: getattr(train, n) for n in ('integration_contract', 'make_config', 'update',
                                              'evaluate', 'checkpoint', 'sha256')}
    install = sampler.InstalledSampler(train)
    checked = False
    def checked_config(*a, **kw):
        nonlocal checked
        check()
        config = originals['make_config'](*a, **kw)
        validate_config(config)
        require(config == expected, 'exact expected architecture/provider/schedule/runtime config differs')
        require(provider.metadata['context'] == 64 and len(provider.splits['train']['y']) == 3477437
                and len(provider.splits['val']['y']) == 158760, 'full context64 train/val census differs')
        checked = True
        return config
    def guarded(fn):
        def call(*a, **kw):
            check()
            result = fn(*a, **kw)
            check()
            return result
        return call
    def source_or_output_hash(path):
        path = Path(path)
        if path.suffix == '.py':
            name = path.stem
            require(name in provider.runtime.pins, 'undeclared config source')
            logical, pin = provider.runtime.pins[name]
            provider.io.verify(logical, 'source', pin)
            return pin
        return guarded(originals['sha256'])(path)
    try:
        train.integration_contract = scoped_integration
        train.make_config = checked_config
        train.sha256 = source_or_output_hash
        install.install()
        for name in ('update', 'evaluate', 'checkpoint'):
            setattr(train, name, guarded(originals[name]))
        check()
        result = train.run(args, provider)
        check()
        require(checked and result['status'] == 'stopped'
                and result['reason'] == 'runtime_stop_after_validation', 'incomplete E2 run')
        for key, value in dict(epoch=2, offset=0, group_index=0, updates=54470, examples=6954874).items():
            require(result['state'].get(key) == value, 'final E2 cursor differs: ' + key)
        return dict(result=result, sampler_contract=sampler.CONTRACT,
                    sampler_audit=dict(install.replacement.audit))
    finally:
        install.restore()
        for name, value in originals.items():
            setattr(train, name, value)


def execute(request):
    plan = build_plan(request)  # Never trust a mutable previously admitted plan.
    require(not plan['blocked_input'], 'missing declared inputs')
    guard = Guard(request)
    guard()
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(lineage=CONTRACT, historical_recovery=False, historical_weights_claim=False,
                   numerical_equivalence='untested', state='started', execution_attempted=True,
                   training_executed='unknown-until-loop-return', plan=plan)
    def write(name, value):
        with (output / name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    write('admission.json', receipt)
    try:
        from gpt_complete import CompleteProvider
        bindings = [FileBinding(**r) for r in plan['bindings']]
        with CompleteProvider.load(request['data'], request['feature_dir'], bindings=bindings,
                check=guard, scope_decision=request['scope_decision'],
                manifest_sha256=request['manifest_sha256'], external=('numpy', 'torch', 'transformers')) as provider:
            train = provider.runtime.load('kion_complete_train')
            args = train.parser().parse_args(['--data', request['data'], '--feature-dir', request['feature_dir'],
                                              '--output', str(output / 'model')])
            for name, value in FIXED.items():
                setattr(args, name, value)
            args.scope_decision = Path(request['scope_decision'])
            args.stop_file = output / 'stop-request.json'
            args.deadline_utc = guard.deadline.isoformat()
            data = read_small(Path(plan['sampler_source']), 1 << 20)
            require(hashlib.sha256(data).hexdigest() == SAMPLER_SHA, 'sampler changed after planning')
            sampler = ModuleType('_portable_original_sampler')
            exec(compile(data, SAMPLER, 'exec'), sampler.__dict__)
            result = run_bound(train, sampler, provider, args, plan['expected_config'], guard)
            receipt.update(state='completed-fresh-E2', training_executed=True, **result,
                           source_receipt=provider.runtime.receipt(), provider_receipt=provider.source_receipt)
    except BaseException as error:
        receipt.update(state='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        receipt.update(guard_calls=guard.calls, peak_rss_bytes=guard.peak_rss,
                       elapsed_seconds=time.monotonic() - guard.started, deadline_utc=guard.deadline.isoformat())
        write('execution.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--execute-training', action='store_true')
    parser.add_argument('--prepare-request', action='store_true')
    for name in ('inventory', 'inventory-sha256', 'bindings', 'bindings-sha256',
                 'expected-config', 'expected-config-sha256', 'output', 'write-request'):
        parser.add_argument('--'+name)
    parser.add_argument('--max-seconds', type=float, default=604800)
    parser.add_argument('--min-free-disk-gib', type=float, default=10)
    parser.add_argument('--max-rss-gib', type=float, default=12)
    args = parser.parse_args(argv)
    if args.prepare_request:
        if args.execute_training or args.request:
            parser.error('request preparation cannot activate training or consume --request')
        fields = ('inventory', 'inventory_sha256', 'bindings', 'bindings_sha256',
                  'expected_config', 'expected_config_sha256', 'output', 'write_request')
        if any(getattr(args, key) is None for key in fields):
            parser.error('preparation requires all input paths/pins and new request/output paths')
        result = prepare_request(inventory_path=args.inventory, inventory_sha256=args.inventory_sha256,
            bindings_path=args.bindings, bindings_sha256=args.bindings_sha256,
            expected_config=args.expected_config, expected_config_sha256=args.expected_config_sha256,
            output=args.output, request_path=args.write_request, max_seconds=args.max_seconds,
            min_free_disk_gib=args.min_free_disk_gib, max_rss_gib=args.max_rss_gib)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if args.request is None:
        parser.error('--request required unless preparing a new request')
    request = json_bytes(read_small(args.request))
    result = execute(request) if args.execute_training else build_plan(request)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
