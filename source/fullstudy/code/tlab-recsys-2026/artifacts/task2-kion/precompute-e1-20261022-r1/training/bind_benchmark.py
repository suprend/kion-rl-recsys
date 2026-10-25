"""Bind fresh shortbench source/metadata only; no approval, DB read or job.

Run after a completed real pilot. Physical SQLite validation belongs inside the
owned GPU worker via the pinned store hook. This helper reads <=8MiB files only.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import worker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT.parent / 'deadline-20261022/control-r2'
MAX_SMALL = 8 << 20


def small_bytes(path):
    path = Path(path)
    if path.suffix.lower() not in ('.py', '.json', '.exe'):
        raise ValueError('Source/JSON/OS executable only; no DB or raw array access')
    if path.stat().st_size > MAX_SMALL:
        raise ValueError('Only <=8MiB source/metadata reads allowed: ' + path.as_posix())
    return path.read_bytes()


def read(path):
    return json.loads(small_bytes(path).decode('utf-8'))


def small_pin(path, expected=None):
    path = Path(path).resolve()
    digest = hashlib.sha256(small_bytes(path)).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Frozen/pilot source metadata changed: ' + path.as_posix())
    return {'path': path.as_posix(), 'sha256': digest}


def payload(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def guard_metadata():
    """Import only current guard's stdlib definitions, no helper/native calls."""
    path = ROOT / 'execution/native_phase_guard.py'
    code = small_bytes(path)
    spec = importlib.util.spec_from_file_location('_shortbench_guard_metadata', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(code, str(path), 'exec'), module.__dict__)
    return module


def replace_option(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv.extend([flag, value])


def build(pilot_request_path, variant, name, *, now=None, cpu_affinity=(4, 5), guard=None):
    if variant not in ('baseline', 'prepared'):
        raise ValueError('Only baseline/prepared variants supported')
    if not re.fullmatch(r'bench-' + variant + r'-[a-z0-9][a-z0-9-]*', name):
        raise ValueError('Name must be a fresh bench-VARIANT-suffix sibling')
    if len(cpu_affinity) != 2 or len(set(cpu_affinity)) != 2 or any(type(x) is not int or not 0 <= x < 64 for x in cpu_affinity):
        raise ValueError('Two explicit distinct CPU indexes required')
    now = datetime.now(timezone.utc) if now is None else now
    if now.utcoffset() != timedelta(0):
        raise ValueError('Fresh UTC clock required')
    deadline = (now + timedelta(minutes=30)).isoformat()
    pilot_request_path = Path(pilot_request_path).resolve()
    if not pilot_request_path.is_relative_to(ROOT):
        raise ValueError('Pilot request must belong to this operation')
    pilot = read(pilot_request_path)
    if pilot.get('phase') != 'cpu-pilot' or pilot.get('mode') != 'pilot':
        raise ValueError('Exact CPU pilot request required')
    pilot_output = Path(pilot['output']).resolve()
    if not pilot_output.is_relative_to(ROOT) or pilot_output == ROOT:
        raise ValueError('Pilot output must belong to this operation')
    # Metadata only: do not stat/hash/open the physical DB in this parent helper.
    pilot_result_path = pilot_output / 'result.json'
    result = read(pilot_result_path)
    expected_store = pilot_output / 'queries.sqlite3'
    # Lexical normalization only. Path.resolve() on a Windows DB filename may
    # open the physical object for final-path canonicalization.
    declared_store = Path(result['store_path'])
    store_matches = (declared_store.is_absolute()
        and os.path.normcase(os.path.abspath(declared_store)) == os.path.normcase(os.path.abspath(expected_store)))
    if (result.get('state') != 'completed' or result.get('mode') != 'pilot'
            or not store_matches
            or result.get('store_source_pins') != pilot['store_source_pins']):
        raise ValueError('Completed exact pilot metadata/store binding required')
    selection_path = Path(result['selection_path']).resolve()
    if selection_path != pilot_output / 'coverage-selection.json':
        raise ValueError('Exact pilot coverage selection path required')
    small_pin(selection_path, result['selection_sha256'])
    selection = read(selection_path)
    groups = selection['first12_epoch0_groups']
    if (len(groups) != 12 or [g['group_index'] for g in groups] != list(range(12))
            or not all(set(g['rows']).issubset(selection['selected_rows']['train']) for g in groups)):
        raise ValueError('Pilot must explicitly include all first12 epoch0 groups')

    run_dir = ROOT / name
    request_path, plan_path = HERE / (name + '-request.json'), HERE / (name + '-plan.json')
    if run_dir.exists() or request_path.exists() or plan_path.exists():
        raise FileExistsError('Fresh run/binding names required; no overwrite or automatic retry')
    prior_request_path = PREVIOUS / 'worker-request.json'
    prior = read(prior_request_path)
    argv = list(prior['train_argv'])
    output = run_dir / 'worker'
    stop = run_dir / 'stop-request.json'
    for flag, value in (('--output', (output / 'model').as_posix()), ('--stop-file', stop.as_posix()),
                        ('--deadline-utc', deadline), ('--max-updates', '12'), ('--stop-after-validation', '0')):
        replace_option(argv, flag, value)
    guard = guard_metadata() if guard is None else guard
    sources = {}
    for path, expected in pilot['sources'].items():
        record = small_pin(path, expected)
        sources[record['path']] = record['sha256']
    additions = [Path(__file__), HERE / 'worker.py', ROOT / 'execution/native_phase_guard.py']
    additions += list(worker.ANCHORS)
    for path in additions:
        record = small_pin(path, worker.ANCHORS.get(path))
        if record['path'] in sources and record['sha256'] != sources[record['path']]:
            raise ValueError('Execution source changed since pilot; explicit migration/review required')
        sources[record['path']] = record['sha256']
    for path, expected in guard.HELPERS.values():
        record = small_pin(path, expected)
        sources[record['path']] = record['sha256']
    common = Path(pilot['common_scripts']).resolve()
    for name_, expected in worker.FROZEN.items():
        record = small_pin(common / name_, expected)
        sources[record['path']] = record['sha256']
    provider_source = small_pin(common / 'kion_complete_pack.py',
                                pilot['sources'][(common / 'kion_complete_pack.py').as_posix()])
    hook = small_pin(ROOT / 'preparation/store_hook.py',
                     pilot['sources'][(ROOT / 'preparation/store_hook.py').as_posix()])
    for descriptor in (pilot['codec_source'], pilot['query_method_source']):
        small_pin(descriptor['path'], descriptor['sha256'])
    # Accepted config is ~8.95MB. Obtain its immutable hash from the small
    # completed E1 receipt; root guard/owned worker verify the actual config.
    prior_completion_path = PREVIOUS / 'job/singlepass-completion.json'
    prior_completion = read(prior_completion_path)
    if (prior_completion.get('state') != 'completed'
            or prior_completion.get('operation') != 'kion-D22-retrieval128-uniform3-E1-singlepass-v1'):
        raise ValueError('Accepted corrected E1 provenance receipt required')
    config_digest = prior_completion['sha256']['model/config.json']
    if not re.fullmatch('[0-9a-f]{64}', config_digest):
        raise ValueError('Invalid accepted config digest')
    expected_config = {'path': (PREVIOUS / 'job/model/config.json').as_posix(), 'sha256': config_digest}
    interpreter = small_pin(guard.BASE, guard.BASE_SHA)
    request = {'contract': worker.CONTRACT, 'state': 'admitted', 'mode': 'benchmark',
        'phase': 'gpu-shortbench', 'variant': variant, 'output': output.as_posix(), 'output_dir': output.as_posix(),
        'stop_file': stop.as_posix(), 'deadline_utc': deadline, 'maximum_seconds': 1200,
        'gpu_uuid': guard.GPU, 'runtime_root': common.parent.as_posix(), 'interpreter': interpreter,
        'source_hashes': sources, 'expected_config': expected_config, 'train_argv': argv,
        'benchmark_updates': 12, 'resource_caps': dict(worker.CAPS),
        'provider_hook': {'path': hook['path'], 'sha256': hook['sha256'], 'install_name': 'install',
            'options': {'enabled': variant == 'prepared', 'store_path': expected_store.as_posix(),
                'store_source_pins': pilot['store_source_pins'], 'provider_source': provider_source,
                'codec_source': pilot['codec_source'], 'query_method_source': pilot['query_method_source'],
                'allow_missing_fallback': True, 'purpose': 'pilot-shortbench'}},
        'required_postchecks': {'same_state_pair_with_other_variant': True,
            'prepared_store_misses': 0, 'prepared_fallback_prepares': 0,
            'no_full_epoch_or_quality_claim': True},
        'binding_status': 'source-bound-pending-independent-native-plan-approval-no-job'}
    metadata = {value['path']: value['sha256'] for value in pilot['metadata'].values()}
    metadata_files = [pilot_request_path, pilot_result_path, selection_path, prior_request_path, prior_completion_path,
                      ROOT / 'contract.json', ROOT / 'resource-admission-r1.json',
                      HERE / 'review-independent-r1.json', HERE / 'source-proof.json']
    for path in metadata_files:
        record = small_pin(path)
        metadata[record['path']] = record['sha256']
    metadata[expected_config['path']] = expected_config['sha256']
    request_payload = payload(request)
    worker_pin = small_pin(HERE / 'worker.py')
    console = Path(os.environ['SYSTEMROOT']) / 'System32/conhost.exe'
    smi = Path(os.environ['SYSTEMROOT']) / 'System32/nvidia-smi.exe'
    plan = {'contract': 'kion-precompute-e1-phase-v1', 'phase': 'gpu-shortbench',
        'interpreter': interpreter, 'worker': worker_pin,
        'worker_request': {'path': request_path.as_posix(), 'sha256': hashlib.sha256(request_payload).hexdigest()},
        'worker_argv': [guard.BASE.as_posix(), '-B', worker_pin['path'], '--request', request_path.as_posix()],
        'max_seconds': 1200, 'deadline_utc': deadline, 'run_dir': run_dir.as_posix(), 'cwd': ROOT.as_posix(),
        'cpu_affinity': list(cpu_affinity), 'console_host': small_pin(console), 'nvidia_smi': small_pin(smi),
        'selected_gpu_uuid': guard.GPU, 'source_hashes': sources, 'metadata_pins': metadata,
        'resource_limits': guard.phase_limits('gpu-shortbench')}
    return request_path, request_payload, plan_path, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-request', type=Path, required=True)
    parser.add_argument('--variant', choices=('baseline', 'prepared'), required=True)
    parser.add_argument('--name')
    parser.add_argument('--cpu-affinity', type=int, nargs=2, default=[4, 5])
    cli = parser.parse_args(argv)
    name = cli.name or ('bench-' + cli.variant + '-r1')
    request_path, request_payload, plan_path, plan = build(cli.pilot_request, cli.variant, name,
                                                        cpu_affinity=cli.cpu_affinity)
    with request_path.open('xb') as stream:
        stream.write(request_payload)
    with plan_path.open('xb') as stream:
        stream.write(payload(plan))
    print(json.dumps({'state': 'source_only_bound_no_approval_no_job', 'request': request_path.as_posix(),
                      'plan': plan_path.as_posix(), 'plan_sha256': canonical(plan),
                      'deadline_utc': plan['deadline_utc'], 'physical_database_access': False}))


if __name__ == '__main__':
    main()
