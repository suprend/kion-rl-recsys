"""Metadata-only conditional binding after closed complete preparation.

No GO/approval/job is created. Physical database verification remains inside the
owned GPU worker. Existing worker and shortbench binder are imported unchanged.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import bind_benchmark as binding
import worker

HERE, ROOT = binding.HERE, binding.ROOT
GPU_GUARD_SHA = '0b45946f866b59aae94f9cf2f1eaa6a550641c849d9468eb04d6d6b4c9024e1f'
TRAINING_SHA = 'd3134effd421ad37abde9268235c09d9ea6b3c1a0c3f505f6c6dfe2514d8c001'
PREPARATION_SHA = 'bb220140d90c0c3e032e96f3c79c973440138c44e8f2d20bc8a2ce959f7edc1d'
PREPARATION_GUARD_SHA = '9d25a5942cda0be54e56297c9c71dd28a3f57307165288c7b372af0e22a2cf2a'
STORE = 'E:/codex-kion-cache-20261022-e1-r1/queries.sqlite3'
ROWS = {'train': 3477437, 'val': 158760}
POLICY = 'kion-finite-precompute-to-E1-root-policy-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_json(record):
    actual = binding.small_pin(record['path'], record['sha256'])
    return binding.read(actual['path']), actual


def exact_record(record, path, digest=None):
    require(Path(record['path']).resolve() == Path(path).resolve(), 'Conditional policy prerequisite path differs')
    if digest is not None:
        require(record['sha256'] == digest, 'Conditional policy frozen source differs')
    return binding.small_pin(path, record['sha256'])


def lexical_store(path):
    # Never resolve/stat/open the physical DB in this metadata owner.
    value = Path(path)
    return value.is_absolute() and os.path.normcase(os.path.abspath(value)) == os.path.normcase(os.path.abspath(STORE))


def utc(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(value.utcoffset() == timedelta(0), 'Explicit UTC timestamp required')
    return value


def check_policy(policy, deadline, maximum, now):
    require(policy.get('contract') == POLICY and policy.get('state') == 'GO-finite-precompute-to-E1',
            'Explicit finite conditional root policy required')
    require(type(maximum) is int and 1 <= maximum <= 24000, 'E1 duration must be1..24000 seconds')
    cap = policy['e1_max_seconds']
    require(type(cap) is int and maximum <= cap <= 24000, 'E1 phase exceeds conditional policy cap')
    start, stop = utc(policy['pipeline_started_utc']), utc(policy['pipeline_deadline_utc'])
    total = policy['pipeline_max_seconds']
    require(type(total) is int and total > 0 and start <= now < stop
            and stop <= start + timedelta(seconds=total), 'Finite global pipeline budget required')
    require(now + timedelta(seconds=maximum + 45) <= deadline <= stop,
            'E1 plus45s cleanup must fit fresh phase and global deadlines')


def preparation_basis(request_path, result_path, guard_path, policy):
    request_path, result_path, guard_path = map(lambda p: Path(p).resolve(), (request_path, result_path, guard_path))
    require(all(path.is_relative_to(ROOT) for path in (request_path, result_path, guard_path)),
            'Preparation metadata must belong to this operation')
    request_pin = exact_record(policy['preparation_request'], request_path)
    request = binding.read(request_path)
    plan, plan_pin = pinned_json(policy['preparation_plan'])
    require(Path(plan_pin['path']).resolve().is_relative_to(ROOT), 'Preparation plan must belong to this operation')
    require(plan['phase'] == request.get('phase') == 'cpu-build' and request.get('mode') == 'full'
            and request.get('expected_rows') == ROWS, 'Exact full preparation regime required')
    require(plan['worker_request'] == request_pin
            and Path(result_path) == Path(request['output']).resolve() / 'result.json'
            and guard_path == Path(plan['run_dir']).resolve() / 'guard.json', 'Preparation output/request/plan lineage differs')
    prep_worker = exact_record(plan['worker'], ROOT / 'full-preparation/worker.py', PREPARATION_SHA)
    require(request['sources'].get(prep_worker['path']) == PREPARATION_SHA
            and plan['source_hashes'].get((ROOT / 'full-execution/native_phase_guard.py').as_posix()) == PREPARATION_GUARD_SHA,
            'Reviewed full preparation worker/guard source pins required')
    for path, digest in request['sources'].items():
        require(plan['source_hashes'].get(path) == digest, 'Preparation plan/request source mismatch')
        binding.small_pin(path, digest)
    for record in request['metadata'].values():
        binding.small_pin(record['path'], record['sha256'])
    result, cleanup = binding.read(result_path), binding.read(guard_path)
    require(cleanup.get('state') == 'completed' and cleanup.get('exit_code') == 0
            and cleanup.get('cleanup_quiescent') is True and cleanup.get('audit_cleanup_quiescent') is True
            and cleanup.get('reason') is None and cleanup.get('log_error') is None
            and cleanup.get('phase') == 'cpu-build' and cleanup.get('plan_sha256') == binding.canonical(plan)
            and cleanup.get('worker_argv') == plan['worker_argv'], 'Preparation must exit0 and positively close exact native phase')
    require(result.get('state') == 'completed' and result.get('mode') == 'full'
            and result.get('phase') == 'cpu-build' and result.get('split_rows') == ROWS
            and result.get('CUDA_initialized') is False and lexical_store(request['store_path'])
            and lexical_store(result['store_path']) and result.get('parity', {}).get('state') == 'exact_bytes_equal',
            'Completed full-row exact-byte preparation required')
    pins = request['store_source_pins']
    require(result.get('store_source_pins') == pins
            and result.get('store_source_pins_sha256') == binding.canonical(pins), 'Prepared store source binding differs')
    for path, digest in request['sources'].items():
        require(pins.get(path) == digest, 'Store must bind every preparation source')
    for record in request['metadata'].values():
        require(pins.get(record['path']) == record['sha256'], 'Store must bind every preparation metadata input')
    require(isinstance(result.get('store_sha256'), str) and re.fullmatch('[0-9a-f]{64}', result['store_sha256']),
            'Owned preparation physical DB hash receipt required')
    go, go_pin = pinned_json(request['full_build_decision'])
    pilot, pilot_pin = pinned_json(go['pilot_result'])
    require(go.get('state') == 'GO' and go.get('mode') == 'full'
            and go.get('store_source_pins_sha256') == binding.canonical(pins)
            and pilot.get('state') == 'completed' and pilot.get('mode') == 'pilot'
            and pilot.get('parity', {}).get('state') == 'exact_bytes_equal'
            and result.get('personal_schema_hash') == pilot.get('personal_schema_hash')
            and isinstance(result.get('personal_schema_hash'), str) and re.fullmatch('[0-9a-f]{64}', result['personal_schema_hash']),
            'Prepared personal schema/full-build decision/pilot binding differs')
    extra = [binding.small_pin(result_path), binding.small_pin(guard_path), request_pin, plan_pin, go_pin, pilot_pin]
    require(result.get('storage_extension') == request['storage_extension'], 'Storage extension binding differs')
    extra.append(binding.small_pin(request['storage_extension']['path'], request['storage_extension']['sha256']))
    # The ownership marker is small JSON metadata; physical DB remains untouched.
    expected_marker = Path(STORE).with_name('cache-owner.json')
    declared_marker = Path(result['cache_owner_marker'])
    require(declared_marker.is_absolute()
            and os.path.normcase(os.path.abspath(declared_marker)) == os.path.normcase(os.path.abspath(expected_marker)),
            'Exact external cache owner marker required')
    extra.append(binding.small_pin(result['cache_owner_marker'], result['cache_owner_marker_sha256']))
    return request, result, extra


def build(preparation_request, preparation_result, preparation_guard, root_decision,
          deadline_utc, maximum_seconds, *, now=None, guard=None):
    now = datetime.now(timezone.utc) if now is None else now
    require(now.utcoffset() == timedelta(0), 'UTC clock required')
    decision_path = Path(root_decision).resolve()
    require(decision_path.is_relative_to(ROOT), 'Root conditional decision belongs to this operation')
    policy, decision_pin = binding.read(decision_path), binding.small_pin(decision_path)
    deadline = utc(deadline_utc)
    check_policy(policy, deadline, maximum_seconds, now)
    sources = {}
    evidence = policy['source_evidence']
    require(type(evidence) is list and len(evidence) > 0, 'Reviewed source/evidence pins required')
    evidence_pins = [binding.small_pin(record['path'], record['sha256']) for record in evidence]
    paired_path = ROOT / 'paired-prefix-result-r1.json'
    require(any(record['path'] == paired_path.as_posix() for record in evidence_pins), 'Actual paired prefix evidence pin required')
    paired = binding.read(paired_path)
    require(paired.get('state') == 'completed-actual-paired-prefix'
            and paired.get('parity', {}).get('same_state_parity') is True
            and paired.get('parity', {}).get('successful_updates_per_variant') == 12,
            'Actual12-update state parity evidence required')
    prep, result, prep_pins = preparation_basis(preparation_request, preparation_result, preparation_guard, policy)
    run_dir = ROOT / 'full-e1-r1'
    request_path, plan_path = HERE / 'full-e1-r1-request.json', HERE / 'full-e1-r1-plan.json'
    require(not any(path.exists() for path in (run_dir, request_path, plan_path)), 'Fresh full-e1-r1 only; no overwrite or retry')
    require(exact_record(policy['gpu_guard'], ROOT / 'execution/native_phase_guard.py', GPU_GUARD_SHA)['sha256'] == GPU_GUARD_SHA,
            'Original reviewed GPU guard required')
    exact_record(policy['training_worker'], HERE / 'worker.py', TRAINING_SHA)
    guard = binding.guard_metadata() if guard is None else guard
    for path, expected in prep['sources'].items():
        record = binding.small_pin(path, expected)
        sources[record['path']] = record['sha256']
    additions = [(HERE / 'worker.py', TRAINING_SHA), (HERE / 'bind_benchmark.py', None),
                 (Path(__file__), None), (ROOT / 'execution/native_phase_guard.py', GPU_GUARD_SHA)]
    additions += list(worker.ANCHORS.items()) + list(guard.HELPERS.values())
    for path, expected in additions:
        record = binding.small_pin(path, expected)
        require(record['path'] not in sources or sources[record['path']] == record['sha256'],
                'Frozen execution source changed since preparation')
        sources[record['path']] = record['sha256']
    common = Path(prep['common_scripts']).resolve()
    for name, expected in worker.FROZEN.items():
        record = binding.small_pin(common / name, expected)
        sources[record['path']] = record['sha256']
    prior_request_path = binding.PREVIOUS / 'worker-request.json'
    prior_completion_path = binding.PREVIOUS / 'job/singlepass-completion.json'
    prior, completed = binding.read(prior_request_path), binding.read(prior_completion_path)
    require(completed.get('state') == 'completed'
            and completed.get('operation') == 'kion-D22-retrieval128-uniform3-E1-singlepass-v1', 'Accepted corrected E1 provenance required')
    expected_config = {'path': (binding.PREVIOUS / 'job/model/config.json').as_posix(),
                       'sha256': completed['sha256']['model/config.json']}
    require(re.fullmatch('[0-9a-f]{64}', expected_config['sha256']) is not None, 'Accepted config digest invalid')
    output, stop = run_dir / 'worker', run_dir / 'stop-request.json'
    argv = list(prior['train_argv'])
    for flag, value in (('--output', (output / 'model').as_posix()), ('--stop-file', stop.as_posix()),
                        ('--deadline-utc', deadline.isoformat()), ('--max-updates', '0'), ('--stop-after-validation', '1')):
        binding.replace_option(argv, flag, value)
    hook = binding.small_pin(ROOT / 'preparation/store_hook.py', prep['sources'][(ROOT / 'preparation/store_hook.py').as_posix()])
    provider = binding.small_pin(common / 'kion_complete_pack.py', prep['sources'][(common / 'kion_complete_pack.py').as_posix()])
    coverage = binding.small_pin(preparation_result)
    interpreter = binding.small_pin(guard.BASE, guard.BASE_SHA)
    lineage = {'mode': 'programmatic-root-conditional-policy-pending-native-plan-approval',
               'policy': decision_pin, 'future_human_plan_review_claim': False,
               'approval_generated_by_helper': False, 'job_started_by_helper': False}
    request = {'contract': worker.CONTRACT, 'state': 'admitted', 'phase': 'gpu-e1', 'mode': 'full-e1',
        'variant': 'prepared', 'output': output.as_posix(), 'output_dir': output.as_posix(), 'stop_file': stop.as_posix(),
        'deadline_utc': deadline.isoformat(), 'maximum_seconds': maximum_seconds, 'gpu_uuid': guard.GPU,
        'runtime_root': common.parent.as_posix(), 'interpreter': interpreter, 'source_hashes': sources,
        'expected_config': expected_config, 'train_argv': argv, 'benchmark_updates': 12,
        'resource_caps': dict(worker.CAPS), 'conditional_admission': lineage,
        'provider_hook': {'path': hook['path'], 'sha256': hook['sha256'], 'install_name': 'install',
            'options': {'enabled': True, 'purpose': 'full-E1', 'allow_missing_fallback': False,
                'coverage_receipt': coverage, 'store_path': result['store_path'],
                'store_source_pins': prep['store_source_pins'], 'provider_source': provider,
                'codec_source': prep['codec_source'], 'query_method_source': prep['query_method_source']}}}
    metadata = {record['path']: record['sha256'] for record in prep['metadata'].values()}
    for record in prep_pins + evidence_pins + [decision_pin, expected_config,
            binding.small_pin(prior_request_path), binding.small_pin(prior_completion_path),
            binding.small_pin(ROOT / 'contract.json'), binding.small_pin(ROOT / 'resource-admission-r1.json')]:
        metadata[record['path']] = record['sha256']
    encoded = binding.payload(request)
    worker_pin = binding.small_pin(HERE / 'worker.py', TRAINING_SHA)
    system = Path(os.environ['SYSTEMROOT']) / 'System32'
    plan = {'contract': 'kion-precompute-e1-phase-v1', 'phase': 'gpu-e1', 'interpreter': interpreter,
        'worker': worker_pin, 'worker_request': {'path': request_path.as_posix(),
            'sha256': binding.hashlib.sha256(encoded).hexdigest()},
        'worker_argv': [guard.BASE.as_posix(), '-B', worker_pin['path'], '--request', request_path.as_posix()],
        'max_seconds': maximum_seconds, 'deadline_utc': deadline.isoformat(), 'run_dir': run_dir.as_posix(),
        'cwd': ROOT.as_posix(), 'cpu_affinity': [4, 5], 'console_host': binding.small_pin(system / 'conhost.exe'),
        'nvidia_smi': binding.small_pin(system / 'nvidia-smi.exe'), 'selected_gpu_uuid': guard.GPU,
        'source_hashes': sources, 'metadata_pins': metadata, 'resource_limits': guard.phase_limits('gpu-e1'),
        'conditional_admission': lineage}
    return request_path, encoded, plan_path, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--emit', action='store_true')
    for name in ('preparation-request', 'preparation-result', 'preparation-guard', 'root-decision', 'deadline-utc'):
        parser.add_argument('--' + name)
    parser.add_argument('--max-seconds', type=int)
    cli = parser.parse_args(argv)
    if cli.describe:
        print(json.dumps({'policy_contract': POLICY, 'policy_state': 'GO-finite-precompute-to-E1',
            'required_policy_fields': ['preparation_request', 'preparation_plan', 'source_evidence',
                'pipeline_started_utc', 'pipeline_deadline_utc', 'pipeline_max_seconds', 'e1_max_seconds',
                'gpu_guard', 'training_worker'], 'output': (ROOT / 'full-e1-r1/worker').as_posix(),
            'database_access': False, 'approval_generation': False, 'job_launch': False}))
        return
    require(all(getattr(cli, name) is not None for name in ('preparation_request', 'preparation_result',
            'preparation_guard', 'root_decision', 'deadline_utc', 'max_seconds')), 'Explicit prerequisite/policy/deadline/cap inputs required')
    request_path, encoded, plan_path, plan = build(cli.preparation_request, cli.preparation_result,
        cli.preparation_guard, cli.root_decision, cli.deadline_utc, cli.max_seconds)
    if cli.emit:
        with request_path.open('xb') as stream:
            stream.write(encoded)
        with plan_path.open('xb') as stream:
            stream.write(binding.payload(plan))
    print(json.dumps({'state': 'emitted-metadata-only' if cli.emit else 'described-actual-binding-no-files',
        'request': request_path.as_posix(), 'plan': plan_path.as_posix(), 'plan_sha256': binding.canonical(plan),
        'conditional_admission': plan['conditional_admission'], 'database_access': False, 'approval_generation': False, 'job_launch': False}))


if __name__ == '__main__':
    main()
