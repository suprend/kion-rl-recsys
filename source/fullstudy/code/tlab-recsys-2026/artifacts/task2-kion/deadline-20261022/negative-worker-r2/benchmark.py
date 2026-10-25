"""Pinned existing36-update native benchmark; root probe installs the sampler.

--worker-request, --output and --stop-file are forwarded unchanged. The root-owned
probe.py and native/resource approvals gate every runtime import and provider load.
The original worker bounds itself at810s inside the900s owned native operation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / 'control-r2'
OLD = Path('D:/tlab-recsys-2026/artifacts/task2-kion/quality-recovery-20261021/pair-probe-r1/worker.py')
OLD_SHA = 'b0a4a06e99ad4b1f4367540041e6be1bd7ce985ad242d95b9fc99cf325cfa3a3'


def bind_completion(probe, request, output, started=None):
    started = time.perf_counter() if started is None else started
    deadline = datetime.fromisoformat(request['deadline_utc'].replace('Z', '+00:00'))
    path = output / 'completion.json'
    receipt = probe.read(path)
    if receipt.get('benchmark_match') is not True or receipt.get('accepted_updates') != 36:
        raise ValueError('Only a complete passed36-update benchmark may bind actual evidence')

    def check(phase):
        if (datetime.now(timezone.utc) >= deadline or (output.parent / 'stop-request.json').exists()
                or time.perf_counter() - started >= 810):
            reason = 'Final benchmark publication budget exhausted: ' + phase
            receipt.update(state='failed-final-publication-boundary', benchmark_match=False, no_match_reason=reason)
            probe.g.write(path, receipt)
            raise ValueError(reason)

    check('before evidence')
    evidence = probe.benchmark_evidence()
    required = {'sampler_contract', 'sampler_specification', 'sampler_audit', 'runtime_performance'}
    if set(evidence) != required or evidence['sampler_contract'] != 'retrieval128-uniform3-forced-training-positive-v1':
        raise ValueError('Root probe must bind actual sampler and runtime cache evidence')
    receipt.update(evidence)
    receipt['execution_source_hashes'] = request['source_hashes']
    check('before receipt publication')
    probe.g.write(path, receipt)
    check('after receipt publication')
    return receipt


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.worker_request.read_text(encoding='utf-8'))
    if request.get('state') != 'admitted' or not (CONTROL / 'probe.py').is_file():
        raise ValueError('Fresh admitted root benchmark request and probe are required')
    paths = [HERE / 'benchmark.py', HERE / 'sampler.py', CONTROL / 'probe.py', OLD]
    for path in paths:
        if request.get('source_hashes', {}).get(path.as_posix()) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Missing or changed exact benchmark execution pin: ' + path.as_posix())
    payload = OLD.read_bytes()
    if hashlib.sha256(payload).hexdigest() != OLD_SHA:
        raise ValueError('Original benchmark worker source changed')
    spec = importlib.util.spec_from_file_location('_retrieval128_reused_benchmark', OLD)
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    exec(compile(payload, str(OLD), 'exec'), worker.__dict__)
    worker.HERE = CONTROL
    result = worker.main(['--worker-request', args.worker_request.as_posix(), '--output', args.output.as_posix(),
                          '--stop-file', args.stop_file.as_posix()])
    bind_completion(sys.modules['_pair_probe_worker_metadata'], request, args.output, started)
    return result


if __name__ == '__main__':
    main()
