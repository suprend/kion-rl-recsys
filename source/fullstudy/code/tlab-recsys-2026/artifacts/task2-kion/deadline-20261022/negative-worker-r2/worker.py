"""Reuse the pinned single-pass worker under a fresh root-owned control contract.

This entrypoint intentionally cannot run without root creating control.py and an
admitted --request. It adds no alternate trainer, evaluator, or artifact writer.
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
CONTROL = HERE.parent / 'control-r2/control.py'
OLD_WORKER = Path('D:/tlab-recsys-2026/artifacts/task2-kion/quality-recovery-20261021/uniform-control-r1/worker.py')
OLD_SHA = 'c39ec2b3cf50723591e0b5da2fef46012947e852970f36dc8c513ab30abe3f9f'


def bind_completion(control, request, started=None):
    """Enrich only this fresh adjunct receipt; original raw model files stay intact."""
    started = time.perf_counter() if started is None else started
    deadline = datetime.fromisoformat(request['deadline_utc'].replace('Z', '+00:00'))
    path = control.OUTPUT / 'singlepass-completion.json'
    receipt = control.read(path)

    def check(phase):
        if (datetime.now(timezone.utc) >= deadline or (control.OUTPUT / 'stop-request.json').exists()
                or time.perf_counter() - started >= 23910):
            reason = 'Final singlepass publication budget exhausted: ' + phase
            receipt.update(state='failed-unbound-sidecar', error=reason, selected_export_claim=False)
            control.g.write(path, receipt)
            raise ValueError(reason)

    check('before evidence')
    evidence = control.read(control.OUTPUT / 'adjunct/hook-evidence.json')
    if receipt.get('state') != 'completed' or evidence.get('sampler_binding_verified') is not True:
        raise ValueError('Only successful actual sampler-bound singlepass may be enriched')
    keys = ('sampler_contract', 'sampler_specification', 'sampler_audit', 'sampler_binding_verified')
    receipt.update({key: evidence[key] for key in keys})
    receipt['execution_source_hashes'] = request['source_hashes']
    receipt['runtime_performance'] = evidence.get('runtime_performance', {'cpu_threads': 2, 'dynamic_cache': False})
    check('before receipt publication')
    control.g.write(path, receipt)
    check('after receipt publication')
    control.validate_result({'output': control.OUTPUT.as_posix()})
    check('after receipt validation')
    return receipt


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding='utf-8'))
    if request.get('state') != 'admitted' or not CONTROL.is_file():
        raise ValueError('Fresh root-owned control and admitted request are required')
    mandatory = [HERE / name for name in ('worker.py', 'hooks.py', 'sampler.py')] + [CONTROL]
    mandatory += [OLD_WORKER, HERE.parents[1] / 'quality-recovery-20261021/uniform-control-r1/hooks.py']
    # All direct old/new helpers are pinned by the admitted request and native plan.
    for path in mandatory:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if request.get('source_hashes', {}).get(path.as_posix()) != actual:
            raise ValueError('Missing or changed execution-source pin: ' + path.as_posix())
    payload = OLD_WORKER.read_bytes()
    if hashlib.sha256(payload).hexdigest() != OLD_SHA:
        raise ValueError('Pinned original single-pass worker changed')
    spec = importlib.util.spec_from_file_location('_retrieval128_reused_worker', OLD_WORKER)
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    exec(compile(payload, str(OLD_WORKER), 'exec'), worker.__dict__)
    worker.HERE = CONTROL.parent
    result = worker.main(['--request', args.request.as_posix()])
    bind_completion(sys.modules['_uniform_worker_metadata'], request, started)
    return result


if __name__ == '__main__':
    main()
