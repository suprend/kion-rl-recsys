"""External worker validation; original native supervisor execution unchanged."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 7 or argv[0] not in ('check', 'run') or set(argv[1::2]) != {'--plan', '--approval', '--state-dir'}:
        raise ValueError('Only exact synchronous native check/run')
    plan = json.loads(Path(argv[argv.index('--plan') + 1]).read_text(encoding='utf-8'))
    path = HERE / 'control.py'
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != plan['input_hashes'][path.as_posix()]:
        raise ValueError('Reviewed new metadata changed')
    spec = importlib.util.spec_from_file_location('_d22_native_control', path)
    p = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = p
    exec(compile(payload, str(path), 'exec'), p.__dict__)
    job = plan['jobs'][0]
    request_path, approval_path = [Path(job['uniform_control'][key]) for key in ('request', 'approval')]
    request, approved_plan, job, _, _, _ = p.checked(request_path, approval_path, running=argv[0] == 'run')
    p.require(plan == approved_plan and Path(argv[argv.index('--plan') + 1]).resolve() == Path(request['native_plan']['path']).resolve()
        and Path(argv[argv.index('--approval') + 1]).resolve() == Path(request['native_approval']['path']).resolve()
        and Path(argv[argv.index('--state-dir') + 1]).resolve() == p.paths(request['phase'])[1], 'exact approved native/resource paths')
    base = p.direct(p.COMMON / 'scripts/kion_night_supervisor.py', p.g.NATIVE_SHA, '_d22_unchanged_native')
    original_validate, original_evidence = base.validate_plan, base.completion_evidence
    def validate(actual, approval, **kwargs):
        p.require(actual == plan and approval['plan_sha256'] == p.g.canonical(actual), 'exact approved actual plan')
        p.validate_plan(actual, request)
        virtual = copy.deepcopy(actual)
        virtual['jobs'][0]['argv'][0] = 'scripts/kion_ready_runtime.py'
        virtual_approval = copy.deepcopy(approval)
        virtual_approval['plan_sha256'] = p.g.canonical(virtual)
        original_validate(virtual, virtual_approval, **kwargs)
        return actual
    def evidence(actual_job):
        p.require(actual_job == job, 'exact new operation')
        p.validate_result(actual_job)
        return original_evidence(actual_job)
    base.validate_plan, base.completion_evidence = validate, evidence
    return base.main(argv)

if __name__ == '__main__':
    raise SystemExit(main())
