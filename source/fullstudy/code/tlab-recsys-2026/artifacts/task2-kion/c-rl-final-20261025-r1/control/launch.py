r"""Root entry point: .\pas.ps1 python -B <campaign>/control/launch.py request-*.json."""
from pathlib import Path
import os
import subprocess
import sys
from common import BASE, BASE_SHA, CONTROL, PAS, ROOT, read, require, sha, validate

def native_env():
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([(PAS/'tools/game-voice/.venv/Lib/site-packages').as_posix(), (PAS/'.venv/Lib/site-packages').as_posix()]), PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='GPU-41efe8ae-15a1-c34d-27f2-7d4d504ea76b')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '4'
    return env

def main():
    require(os.name == 'nt', 'Native Windows required')
    require(len(sys.argv) == 2, 'Exactly one admitted request path required')
    request_path = (ROOT / sys.argv[1]).resolve(strict=True)
    digest = sha(request_path)
    request = validate(read(request_path), request_path, digest, verify_pins=False)
    # The reviewed launcher/common pair is the bootstrap. Verify the complete
    # declared source closure before executing the supervisor or native helpers.
    # Inputs are left to the bounded worker so large hashes count against its cap.
    for path, expected in request['source_hashes'].items():
        require(sha(path) == expected, 'Pinned source changed: ' + path)
    require(sha(BASE) == BASE_SHA, 'Pinned BASE interpreter changed')
    command = [BASE.as_posix(), '-B', (CONTROL/'supervise.py').as_posix(), '--request', request_path.as_posix(), '--request-sha256', digest]
    # The supervisor owns all process creation and the lock; there is no unbounded preflight worker.
    return subprocess.run(command, cwd=ROOT, env=native_env(), creationflags=subprocess.CREATE_NO_WINDOW).returncode

if __name__ == '__main__':
    raise SystemExit(main())
