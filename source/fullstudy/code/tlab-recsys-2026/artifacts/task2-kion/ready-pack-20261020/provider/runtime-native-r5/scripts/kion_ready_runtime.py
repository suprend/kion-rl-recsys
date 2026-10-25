"""Frozen ready launch route; root native supervisor owns every launch."""
from pathlib import Path
import runpy
import sys


PAS_SITE = Path("D:/personal-agent-system/.venv/Lib/site-packages")
ML_SITE = Path("D:/personal-agent-system/tools/game-voice/.venv/Lib/site-packages")
BASE_EXE = Path("C:/Users/Михаил/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe")
ENTRIES = {"benchmark": "kion_ready_gpu_preflight.py", "sft": "kion_complete_train.py",
           "rl": "kion_policy_rl.py", "rl-bridge": "kion_complete_rl.py"}


def configure_paths():
    if "torch" in sys.modules:
        raise RuntimeError("Torch must not be imported by the launch bootstrap")
    if Path(sys.executable).resolve() != BASE_EXE.resolve():
        raise RuntimeError("Launch the pinned base interpreter directly; no venv launcher")
    if not PAS_SITE.is_dir() or not ML_SITE.is_dir():
        raise RuntimeError("Pinned dependency directory is missing")
    # Frozen scripts stay first; the existing ML venv precedes base distributions.
    sys.path.insert(1, ML_SITE.as_posix())
    # Append only: stdlib and ML packages retain priority over PAS packages.
    sys.path.append(PAS_SITE.as_posix())


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ENTRIES:
        raise SystemExit("Expected fixed ready entry: benchmark, sft, rl or rl-bridge")
    configure_paths()
    target = Path(__file__).resolve().with_name(ENTRIES[sys.argv[1]])
    if not target.is_file():
        raise RuntimeError("Frozen ready entrypoint is missing")
    sys.argv = [target.as_posix(), *sys.argv[2:]]
    runpy.run_path(target.as_posix(), run_name="__main__")


if __name__ == "__main__":
    main()
