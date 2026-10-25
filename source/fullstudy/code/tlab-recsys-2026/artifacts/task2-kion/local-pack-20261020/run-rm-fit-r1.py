"""One authorized bounded CPU CatBoost pair; no GPU, retry or queue expansion."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
BUNDLE = BASE / "bundle-rm-r1"
DATA = Path("D:/tlab-recsys-2026/data/processed/kion-reward-local-20261020-r1")
ROOT = BASE / "rm-r1"
PYTHON = "D:/tlab-recsys-2026/.venv/Scripts/python.exe"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ROOT.exists():
        raise FileExistsError("Existing RM operation must be reconciled before reuse")
    freeze = json.loads((BASE / "freeze-rm-r1.json").read_text(encoding="utf-8"))
    for name, expected in freeze["sources"].items():
        if digest(BUNDLE / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    if digest(DATA / "manifest.json") != "42d7532a5daea8af36198e0a4ba6ba3b134cd66e93dcfd75961913a14624144c":
        raise ValueError("Audited data manifest changed")
    ROOT.mkdir()
    command = [PYTHON, "-u", str(BUNDLE / "scripts/kion_reward_boosting.py"), "fit", "--bundle", str(DATA),
               "--output", str(ROOT / "scorer"), "--backend", "catboost", "--seed", "2026"]
    state = {"state": "prepared", "argv": command, "GPU": False, "CPU_threads": 2, "fixed_configs": 2,
             "maximum_seconds": 3600, "started_utc": datetime.now(timezone.utc).isoformat(),
             "data_manifest_sha256": digest(DATA / "manifest.json"), "freeze_sha256": digest(BASE / "freeze-rm-r1.json")}

    def publish():
        (ROOT / "operation.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    publish()
    begin = time.perf_counter()
    spec = importlib.util.spec_from_file_location("cpu_owned_identity", BUNDLE / "scripts/kion_night_supervisor.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    with (ROOT / "fit.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        state.update(state="running", process_identity=native.process_identity(process.pid))
        publish()
        try:
            code = process.wait(timeout=3600)
            state.update(state="completed" if code == 0 else "failed", exit_code=code)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
            state.update(state="timed_out", exit_code=process.returncode)
        finally:
            state.update(elapsed_seconds=time.perf_counter() - begin, finished_utc=datetime.now(timezone.utc).isoformat())
            publish()
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
