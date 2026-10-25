"""Guarded one-epoch handoff; this controller never edits the running trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

import psutil


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(config):
    hashes = config["contract"]["inputs_sha256"]
    if set(hashes) != {"metadata.json", "train.npz", "val.npz"}:
        raise RuntimeError("Unexpected prepared input hash keys")
    data = Path(config["data"])
    for name, expected in hashes.items():
        if Path(name).name != name or sha256(data / name) != expected:
            raise RuntimeError(f"Prepared input hash changed: {name}")


def same_path(left, right):
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))


def option(command, name):
    if command.count(name) != 1:
        raise ValueError(f"Expected exactly one {name}")
    index = command.index(name)
    if index + 1 >= len(command):
        raise ValueError(f"Missing {name} value")
    return command[index + 1]


def owned_command(command, script, output):
    try:
        return (len(command) > 1 and same_path(command[1], script)
                and same_path(option(command, "--output"), output))
    except (ValueError, TypeError, OSError):
        return False


def identity(process):
    with process.oneshot():
        return {"pid": process.pid, "create_time": process.create_time(),
                "exe": process.exe(), "cmdline": process.cmdline()}


def dedicated_console_helper(saved):
    # Only a descendant enumerated from the owned root may reach this check.
    # Windows venv redirectors create a dedicated console host; an arbitrary
    # executable with the same basename is never part of the owned tree.
    system_root = os.environ.get("SystemRoot")
    if not system_root:
        return False
    expected = Path(system_root) / "System32" / "conhost.exe"
    command = saved["cmdline"]
    if not same_path(saved["exe"], expected) or len(command) != 2:
        return False
    executable = command[0].removeprefix("\\??\\")
    return (same_path(executable, expected) and command[1].startswith("0x")
            and bool(command[1][2:]) and all(char in "0123456789abcdefABCDEF" for char in command[1][2:]))


def matching_process(saved, process_factory=psutil.Process):
    try:
        process = process_factory(saved["pid"])
        if identity(process) != saved:
            raise RuntimeError(f"Process identity changed: {saved['pid']}")
        return process
    except psutil.NoSuchProcess:
        return None


def capture_tree(pid, script, output, process_factory=psutil.Process):
    root = process_factory(pid)
    processes = [root, *root.children(recursive=True)]
    snapshots = [identity(process) for process in processes]
    if (not snapshots or not owned_command(snapshots[0]["cmdline"], script, output)
            or any(not owned_command(item["cmdline"], script, output)
                   and not dedicated_console_helper(item) for item in snapshots[1:])):
        raise RuntimeError("Training tree contains an unrelated command")
    # Recheck identities after enumeration, before arming.
    for item in snapshots:
        if matching_process(item, process_factory) is None:
            raise RuntimeError("Training tree exited during capture")
    return snapshots


def active_writer(script, output):
    for process in psutil.process_iter():
        try:
            # Protected Windows system processes cannot be queried. A matching
            # trainer must be Python; access failures for Python stay fail-closed.
            try:
                name = process.name().lower()
            except psutil.AccessDenied:
                continue
            if not name.startswith("python"):
                continue
            if owned_command(process.cmdline(), script, output):
                return process.pid
        except psutil.NoSuchProcess:
            continue
    return None


def boundary(progress):
    history = progress.get("history", [])
    return (progress.get("epoch", 0) >= 1 and bool(history)
            and history[0].get("epoch") == 1 and bool(history[0].get("validation")))


def verify_checkpoint(path, config):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["state"]
    if state.get("epoch") != 1 or state.get("offset") != 0 or len(state.get("history", [])) != 1 or not boundary(state):
        raise RuntimeError("Checkpoint is not the completed first epoch")
    if saved["config"].get("contract") != config.get("contract") or saved["config"].get("data") != config.get("data"):
        raise RuntimeError("Checkpoint config/input/source hashes differ")
    return state


def snapshot_checkpoint(source, destination, config):
    # A hard link pins the atomically published best.pt inode against later replacement.
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError:
        temporary = destination.with_name(destination.name + ".copying")
        with temporary.open("xb") as target, source.open("rb") as original:
            shutil.copyfileobj(original, target, 8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        # Reserve exclusively, then atomically replace only our own reservation.
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        os.replace(temporary, destination)
    return verify_checkpoint(destination, config)


def stop_tree(snapshots, process_factory=psutil.Process, sleep=time.sleep, timeout=20):
    captured = {item["pid"] for item in snapshots}
    # Validate the entire set and descendant closure before signalling anything.
    live = []
    for item in snapshots:
        process = matching_process(item, process_factory)
        if process is not None:
            if any(child.pid not in captured for child in process.children(recursive=True)):
                raise RuntimeError("Training tree acquired an uncaptured descendant")
            live.append(item)
    for item in reversed(live):
        process = matching_process(item, process_factory)
        if process is not None:
            process.terminate()
    for escalate in (False, True):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = [item for item in snapshots if matching_process(item, process_factory) is not None]
            if not remaining:
                return
            if escalate:
                for item in remaining:
                    process = matching_process(item, process_factory)
                    if process is not None:
                        process.kill()
                escalate = False
            sleep(.2)
    raise RuntimeError("Owned training processes did not stop; small run forbidden")


def small_command(python, script, config, output):
    training = config["contract"]["training"]
    command = [str(python), str(script), "--data", config["data"], "--output", str(output),
               "--batch-size", "64", "--accumulation", "2", "--hidden-size", "256",
               "--layers", "4", "--heads", "4", "--device", "cuda"]
    # Keep the original five-epoch LR schedule and all other training settings.
    for key in ("epochs", "lr", "warmup_ratio", "weight_decay", "seed", "patience", "validation_limit", "bucket_size"):
        command.extend(["--" + key.replace("_", "-"), str(training[key])])
    return command


def run(args, *, sleep=time.sleep, stop=stop_tree, popen=subprocess.Popen):
    source = args.source_output.resolve()
    small = args.small_output.resolve()
    job = args.job.resolve()
    if same_path(source, small):
        raise ValueError("Source and small output must differ")
    job.mkdir(parents=True, exist_ok=False)
    receipt_path = job / "controller.json"
    receipt = {"status": "arming", "started_utc": utc(), "controller_pid": os.getpid(),
               "source_output": source.as_posix(), "small_output": small.as_posix(),
               "expected_stop_reason": "completed large model epoch 1 and validation; continue with 4x256 full training"}
    def record(status, **values):
        receipt.update(status=status, updated_utc=utc(), **values)
        write_json(receipt_path, receipt)
    try:
        config = read_json(source / "config.json")
        source_job = read_json(args.source_job / "process.json")
        script = Path(source_job["command"][1]).resolve()
        if not owned_command(source_job["command"], script, source):
            raise RuntimeError("Source receipt command/output mismatch")
        if script.name != "kion_full_train.py" or config.get("benchmark") or config["contract"]["training"]["epochs"] != 5:
            raise RuntimeError("Unexpected source training contract")
        hashes = config["contract"]["sources_sha256"]
        for name, expected in hashes.items():
            if Path(name).name != name or sha256(script.with_name(name)) != expected:
                raise RuntimeError("Trainer source hash changed")
        verify_inputs(config)
        if not same_path(args.python, source_job["command"][0]):
            raise RuntimeError("Small training must use the original GPU interpreter")
        snapshots = capture_tree(source_job["training_pid"], script, source)
        command = small_command(args.python, script, config, small)
        if small.exists() and any(small.iterdir()):
            raise RuntimeError("Small output must be empty")
        if active_writer(script, small):
            raise RuntimeError("Small output already has an active writer")
        record("waiting", source_config=config, source_config_sha256=sha256(source / "config.json"),
               source_command=source_job["command"], source_processes=snapshots, small_command=command,
               source_supervisor_pid=source_job.get("supervisor_pid"))
        while True:
            progress = read_json(source / "progress.json")
            if boundary(progress):
                break
            if any(matching_process(item) is None for item in snapshots):
                raise RuntimeError("Source exited before validated epoch boundary")
            sleep(args.poll_seconds)
        checkpoint = source / "epoch-1.pt"
        state = snapshot_checkpoint(source / "best.pt", checkpoint, config)
        verify_inputs(config)
        record("stopping", first_epoch_state=state, first_epoch_validation=state["history"][0],
               checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256(checkpoint), boundary_observed_utc=utc())
        stop(snapshots)
        if any(matching_process(item) is not None for item in snapshots):
            raise RuntimeError("Old training process remains alive")
        if active_writer(script, source):
            raise RuntimeError("An uncaptured source output writer remains; small run forbidden")
        verify_checkpoint(checkpoint, config)
        for name, expected in hashes.items():
            if sha256(script.with_name(name)) != expected:
                raise RuntimeError("Trainer source hash changed before small launch")
        verify_inputs(config)
        if active_writer(script, small) or (small.exists() and any(small.iterdir())):
            raise RuntimeError("Small output is no longer fresh")
        small.mkdir(parents=True, exist_ok=True)
        record("starting_small", source_stopped_utc=utc())
        with (job / "small.stdout.log").open("xb") as stdout, (job / "small.stderr.log").open("xb") as stderr:
            child = popen(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                          close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                          cwd=str(script.parent.parent))
            record("small_running", small_pid=child.pid, small_started_utc=utc())
            code = child.wait()
        record("completed" if code == 0 else "failed", small_exit_code=code, ended_utc=utc())
        return code
    except Exception as error:
        record("failed", error=f"{type(error).__name__}: {error}", ended_utc=utc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-output", "source-job", "small-output", "job", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
