"""Explicit portable inference; this module defines no training operation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import platform
import sys

from .io import close_arrays, load_arrays, read_json, require, sha256, verify_array, write_json
from .verify import ARMS, verify, verify_fixture, verify_rm_fixture, verify_runtime

def _fresh_output(output, bundle, inputs):
    output = Path(output).resolve()
    bundle, inputs = Path(bundle).resolve(), Path(inputs).resolve()
    require(not output.exists() and output.parent.is_dir(), "Fresh output with existing parent required")
    require(not inputs.is_relative_to(output) and not output.is_relative_to(inputs), "Output overlaps fixture inputs")
    require(not bundle.is_relative_to(output), "Output cannot contain the selected bundle")
    require(not output.is_relative_to(bundle / "reproduction" / "kion_replay"), "Output cannot mutate runtime sources")
    output.mkdir()
    return output


def _versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "torch", "transformers", "catboost", "scipy"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return result


def _comparison(np, actual, expected, raw, mode):
    from .metrics import rank
    require(actual.shape == expected.shape and bool(np.isfinite(expected).all()), "Invalid validation-only oracle")
    identical = bool(np.array_equal(actual, expected))
    same_rank = all(rank(a, r) == rank(e, r) for a, e, r in zip(actual, expected, raw))
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"passed": identical if mode == "exact" else None,
            "scores_identical": identical, "top10_identical": same_rank,
            "maximum_absolute_error": float(np.max(difference)), "mean_absolute_error": float(np.mean(difference)),
            "gate": "exact-values" if mode == "exact" else "descriptive-only-no-bound"}


def run_inference(bundle, inputs, output, tier="frozen-encoder", device="cuda", precision="cuda-fp16",
                  oracle="exact", cpu_threads=4, check=None):
    """Root wrappers may call this directly inside their owned bounded process."""
    check = check or (lambda **fields: None)
    require(tier in ("cached-features", "frozen-encoder"), "Unsupported replay tier")
    require(device in ("cpu", "cuda") and precision in ("fp32", "cuda-fp16"), "Unsupported device/precision")
    require(oracle in ("none", "exact"), "Unknown oracle mode")
    require(type(cpu_threads) is int and 1 <= cpu_threads <= 4, "CPU thread cap is 1..4")
    require(precision != "cuda-fp16" or device == "cuda", "cuda-fp16 requires device=cuda")
    check(phase="portable-verify")
    runtime = verify_runtime(bundle)
    manifest, fixture = verify_fixture(inputs)
    if tier == "frozen-encoder":
        require("gpt" in manifest, "Frozen encoder tier requires full producer peer inputs")
    if oracle != "none":
        require(all(arm in manifest.get("expected", {}) for arm in ARMS), "Validation-only per-arm oracles required")
    output = _fresh_output(output, bundle, inputs)
    check(phase="portable-before-numerical-imports")
    import numpy as np
    import torch
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    if device == "cuda":
        require(torch.cuda.is_available(), "Requested CUDA device is unavailable; no silent CPU fallback")
    names = ["encoded", "candidate_ids", "candidate_raw_ids", "base_logits", "raw_features", "source_values", "item_table"]
    if tier == "frozen-encoder":
        names += ["gpt_x", "gpt_lengths", "gpt_user_features"]
    arrays = load_arrays(np, inputs, manifest, names)
    outputs, encoder_diagnostics, parity = {}, None, {}
    try:
        q = base = None
        if tier == "frozen-encoder":
            from .gpt import encode
            check(phase="portable-frozen-encoder")
            encoded = encode(torch, np, inputs, manifest, arrays, device, precision)
            q, base = encoded["encoded"], encoded["base_logits"]
            encoder_diagnostics = encoded["diagnostics"]
            for name, values, cached in (("encoded", q, arrays["encoded"]), ("base_logits", base, arrays["base_logits"])):
                np.save(output / (name + ".npy"), values, allow_pickle=False)
                outputs[name] = output / (name + ".npy")
                exact = bool(np.array_equal(values, cached))
                difference = np.abs(values.astype(np.float64) - cached.astype(np.float64))
                parity[name] = {"passed": exact if oracle == "exact" else None,
                                "values_identical": exact, "maximum_absolute_error": float(np.max(difference)),
                                "mean_absolute_error": float(np.mean(difference)),
                                "gate": "exact-values" if oracle == "exact" else "descriptive-only-no-bound"}
        from .head import score
        scores = score(torch, np, inputs, manifest, arrays, device, precision, q, base, check)
        # Oracle payloads are first decoded here, after all genuine model outputs.
        for arm in ARMS:
            if arm in manifest.get("expected", {}):
                path = verify_array(inputs, manifest["expected"][arm])
                expected = np.load(path, allow_pickle=False)
                parity[arm] = _comparison(np, scores[arm], expected, arrays["candidate_raw_ids"], oracle)
        from .metrics import summarize
        summaries = {}
        for arm in ARMS:
            path = output / ("scores-" + arm + ".npy")
            np.save(path, scores[arm], allow_pickle=False)
            outputs[arm] = path
            summaries[arm] = summarize(scores[arm], arrays["candidate_raw_ids"], arrays["candidate_ids"], scores["C"])
        summary = {"contract": "kion-portable-inference-v1", "state": "completed",
                   "tier": tier, "rows": manifest["rows"], "candidate_width": 1024,
                   "device": device, "precision": precision, "oracle_mode": oracle,
                   "oracle_passed": all(item["passed"] for item in parity.values()) if oracle == "exact" and parity else None,
                   "parity": parity, "arms": summaries, "encoder": encoder_diagnostics,
                   "raw_features_recomputed": False, "candidate_pool_rebuilt": False,
                   "RM_recomputed": False, "training_updates": 0, "optimizer_created": False,
                   "scope": "bounded transformed model-input fixture with source user IDs omitted; not a full-cohort or raw-data rerun"}
        write_json(output / "summary.json", summary)
        outputs["summary"] = output / "summary.json"
        # Recheck files and source pins after numerical work, without trusting the oracle.
        check(phase="portable-final-integrity")
        runtime_after = verify_runtime(bundle)
        _, fixture_after = verify_fixture(inputs)
        require(runtime_after == runtime and fixture_after == fixture, "Runtime/fixture changed during inference")
        descriptors = {name: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in outputs.items()}
        receipt = {"contract": "kion-portable-run-receipt-v1", "state": "completed",
                   "finished_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
                   "fixture": fixture, "outputs": descriptors, "versions": _versions(),
                   "training_updates": 0, "optimizer_created": False, "oracle_mode": oracle,
                   "oracle_passed": summary["oracle_passed"], "cpu_threads": cpu_threads,
                   "tier": tier, "device": device, "precision": precision}
        if device == "cuda":
            receipt["cuda"] = {"torch_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
        write_json(output / "run-receipt.json", receipt)
        return summary
    finally:
        close_arrays(arrays)


def run_reward(bundle, inputs, output, check=None):
    check = check or (lambda **fields: None)
    check(phase="portable-reward-verify")
    runtime = verify_runtime(bundle)
    manifest, fixture = verify_rm_fixture(inputs)
    output = _fresh_output(output, bundle, inputs)
    check(phase="portable-reward-before-imports")
    from .rm import run
    summary = run(inputs, manifest, output)
    write_json(output / "rm-summary.json", summary)
    require(verify_runtime(bundle) == runtime and verify_rm_fixture(inputs)[1] == fixture, "Runtime/fixture changed during RM inference")
    files = {path.name: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in output.iterdir() if path.is_file()}
    write_json(output / "run-receipt.json", {"contract": "kion-portable-run-receipt-v1", "state": "completed",
               "runtime": runtime, "fixture": fixture, "versions": _versions(), "outputs": files,
               "training_updates": 0, "optimizer_created": False, "tier": "frozen-RM"})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Portable frozen KION inference; no training command")
    sub = parser.add_subparsers(dest="command", required=True)
    integrity = sub.add_parser("verify", help="stdlib hashes, headers, source syntax; no numerical imports")
    integrity.add_argument("--bundle", required=True)
    integrity.add_argument("--inputs")
    infer = sub.add_parser("infer", help="execute real frozen head/encoder inference")
    for command in (infer, sub.add_parser("reward", help="separate-process frozen CatBoost inference")):
        command.add_argument("--bundle", required=True)
        command.add_argument("--inputs", required=True)
        command.add_argument("--output", required=True)
    infer.add_argument("--tier", choices=("cached-features", "frozen-encoder"), required=True)
    infer.add_argument("--device", choices=("cpu", "cuda"), required=True)
    infer.add_argument("--precision", choices=("fp32", "cuda-fp16"), required=True)
    infer.add_argument("--oracle", choices=("none", "exact"), default="exact")
    infer.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify(args.bundle, args.inputs)
        elif args.command == "reward":
            result = run_reward(args.bundle, args.inputs, args.output)
        else:
            result = run_inference(args.bundle, args.inputs, args.output, args.tier, args.device,
                                   args.precision, args.oracle, args.cpu_threads)
        import json
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 1 if result.get("oracle_passed") is False else 0
    except (ValueError, OSError, KeyError) as error:
        print("portable replay failed: " + str(error), file=sys.stderr)
        return 1
