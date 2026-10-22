"""Run the unchanged upstream graders and compare against their certified results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "uncompute"
CASES = {
    "smoke": [
        ("coulomb-potential", "haner-piecewise"),
        ("coulomb-potential", "discovered"),
    ],
    "fe2s2": [
        ("chemistry-fe2s2", "babbush-composed"),
        ("chemistry-fe2s2", "discovered"),
    ],
}


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM), *args], text=True
    ).strip()


def provenance(args):
    pin = json.loads((ROOT / "config" / "upstream.json").read_text())
    head = git("rev-parse", "HEAD")
    lock_hash = hashlib.sha256((UPSTREAM / "uv.lock").read_bytes()).hexdigest()
    if head != pin["commit"] or lock_hash != pin["uv_lock_sha256"]:
        raise RuntimeError(
            "Upstream revision or dependency lock does not match config/upstream.json"
        )
    status = git("status", "--porcelain")
    if status:
        raise RuntimeError(
            f"Baseline reproduction requires unchanged upstream files:\n{status}"
        )
    from uncompute.evalcore.kernel import KERNEL_ABI, kernel_available

    if args.executor == "kernel" and not kernel_available():
        raise RuntimeError("C kernel unavailable; run bash scripts/setup_lab_wsl.sh first")
    return {
        "started_utc": datetime.now(UTC).isoformat(),
        "upstream": pin,
        "upstream_status": status,
        "python": sys.version,
        "platform": platform.platform(),
        "kernel_abi": KERNEL_ABI,
        "settings": vars(args),
        "packages": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
            )
        ),
        "verification_scope": "Original task's complete structured/random sample; not an all-input proof.",
    }


def run_case(task, circuit, args, run_dir):
    case_dir = run_dir / f"{task}__{circuit}"
    case_dir.mkdir()
    source = f"circuits/{circuit}/circuit.py"
    rows = [
        json.loads(line)
        for line in (UPSTREAM / "tasks" / task / "results.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    expected = next(row for row in rows if row["source"] == source)
    cmd = [
        sys.executable,
        "-u",
        "tools/eval.py",
        task,
        "--jobs",
        str(args.jobs),
        "--out",
        str(case_dir),
    ]
    if args.source == "generated":
        cmd += [
            "--circuit",
            str(UPSTREAM / "tasks" / task / source),
            "--capture",
            str(case_dir / "circuit.ops"),
        ]
    else:
        cmd += [
            "--ops",
            str(UPSTREAM / "tasks" / task / "circuits" / circuit / "circuit.ops.xz"),
        ]
    write_json(
        case_dir / "invocation.json",
        {"argv": cmd, "cwd": str(UPSTREAM), "expected": expected},
    )
    env = dict(
        os.environ,
        UNCOMPUTE_KERNEL="1" if args.executor == "kernel" else "0",
        PYTHONUNBUFFERED="1",
    )
    start = time.monotonic()
    print(
        f"\nRunning {task}/{circuit} ({args.executor}, {args.source}, full upstream sample)",
        flush=True,
    )
    with (case_dir / "console.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=UPSTREAM,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    report_path = case_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    metrics = report.get("metrics", {})
    actual = {
        "verdict": report.get("verdict"),
        "h_ops": report.get("h_ops"),
        **{key: metrics.get(key) for key in ("toffoli", "qubits", "tq_cost")},
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in actual
        if actual[key] != expected[key]
    }
    result = {
        "task": task,
        "circuit": circuit,
        "returncode": returncode,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "passed": returncode == 0 and not mismatches,
        "actual": actual,
        "expected": expected,
        "mismatches": mismatches,
        "metrics": metrics,
        "report": str(report_path.relative_to(ROOT)),
    }
    write_json(case_dir / "comparison.json", result)
    print(
        f"Comparison: {'PASS' if result['passed'] else 'FAIL'}; {result['elapsed_seconds']:.1f} seconds",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["smoke", "fe2s2", "all"], default="smoke")
    parser.add_argument("--executor", choices=["kernel", "reference"], default="kernel")
    parser.add_argument(
        "--source", choices=["generated", "stored"], default="generated"
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    metadata = provenance(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = ROOT / "runs" / f"{stamp}_{args.suite}_{args.executor}"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "environment.json", metadata)
    print(f"Run directory: {run_dir}", flush=True)
    cases = (
        CASES["smoke"] + CASES["fe2s2"] if args.suite == "all" else CASES[args.suite]
    )
    results = []
    for task, circuit in cases:
        results.append(run_case(task, circuit, args, run_dir))
        write_json(
            run_dir / "summary.json",
            {
                "complete": len(results) == len(cases),
                "passed": all(r["passed"] for r in results),
                "results": results,
            },
        )
        if not results[-1]["passed"]:
            return 1
    print(f"\nAll {len(results)} comparisons passed. Reports: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
