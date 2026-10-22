"""Reproducible complete-walk baselines, optionally with earlier microkernels."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from qpe_lab.contracts import project_root, write_json
from qpe_lab.evaluation.protocol import ACCEPTED, evaluate
from qpe_lab.tasks.bundle import freeze_judge
from qpe_lab.tasks.task import load_walk


def search(task_path, folder, timeout, candidates=None):
    """A frozen, reproducible candidate-only loop; all deliverables stay Python."""
    task = load_walk(task_path)
    if task.contract["cost_model"] != "clifford_t":
        raise ValueError(
            "Search ranking requires a fixed Clifford+T contract, not partial RY counts"
        )
    pin_path = folder / "judge.json"
    pin = freeze_judge(task_path, pin_path)
    generated = folder / "candidates"
    generated.mkdir()
    jobs = []
    if candidates:
        jobs = [(Path(path).stem, Path(path).resolve()) for path in candidates]
    else:
        options = [("grouped", {}), ("ungrouped", {"grouped": False})]
        options += [
            (f"qrom-b{block}", {"select_method": "qrom", "qrom_block_size": block})
            for block in (1, 2)
        ]
        options += [
            (
                f"{name}-lsb",
                dict(
                    settings,
                    outer_order=list(range(task.label_bits)),
                    inner_order=list(range(task.term_bits)),
                ),
            )
            for name, settings in list(options)
            if name != "ungrouped"
        ]
        if "preparation" in task.model:
            options = [
                (
                    f"alias-b{block}-u{unlookup}",
                    {
                        "implementation": "dynamic",
                        "prepare_method": "alias",
                        "rotation_method": "phase_gradient",
                        "qrom_block_size": block,
                        "unlookup_block_size": unlookup,
                        "angle_block_size": 1,
                    },
                )
                for block, unlookup in ((1, 4), (2, 8), (4, 8), (4, 16), (8, 16))
                if block <= 1 << task.label_bits
                and unlookup <= 1 << (task.label_bits + task.term_bits)
            ]
        elif "paper" in task.model:
            options = [
                (
                    f"paper-b{block}-u{unlookup}",
                    {
                        "implementation": "paper_strict"
                        if "uniform_bits" in task.model["paper"]
                        else "paper",
                        "outer_block_size": min(2, 1 << task.label_bits),
                        "outer_unlookup_size": min(8, 1 << task.label_bits),
                        "qrom_block_size": min(block, 1 << task.term_bits),
                        "unlookup_block_size": min(
                            unlookup, 1 << (task.label_bits + task.term_bits)
                        ),
                    },
                )
                for block, unlookup in ((2, 16), (4, 16), (8, 16), (8, 32), (16, 32))
            ]
        for name, settings in options:
            path = generated / f"{name}.py"
            path.write_text(f"from qpe_lab import build_walk\nbuild_walk(**{settings!r}).emit()\n")
            jobs.append((name, path))
    records, pareto = [], []
    objective = task.contract["objective"]
    for name, candidate in jobs:
        run, report = evaluate(task_path, candidate, timeout=timeout, judge_path=pin_path)
        cost = {key: value for key, value in report["cost"].items() if key != "synthesis"}
        if "synthesis" in report["cost"]:
            cost["walk_error_bound"] = report["cost"]["synthesis"]["walk_operator_error_bound"]
        record = {
            "name": name,
            "run": str(run),
            "candidate": str(candidate),
            "correctness_status": report["correctness_status"],
            "cost": cost,
            "h_ops": report.get("h_ops"),
            "detail": report.get("detail"),
        }
        records.append(record)
        accepted = [
            r
            for r in records
            if r["correctness_status"] in ACCEPTED and r["cost"]["status"] == "complete"
        ]
        pareto = [
            r
            for r in accepted
            if not any(
                all(s["cost"][k] <= r["cost"][k] for k in ("t_count", "qubits"))
                and any(s["cost"][k] < r["cost"][k] for k in ("t_count", "qubits"))
                for s in accepted
            )
        ]
        best = (
            min(accepted, key=lambda r: (r["cost"][objective], r["cost"]["qubits"], r["name"]))
            if accepted
            else None
        )
        if best:
            # Use the evaluated source snapshot, never a subsequently edited host file.
            source = Path(best["run"]) / "candidate" / Path(best["candidate"]).name
            best["python_generator"] = str(source)
        summary = {
            "schema": "qpe.search.v1",
            "task_sha256": task.task_hash,
            "judge_sha256": pin["judge_sha256"],
            "objective": objective,
            "runs": records,
            "complete": len(records) == len(jobs),
            "best": best,
            "pareto": [r["name"] for r in pareto],
            "trust_scope": "candidate generator isolated; judge pin detects changes, not a security boundary against a host-privileged agent",
        }
        write_json(folder / "summary.json", summary)
        print(
            json.dumps(
                {
                    "name": name,
                    "status": record["correctness_status"],
                    "t_count": cost.get("t_count"),
                    "qubits": cost.get("qubits"),
                    "best": best["name"] if best else None,
                }
            ),
            flush=True,
        )
    return 0 if any(r["correctness_status"] in ACCEPTED for r in records) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-microkernels", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--task", default="benchmarks/dfthc/fe2s2/walk/walk.task.json")
    parser.add_argument("--candidates", nargs="*")
    args = parser.parse_args()
    root = project_root()
    folder = root / "runs" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ") + "_dfthc-suite")
    folder.mkdir(parents=True)
    if args.search:
        result = search(Path(args.task).resolve(), folder, args.timeout, args.candidates)
        print(f"Summary: {folder / 'summary.json'}")
        return result
    jobs = [
        (
            f"{system}-{candidate}",
            f"benchmarks/dfthc/{system}/walk/walk.task.json",
            f"candidates/dfthc_walk/{candidate}.py",
        )
        for system in ("toy", "fe2s2")
        for candidate in ("baseline", "grouped")
    ]
    if args.include_microkernels:
        jobs += [
            (
                "lookup-regression",
                "benchmarks/dfthc/fe2s2/lookup.task.json",
                "candidates/fe2s2_angles/unary.py",
            ),
            (
                "rotation-regression",
                "benchmarks/dfthc/fe2s2/rotation.task.json",
                "candidates/fe2s2_rotation/unary.py",
            ),
        ]
    records = []
    for name, task, candidate in jobs:
        run, report = evaluate(root / task, root / candidate, timeout=args.timeout)
        record = {
            "name": name,
            "run": str(run.relative_to(root)),
            "correctness_status": report["correctness_status"],
            "h_ops": report.get("h_ops"),
            "cost": {key: value for key, value in report["cost"].items() if key != "synthesis"},
            "detail": report.get("detail"),
        }
        records.append(record)
        write_json(
            folder / "summary.json",
            {
                "runs": records,
                "complete": len(records) == len(jobs),
                "all_verified": all(r["correctness_status"] in ACCEPTED for r in records),
            },
        )
        print(
            json.dumps(
                {"name": name, "status": report["correctness_status"], "run": str(run)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(f"Summary: {folder / 'summary.json'}")
    return 0 if all(r["correctness_status"] in ACCEPTED for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
