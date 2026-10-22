import argparse
import json
import math
import sys
from pathlib import Path

from qpe_lab.tasks.task import load_walk


def main():
    parser = argparse.ArgumentParser(description="Evaluate Python circuit generators")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Generate, verify, and count one candidate")
    evaluate.add_argument("--task", required=True)
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--timeout", type=float, default=120)
    evaluate.add_argument("--no-kernel-crosscheck", action="store_true")
    evaluate.add_argument("--judge", help="Frozen judge pin created before optimization")
    freeze = commands.add_parser("freeze", help="Pin task, judge, and installed numerical runtime")
    freeze.add_argument("--task", required=True)
    freeze.add_argument("--output", required=True)
    training_freeze = commands.add_parser(
        "freeze-training", help="Maintainer: freeze concrete reference and reward"
    )
    training_freeze.add_argument("--task", required=True)
    training_freeze.add_argument("--circuit", required=True)
    training_freeze.add_argument("--output", required=True)
    training_freeze.add_argument("--scope", choices=("core", "walk"), default="core")
    training_freeze.add_argument("--enable-lean", action="store_true")
    training_freeze.add_argument(
        "--acceptance", choices=("exact_or_sampled", "lean_required"), default="exact_or_sampled"
    )
    delivery = commands.add_parser(
        "agent-task", help="Deliver a frozen task and system prompt to an agent launcher"
    )
    delivery.add_argument("--bundle", required=True)
    delivery.add_argument("--bundle-sha256", required=True)
    delivery.add_argument("--workspace", required=True)
    delivery.add_argument("--budget", required=True)
    delivery.add_argument("--output", help="Write task JSON here; default: stdout")
    session = commands.add_parser(
        "agent-session", help="Start or resume an isolated optimization toolbox"
    )
    session.add_argument(
        "--session", required=True, help="Dedicated session directory inside runs/"
    )
    session.add_argument("--bundle")
    session.add_argument("--bundle-sha256")
    session.add_argument("--max-submissions", type=int)
    session.add_argument("--seconds", type=float)
    session.add_argument("--initial-candidate")
    session.add_argument(
        "--progress-events",
        action="store_true",
        help="Interleave public progress events on stdout; otherwise use stderr",
    )
    for name in (
        "evaluate-training",
        "training-worker",
        "scientific-review",
        "complete-verification",
    ):
        training = commands.add_parser(name, help="Trainer-owned fixed-bundle evaluation")
        training.add_argument("--bundle", required=True)
        training.add_argument("--bundle-sha256", required=True)
        if name == "evaluate-training":
            training.add_argument("--candidate", required=True)
        if name in {"scientific-review", "complete-verification"}:
            training.add_argument("--run", required=True)
        if name in {"evaluate-training", "training-worker", "complete-verification"}:
            training.add_argument("--progress-events", action="store_true")
    training_verify = commands.add_parser("_verify-training", help=argparse.SUPPRESS)
    for name in ("bundle", "bundle_sha256", "stream", "output", "challenge"):
        training_verify.add_argument(name)
    training_verify.add_argument("--seconds", type=float, default=600)
    training_verify.add_argument("--progress")
    inspect = commands.add_parser("inspect", help="Read a fixed circuit task contract")
    inspect.add_argument("task")
    export = commands.add_parser(
        "export-gates", help="Verify and expand a walk artifact to elementary gates"
    )
    export.add_argument("--task", required=True)
    export.add_argument("--circuit", required=True)
    export.add_argument("--output", required=True)
    decode = commands.add_parser("decode-phase", help="Decode a walk eigenphase in radians")
    decode.add_argument("--task", required=True)
    decode.add_argument("--phase", required=True, type=float)
    worker = commands.add_parser("_verify", help=argparse.SUPPRESS)
    worker.add_argument("task")
    worker.add_argument("stream")
    worker.add_argument("output")
    worker.add_argument("--kernel", action="store_true")
    args = parser.parse_args()
    if args.command == "agent-session":
        from qpe_lab.evaluation.protocol import progress_printer
        from qpe_lab.evaluation.session import session_worker

        return session_worker(
            args.session,
            bundle=args.bundle,
            bundle_sha256=args.bundle_sha256,
            max_submissions=args.max_submissions,
            seconds=args.seconds,
            initial_candidate=args.initial_candidate,
            progress=progress_printer(sys.stdout if args.progress_events else sys.stderr),
        )
    if args.command == "agent-task":
        from qpe_lab.tasks.prompt import agent_task

        result = agent_task(
            args.bundle, args.bundle_sha256, workspace=args.workspace, budget=args.budget
        )
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(json.dumps({"output": str(path.resolve()), "scope": result["scope"]}))
        else:
            print(text, end="")
        return 0
    if args.command == "freeze-training":
        from qpe_lab.evaluation.protocol import freeze_training

        print(
            json.dumps(
                freeze_training(
                    args.task,
                    args.circuit,
                    args.output,
                    scope=args.scope,
                    acceptance=args.acceptance,
                    enable_lean=args.enable_lean,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "_verify-training":
        from qpe_lab.evaluation.protocol import verify_training_worker

        return verify_training_worker(
            args.bundle,
            args.bundle_sha256,
            args.stream,
            args.output,
            args.challenge,
            seconds=args.seconds,
            progress_path=args.progress,
        )
    if args.command == "complete-verification":
        from qpe_lab.evaluation.protocol import ACCEPTED, complete_verification, progress_printer

        result = complete_verification(
            args.bundle,
            args.bundle_sha256,
            args.run,
            progress=progress_printer(sys.stdout if args.progress_events else sys.stderr),
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["correctness_status"] in ACCEPTED else 1
    if args.command == "scientific-review":
        from qpe_lab.evaluation.protocol import scientific_review

        result = scientific_review(args.bundle, args.bundle_sha256, args.run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "passed" else 1
    if args.command == "training-worker":
        from qpe_lab.evaluation.protocol import progress_printer, training_worker

        return training_worker(
            args.bundle,
            args.bundle_sha256,
            progress=progress_printer(sys.stdout if args.progress_events else sys.stderr),
        )
    if args.command == "evaluate-training":
        from qpe_lab.evaluation.protocol import (
            ACCEPTED,
            evaluate_training,
            progress_printer,
            submission_response,
        )

        folder, report = evaluate_training(
            args.bundle,
            args.bundle_sha256,
            args.candidate,
            progress=progress_printer(sys.stdout if args.progress_events else sys.stderr),
        )
        print(
            json.dumps(
                submission_response(folder, report),
                indent=None if args.progress_events else 2,
            )
        )
        return 0 if report["correctness_status"] in ACCEPTED else 1
    if args.command == "freeze":
        from qpe_lab.tasks.bundle import freeze_judge

        pin = freeze_judge(args.task, args.output)
        print(
            json.dumps(
                {"output": str(Path(args.output).resolve()), "judge_sha256": pin["judge_sha256"]}
            )
        )
        return 0
    if args.command in ("export-gates", "decode-phase"):
        from qpe_lab.contracts import WalkTask

        task = load_walk(args.task)
        if args.command == "decode-phase":
            if not math.isfinite(args.phase):
                parser.error("phase must be finite")
            from qpe_lab.evaluation.walk import walk_normalization

            normalization = walk_normalization(task)
            print(
                json.dumps(
                    {
                        "phase_radians": args.phase,
                        "energy": normalization["energy_offset"]
                        + normalization["Lambda"] * (1 + math.cos(args.phase)),
                        "normalization": normalization,
                        "target": "quantized SOS model",
                    },
                    indent=2,
                )
            )
        else:
            from qpe_lab.circuits.gadgets.synthesis import expand_clifford_t, expand_walk
            from qpe_lab.circuits.ops import read_walk
            from qpe_lab.evaluation.walk import verify_walk

            report = verify_walk(task, args.circuit)
            doc = read_walk(args.circuit)
            dynamic = doc.get("implementation", {}).get("kind") in (
                "dynamic",
                "paper",
                "paper_strict",
            )
            header = {
                "schema": "qpe.elementary.v2" if dynamic else "qpe.elementary.v1",
                "qubits": report["cost"]["qubits"],
                "logical_qubits": report["cost"]["logical_qubits"],
                "wire_order": (
                    "label,term,system(spin-blocked),spin_inner,spin_outer,enable,workspace; little endian"
                    if "paper" in task.model
                    else "label,term,system(spin-blocked),enable,workspace; little endian"
                ),
            }
            if dynamic:
                header.update(
                    registers=report["cost"]["registers"],
                    classical_semantics={
                        "hmr": "value is fresh outcome id; X-basis measurement, reset wire to zero",
                        "oxor": "value=[new id,left id,right id]; left=-1 means literal zero",
                        "condition": "value={condition:id}; apply gate iff outcome is one",
                        "assert_zero": "ideal-contract assertion, not a reset; compiled workspace error is bounded by walk error budget",
                    },
                    walk_error_budget=task.contract["walk_error_budget"],
                )
            with Path(args.output).open("w") as output:
                output.write(json.dumps(header) + "\n")
                output.writelines(
                    json.dumps(
                        {"gate": name, "wires": wires, "value" if dynamic else "angle": angle}
                    )
                    + "\n"
                    for name, wires, angle in (
                        expand_clifford_t(task, doc, report["cost"]["synthesis"])
                        if report["cost"]["status"] == "complete"
                        else expand_walk(task, doc)[0]
                    )
                )
            print(
                json.dumps(
                    {
                        "output": str(Path(args.output).resolve()),
                        "cost": {
                            k: v
                            for k, v in report["cost"].items()
                            if k not in ("synthesis", "registers")
                        },
                    },
                    indent=2,
                )
            )
        return 0
    if args.command == "inspect":
        from qpe_lab.contracts import RotationTask, WalkTask
        from qpe_lab.tasks.task import read_task

        task = read_task(args.task)
        if isinstance(task, WalkTask):
            print(
                json.dumps(
                    {
                        "name": task.name,
                        "kind": "walk",
                        "generators": len(task.model["generators"]),
                        "system_modes": task.mode_count,
                        "label_bits": task.label_bits,
                        "term_bits": task.term_bits,
                        "controlled": task.controlled,
                        "max_qubits": task.max_qubits,
                        "model_sha256": task.table_hash,
                        "optimization": task.contract,
                    },
                    indent=2,
                )
            )
            return 0
        print(
            json.dumps(
                {
                    "name": task.name,
                    "address_bits": task.address_bits,
                    "word_bits": task.word_bits,
                    "addresses": len(task.words),
                    "max_qubits": task.max_qubits,
                    "table_sha256": task.table_hash,
                    **(
                        {
                            "kind": "rotation",
                            "angle_bits": task.angle_bits,
                            "system_modes": task.mode_count,
                            "controlled": task.controlled,
                        }
                        if isinstance(task, RotationTask)
                        else {"kind": "lookup"}
                    ),
                },
                indent=2,
            )
        )
        return 0
    from qpe_lab.evaluation.protocol import ACCEPTED, verify_worker
    from qpe_lab.evaluation.protocol import evaluate as run

    if args.command == "_verify":
        return verify_worker(args.task, args.stream, args.output, args.kernel)
    folder, report = run(
        args.task,
        args.candidate,
        timeout=args.timeout,
        kernel=not args.no_kernel_crosscheck,
        judge_path=args.judge,
    )
    print(
        json.dumps(
            {
                "run": str(folder),
                "correctness_status": report["correctness_status"],
                "cost": {key: value for key, value in report["cost"].items() if key != "synthesis"},
                "detail": report.get("detail"),
            },
            indent=2,
        )
    )
    return 0 if report["correctness_status"] in ACCEPTED else 1
