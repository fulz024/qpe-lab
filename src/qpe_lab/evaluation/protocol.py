"""Orchestrate reference acceptance, candidate execution and evaluated reports."""

import hashlib
import json
import math
import shutil
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from qpe_lab.contracts import (
    RotationTask,
    WalkTask,
    digest,
    project_root,
    write_json,
)
from qpe_lab.evaluation.cost import evaluation_feedback, training_reward
from qpe_lab.evaluation.sandbox import generate, minimal_env, run_job
from qpe_lab.evaluation.verification.engine import verify_entry
from qpe_lab.evaluation.verification.sampled import reserve_sampling_attempt
from qpe_lab.tasks.bundle import (
    check_judge,
    judge_fingerprint,
    load_training,
    prepare_training,
    provenance,
    seal_training,
)
from qpe_lab.tasks.task import read_task

ACCEPTED = {"verified", "exact_verified", "bounded_verified"}


class EvaluationProgress:
    """Public scheduling telemetry. Never forwards challenges or candidate logs."""

    def __init__(self, callback=None):
        self.callback, self.run = callback, None
        self.started = time.monotonic()
        self.deadline = None
        self.current = {"stage": "preparing"}
        self.last = {}
        self.heartbeat()

    def stage(self, name):
        self.current = {"stage": name}
        self.heartbeat()

    def heartbeat(self):
        event = {"event": "progress", **self.current}
        if self.run is not None:
            event["run"] = str(self.run)
            snapshot = self.run / "verifier-progress.json"
            if self.current["stage"] == "verification" and snapshot.exists():
                data = json.loads(snapshot.read_text())
                # Explicit allowlist; future verifier diagnostics remain private.
                event.update(
                    {
                        k: data[k]
                        for k in ("entry", "phase", "backend", "phase_budget_seconds")
                        if k in data
                    }
                )
                if "phase_started" in data:
                    event["phase_elapsed_seconds"] = max(
                        0, time.monotonic() - data["phase_started"]
                    )
        event["elapsed_seconds"] = time.monotonic() - self.started
        event["remaining_seconds"] = (
            max(0, self.deadline - time.monotonic()) if self.deadline is not None else None
        )
        self.last = event
        if self.run is not None:
            with (self.run / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
        if self.callback is not None:
            self.callback(event)


def progress_printer(stream):
    """The stdout protocol opts in; CLI stderr can always show progress."""
    return lambda event: print(json.dumps(event, ensure_ascii=False), file=stream, flush=True)


def run_training_verifier(folder, pin, manifest, frozen, run, remaining, activity):
    """One fresh, ledger-accounted challenge against an immutable artifact."""
    root = project_root()
    frozen_hash = digest(frozen)
    draw = reserve_sampling_attempt(root / "runs/training-ledger.sqlite3", pin, frozen_hash)
    challenge = run / "sampling-private.json"
    write_json(challenge, draw)
    challenge.chmod(0o600)
    activity.stage("verification")
    worker = run_job(
        [
            sys.executable,
            "-m",
            "qpe_lab",
            "_verify-training",
            str(folder),
            pin,
            str(frozen),
            str(run / "verification.json"),
            str(challenge),
            "--seconds",
            str(remaining),
            "--progress",
            str(run / "verifier-progress.json"),
        ],
        root,
        run / "verifier.log",
        remaining,
        minimal_env() | {"PYTHONPATH": str(root / "src")},
        heartbeat=activity.heartbeat,
    )
    if worker["timed_out"]:
        verified = {
            "correctness_status": "timeout",
            "reward": 0.0,
            "detail": "Verifier exceeded frozen deadline",
            "failure_stage": "verification",
        }
    else:
        verified = json.loads((run / "verification.json").read_text())
        if verified["h_ops"] != frozen_hash or digest(frozen) != frozen_hash:
            raise ValueError("Frozen submitted artifact changed")
        if verified["correctness_status"] in ACCEPTED and (
            worker["returncode"]
            or digest(run / "verified-gates.json") != verified["verified_gates_sha256"]
        ):
            raise ValueError("Verification process or final gate artifact mismatch")
    load_training(folder, pin)
    if verified["correctness_status"] in ACCEPTED:
        verified["reward"] = training_reward(
            verified["correctness_status"], verified["cost"], manifest["reward"]
        )
    return worker, verified


def evaluate(
    task_path, candidate_path, *, timeout=120, kernel=True, judge_path=None, progress=None
):
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    activity = EvaluationProgress(progress)
    root = project_root()
    task = read_task(task_path)
    candidate = Path(candidate_path).resolve()
    if not candidate.is_file() or candidate.suffix != ".py":
        raise ValueError("Candidate must be a Python generator file")
    prov = provenance(root)
    pin = json.loads(Path(judge_path).read_text()) if judge_path else None
    pin_hash = digest(judge_path) if judge_path else None
    if pin is not None:
        check_judge(pin, task, prov)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    kind = (
        "walk"
        if isinstance(task, WalkTask)
        else "rotation"
        if isinstance(task, RotationTask)
        else "lookup"
    )
    run = root / "runs" / f"{stamp}_{kind}_{candidate.stem}"
    out = run / "generator"
    out.mkdir(parents=True)
    inputs = run / "input"
    inputs.mkdir()
    # Freeze trusted task files separately from the generator's only writable mount.
    spec = json.loads(task.path.read_text())
    table_source = (task.path.parent / spec["table"]).resolve()
    shutil.copyfile(task.path, inputs / "task.json")
    frozen_table = inputs / spec["table"]
    frozen_table.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(table_source, frozen_table)
    frozen_task = read_task(inputs / "task.json")
    snapshot = run / "candidate"
    snapshot.mkdir()
    # Freeze the generator and its sibling Python helpers; no host project mount.
    for source in sorted(candidate.parent.glob("*.py")):
        if source.is_symlink():
            raise ValueError("Candidate helper symlinks are not supported")
        shutil.copyfile(source, snapshot / source.name)
    candidate_copy = snapshot / candidate.name
    base = {
        "schema": "qpe.report.v1",
        "task": task.name,
        "task_sha256": task.task_hash,
        "table_sha256": task.table_hash,
        "candidate_sha256": digest(candidate_copy),
        "candidate_source": str(candidate),
        "environment": prov,
        "judge_sha256": judge_fingerprint(prov),
        "judge_pin_sha256": pin_hash,
        "isolation": "bubblewrap: read-only code/input; private PID/network/tmp; writable output",
        "trust_scope": "isolates submitted generator; current interactive agent still has host access",
        "cost": {"status": "partial"},
    }
    base["candidate_files"] = {p.name: digest(p) for p in sorted(snapshot.glob("*.py"))}
    write_json(
        run / "invocation.json",
        {
            "task": str(task.path),
            "candidate": str(candidate),
            "timeout_seconds": timeout,
            "kernel_crosscheck": kernel,
        },
    )
    write_json(run / "provenance.json", base)
    started = time.monotonic()
    activity.run, activity.deadline = run, started + timeout
    try:
        activity.stage("generation")
        job = generate(
            root, candidate_copy, frozen_task, out, timeout, heartbeat=activity.heartbeat
        )
        base["generation"] = job
        if job["timed_out"]:
            raise TimeoutError("Generator exceeded total deadline")
        if job["returncode"]:
            raise ValueError(generator_failure(run))
        stream = out / "circuit.ops"
        if not stream.exists() or not stat.S_ISREG(stream.lstat().st_mode) or stream.is_symlink():
            raise ValueError("Generator must emit a regular circuit.ops file")
        if stream.stat().st_size > 64 * 1024**2:
            raise ValueError("Stream exceeds size cap")
        frozen = run / "circuit.ops"
        shutil.copyfile(stream, frozen)
        frozen.chmod(0o444)
        frozen_hash = digest(frozen)
        base["h_ops"] = frozen_hash
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("No time left for verification")
        command = [
            sys.executable,
            "-m",
            "qpe_lab",
            "_verify",
            str(inputs / "task.json"),
            str(frozen),
            str(run / "verification.json"),
        ]
        if kernel:
            command += ["--kernel"]
        env = minimal_env() | {"PYTHONPATH": str(root / "src"), "UNCOMPUTE_KERNEL": "1"}
        activity.stage("reference_verification")
        worker = run_job(
            command, root, run / "verifier.log", remaining, env, heartbeat=activity.heartbeat
        )
        base["verification_process"] = worker
        if worker["timed_out"]:
            raise TimeoutError("Verification exceeded total deadline")
        if not (run / "verification.json").exists():
            raise ValueError("Verification worker failed; see verifier.log")
        base.update(json.loads((run / "verification.json").read_text()))
        if worker["returncode"] != 0 and base["correctness_status"] in ACCEPTED:
            raise ValueError("Verification worker exited unsuccessfully after claiming acceptance")
        if digest(frozen) != frozen_hash or base["h_ops"] != frozen_hash:
            raise ValueError("Frozen artifact changed")
        if pin is not None:
            if digest(judge_path) != pin_hash:
                raise ValueError("Judge pin changed during evaluation")
            check_judge(pin, read_task(task_path), provenance(root))
    except Exception as exc:  # noqa: BLE001 -- persist a failure report at the job boundary
        base.update(
            correctness_status="timeout" if isinstance(exc, TimeoutError) else "error",
            detail=str(exc),
        )
    base["seconds"] = time.monotonic() - started
    write_json(run / "report.json", base)
    activity.stage("finished")
    return run, base


def verify_worker(task_path, stream_path, output, kernel):
    from qpe_lab.circuits.checks import LookupChecker
    from qpe_lab.circuits.compact_reader import interpret
    from qpe_lab.circuits.sim import Incorrect, Unsupported
    from qpe_lab.evaluation.kernel import sample_lookup

    task = read_task(task_path)
    result = {"h_ops": digest(stream_path), "cost": {"status": "partial"}}
    try:
        if isinstance(task, WalkTask):
            from qpe_lab.evaluation.walk import verify_walk

            result.update(verify_walk(task, stream_path))
            result["kernel_crosscheck"] = {
                "applicable": False,
                "reason": "upstream kernel does not support complex rotations; elementary and SOS simulators used instead",
            }
        elif isinstance(task, RotationTask):
            from qpe_lab.evaluation.rotation import verify_rotation

            result.update(
                verify_rotation(task, stream_path, Path(output).parent / "hybrid", kernel=kernel)
            )
        else:
            checker = LookupChecker(task)
            result["cost"] = interpret(stream_path, checker, task.max_qubits)
            result.update(correctness_status="exact_verified", evidence=checker.evidence())
            if kernel:
                destination = Path(output).parent / "kernel"
                destination.mkdir()
                rc, report = sample_lookup(task, Path(stream_path), destination)
                result["kernel_crosscheck"] = report
                if rc != 0 or report is None:
                    raise Incorrect("Independent C kernel cross-check failed")
                for key in ("toffoli", "clifford", "qubits", "ops"):
                    if result["cost"][key] != report["metrics"][key]:
                        raise Incorrect(f"Kernel/symbolic counter discrepancy: {key}")
    except Unsupported as exc:
        result.update(
            correctness_status="unsupported",
            detail=str(exc),
            verification_attempts=getattr(exc, "verification_attempts", []),
        )
    except (Incorrect, ValueError) as exc:
        result.update(correctness_status="fail", detail=str(exc))
    except Exception as exc:  # noqa: BLE001 -- never promote a checker exception to acceptance
        result.update(correctness_status="error", detail=f"{type(exc).__name__}: {exc}")
    write_json(output, result)
    return 0 if result["correctness_status"] in ACCEPTED else 1


def freeze_training(
    task_path,
    circuit_path,
    output,
    *,
    scope="core",
    acceptance="exact_or_sampled",
    enable_lean=False,
):
    """Maintainer-only: accept and freeze concrete gates before training starts.

    The training contract is a separate copy with the T-times-qubits objective.
    Subsequent evaluation never recompiles the reference or refreshes this pin.
    """
    from qpe_lab.circuits.checks import elementary_cost
    from qpe_lab.circuits.gadgets.synthesis import (
        core_reference,
        expand_clifford_t,
        expand_core_walk,
    )
    from qpe_lab.evaluation.walk import crosscheck_paper_walk, verify_walk

    folder, task, doc, original_task_hash, before = prepare_training(
        task_path,
        circuit_path,
        output,
        scope=scope,
        include_lean=enable_lean or acceptance == "lean_required",
    )
    report = verify_walk(task, folder / "baseline.ops")
    if report["correctness_status"] not in ACCEPTED or report["cost"]["status"] != "complete":
        raise ValueError("Training reference must have a complete accepted Clifford+T cost")
    gates = list(expand_clifford_t(task, doc, report["cost"]["synthesis"]))
    registers = report["cost"]["registers"]
    names = (
        "label",
        "term",
        "system",
        "spin_inner",
        "spin_outer",
        "enable",
        "outer_sigma",
        "inner_sigma",
        "outer_uniform",
        "inner_uniform",
        "outer_success",
        "inner_success",
    )
    protected = sorted(q for name in names for q in registers[name])
    if len(set(protected)) != len(protected) or len(protected) != report["cost"]["logical_qubits"]:
        raise ValueError("Incomplete protected input/output register contract")
    cost = elementary_cost(gates, protected)
    for key in ("t_count", "qubits", "tq_cost"):
        if cost[key] != report["cost"][key]:
            raise ValueError(f"Reference export/counter mismatch: {key}")
    if cost["tq_cost"] <= 0:
        raise ValueError("Training normalization requires positive reference TQ cost")
    error = report["cost"]["synthesis"]["walk_operator_error_bound"]
    if not math.isfinite(error) or not 0 <= error <= task.contract["walk_error_budget"]:
        raise ValueError("Reference exceeds the frozen synthesis error budget")
    reference = {
        "schema": "qpe.gate-reference.v2",
        "ancilla_initial": "zero",
        "protected_wires": protected,
        "gates": gates,
    }
    if scope == "core":
        reference, entry_costs = core_reference(task, doc)
        from qpe_lab.evaluation.cost import circuit_pair_cost

        cost = circuit_pair_cost(reference["entries"])
        cost["entries"] = entry_costs
        report["core_integration"] = crosscheck_paper_walk(task, doc, expander=expand_core_walk)
        # Equality to the frozen, already discrete core. The borrowed resource
        # state is an explicit precondition, not a free approximate rotation.
        error = 0.0
    return seal_training(
        folder, task, original_task_hash, before, reference, report, cost, error, scope, acceptance
    )


def verify_training_worker(
    bundle,
    expected_sha256,
    stream,
    output,
    challenge=None,
    *,
    exact_only=False,
    seconds=600,
    progress_path=None,
):
    """Data-only, isolated interpreter invocation; never imports submitted code."""
    from qpe_lab.circuits.ops import read_circuit, read_reference
    from qpe_lab.circuits.sim import Incorrect, Unsupported
    from qpe_lab.evaluation.cost import circuit_pair_cost
    from qpe_lab.evaluation.verification.engine import ProofCache
    from qpe_lab.evaluation.verification.sampled import SamplingContext

    deadline = time.monotonic() + seconds
    result = {
        "h_ops": digest(stream),
        "cost": {"status": "partial"},
        "reward": 0.0,
        "verification_schedule": {},
    }
    try:
        folder, task, manifest = load_training(bundle, expected_sha256)
        reference = read_reference(folder / "reference.json")
        proposal = read_circuit(stream)
        reference_hash = manifest["files"]["reference.json"]
        if proposal["reference_sha256"] != reference_hash:
            raise Incorrect("Candidate targets a different frozen reference")
        entries = reference.get("entries", {"walk": reference})
        if set(proposal["entries"]) != set(entries):
            raise Incorrect("Candidate must implement every frozen entry exactly once")
        context = None
        if not exact_only:
            draw = json.loads(Path(challenge).read_text())
            if draw["h_ops"] != result["h_ops"] or draw["bundle_sha256"] != expected_sha256:
                raise Incorrect("Sampling challenge does not match the frozen submission")
            attempt = draw["attempt"]
            if (
                type(attempt) is not int
                or attempt < 1
                or draw["confidence_budget"]
                != manifest["verification"]["campaign_confidence_budget"]
                / (attempt * (attempt + 1))
            ):
                raise Incorrect("Sampling confidence budget differs from frozen policy")
            seed = bytes.fromhex(draw["seed"])
            if len(seed) != 32:
                raise Incorrect("Invalid sampling seed length")
            if hashlib.sha256(seed).hexdigest() != draw["seed_commitment"]:
                raise Incorrect("Sampling seed commitment mismatch")
            context = SamplingContext(
                seed,
                draw["confidence_budget"],
                shots=manifest["verification"]["basis_diagnostic_shots"],
            )
            result["sampling_campaign"] = {k: v for k, v in draw.items() if k != "seed"}
        checked, reports, cache = {}, {}, ProofCache()
        resources = {}
        if "contract" in reference:
            contract = reference["contract"]
            gradient = contract["registers"]["gradient"]
            catalyst = contract["catalyst"]
            if catalyst["state"] != "phase-gradient" or catalyst["bits"] != len(gradient):
                raise Incorrect("Invalid frozen phase-gradient contract")
            resources = {q: [len(gradient), -(1 << j)] for j, q in enumerate(gradient)}
        for index, (name, entry) in enumerate(entries.items()):
            candidate = proposal["entries"][name]

            def progress(data, entry_name=name):
                if progress_path is not None:
                    temporary = Path(progress_path).with_suffix(".next.json")
                    write_json(temporary, data | {"entry": entry_name})
                    temporary.replace(progress_path)

            allowance = max(0.001, (deadline - time.monotonic() - 2) / (len(entries) - index))

            gates, reports[name] = verify_entry(
                entry,
                candidate,
                reference_hash,
                max_qubits=manifest["limits"]["max_qubits"],
                exact_seconds=manifest["verification"]["exact_seconds_per_entry"]
                * max(1, seconds / manifest["limits"]["timeout_seconds"]),
                context=context,
                exact_only=exact_only,
                cache=cache,
                resource_inputs=resources,
                enabled_backends=manifest["verification"]["backends"],
                required_backend="lean"
                if manifest["verification"]["acceptance"] == "lean_required"
                else None,
                seconds=allowance,
                schedule=manifest["verification"]["schedule"],
                progress=progress,
            )
            result["verification_schedule"][name] = reports[name]["verification_schedule"]
            checked[name] = entry | {"gates": gates}
        if "walk" in checked:
            exported, report = checked["walk"], reports["walk"]
        else:
            exported = reference | {"entries": checked}
            report = {
                "correctness_status": "bounded_verified",
                "cost": circuit_pair_cost(checked),
                "evidence": {
                    "entries": {k: v["evidence"] for k, v in reports.items()},
                    "sampling": any(v["evidence"].get("sampling", False) for v in reports.values()),
                },
            }
        report.update(
            correctness_status="verified",
            verification_level=2,
            verification_label="验证通过",
            scientific_review={"status": "not_run"},
        )
        cost = report["cost"]
        if cost["qubits"] > manifest["limits"]["max_qubits"]:
            raise Unsupported("Shared physical allocation exceeds qubit limit")
        for key in ("operations", "clifford", "measurements", "classical_xor"):
            if cost[key] > manifest["limits"]["max_" + key]:
                raise Unsupported(f"Frozen training {key} limit exceeded")
        target = Path(output).with_name("verified-gates.json")
        target.write_text(
            json.dumps(
                exported,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        report["verification_schedule"] = result["verification_schedule"]
        result.update(report)
        result.update(
            scope=manifest["scope"],
            error_scope=manifest["error_scope"],
            verified_gates_sha256=digest(target),
            reference_error_bound=manifest["reference_error_bound"],
            walk_error_budget=task.contract["walk_error_budget"],
        )
    except Unsupported as exc:
        if "name" in locals():
            result["verification_schedule"][name] = getattr(exc, "verification_schedule", [])
        result.update(
            correctness_status="unsupported",
            detail=str(exc),
            verification_attempts=getattr(exc, "verification_attempts", []),
        )
    except (Incorrect, ValueError) as exc:
        result.update(correctness_status="fail", detail=str(exc))
    except Exception as exc:  # noqa: BLE001 -- acceptance is fail-closed
        result.update(correctness_status="error", detail=f"{type(exc).__name__}: {exc}")
    write_json(output, result)
    return 0 if result["correctness_status"] in ACCEPTED else 1


def evaluate_training(bundle, expected_sha256, candidate_path, *, deadline=None, progress=None):
    """Trainer API: fixed task, reference, deadline, judge and reward rule.

    Learners supply Python only. Expose this API via training-worker rather than
    giving learners the host shell or the ability to select the bundle digest.
    """
    activity = EvaluationProgress(progress)
    root = project_root()
    folder, task, manifest = load_training(bundle, expected_sha256)
    candidate = Path(candidate_path).resolve()
    if not candidate.is_file() or candidate.suffix != ".py":
        raise ValueError("Candidate must be a Python generator")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = root / "runs" / f"{stamp}_training"
    out, snapshot = run / "generator", run / "candidate"
    out.mkdir(parents=True)
    snapshot.mkdir()
    activity.run = run
    helpers = sorted(candidate.parent.glob("*.py"))
    if len(helpers) > 32:
        raise ValueError("Candidate exceeds the 32 Python-file limit")
    for source in helpers:
        if source.is_symlink() or not source.is_file() or source.stat().st_size > 1024**2:
            raise ValueError("Invalid candidate helper or helper exceeds 1 MiB")
        shutil.copyfile(source, snapshot / source.name)
    candidate_copy = snapshot / candidate.name
    report = {
        "schema": "qpe.training-report.v1",
        "bundle_sha256": expected_sha256,
        "judge_sha256": manifest["judge_sha256"],
        "task_sha256": task.task_hash,
        "candidate_files": {p.name: digest(p) for p in snapshot.glob("*.py")},
        "cost": {"status": "partial"},
        "reward": 0.0,
        "reward_policy": manifest["reward"],
        "correctness_status": "error",
        "isolation": "single-process seccomp + bubblewrap; only output file and log writable",
        "trust_scope": "trainer owns judge/bundle/results; learner has submission API only",
    }
    started = time.monotonic()
    timeout = manifest["limits"]["timeout_seconds"]
    if deadline is not None:
        timeout = min(timeout, max(0.01, deadline - time.time()))
    activity.deadline = started + timeout
    try:
        activity.stage("generation")
        job = generate(
            root,
            candidate_copy,
            task,
            out,
            timeout,
            reference=folder / "reference.json",
            heartbeat=activity.heartbeat,
        )
        report["generation"] = job
        if job["timed_out"]:
            raise TimeoutError("Generator exceeded frozen deadline")
        if job["returncode"]:
            raise ValueError(generator_failure(run))
        activity.stage("freezing")
        stream = out / "circuit.ops"
        if (
            stream.is_symlink()
            or not stat.S_ISREG(stream.lstat().st_mode)
            or stream.stat().st_size > 64 * 1024**2
        ):
            raise ValueError("Invalid generator output")
        frozen = run / "circuit.ops"
        shutil.copyfile(stream, frozen)
        frozen.chmod(0o444)
        report["h_ops"] = digest(frozen)
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("No time remains for verification")
        worker, verified = run_training_verifier(
            folder, expected_sha256, manifest, frozen, run, remaining, activity
        )
        report["verification_process"] = worker
        report.update(verified)
    except Exception as exc:  # noqa: BLE001 -- no failure path grants reward
        report.update(
            correctness_status="timeout" if isinstance(exc, TimeoutError) else "error",
            reward=0.0,
            detail=str(exc),
            failure_stage=activity.current["stage"],
        )
    report["seconds"] = time.monotonic() - started
    report["timing"] = {
        "total_seconds": time.monotonic() - activity.started,
        "work_budget_seconds": timeout,
        "generation_seconds": report.get("generation", {}).get("seconds"),
        "verification_seconds": report.get("verification_process", {}).get("seconds"),
        "last_progress": activity.last,
    }
    report["scope"] = manifest["scope"]
    report["feedback"] = evaluation_feedback(report, manifest["baseline_cost"])
    write_json(run / "report.json", report)
    activity.stage("finished")
    return run, report


def scientific_review(bundle, expected_sha256, run):
    """Maintainer review of the same frozen artifact; reward is method-independent."""
    folder, _, manifest = load_training(bundle, expected_sha256)
    run = Path(run).resolve()
    report = json.loads((run / "report.json").read_text())
    if report["bundle_sha256"] != expected_sha256 or digest(run / "circuit.ops") != report["h_ops"]:
        raise ValueError("Review artifact differs from the accepted submission")
    if report["correctness_status"] not in ACCEPTED:
        raise ValueError("Scientific review requires a verified candidate")
    target = run / "scientific-review"
    target.mkdir(exist_ok=True)
    verify_training_worker(
        folder, expected_sha256, run / "circuit.ops", target / "verification.json", exact_only=True
    )
    exact = json.loads((target / "verification.json").read_text())
    review = {
        "status": "passed" if exact["correctness_status"] in ACCEPTED else "incomplete",
        "h_ops": report["h_ops"],
        "bundle_sha256": expected_sha256,
        "reference_checks_sha256": manifest["files"]["baseline-verification.json"],
        "scope": "frozen reference component/physics checks and exact candidate equivalence under the declared input contract",
        "evidence": exact.get("evidence"),
        "detail": exact.get("detail"),
    }
    if review["status"] == "passed":
        if exact["cost"]["tq_cost"] != report["cost"]["tq_cost"]:
            raise ValueError("Review resource count changed")
        report.update(verification_level=3, verification_label="科学复核通过")
    elif exact["correctness_status"] == "fail":
        review["status"] = "counterexample"
        report.update(
            correctness_status="fail",
            reward=0.0,
            verification_level=1,
            verification_label="验证失败",
        )
    load_training(folder, expected_sha256)
    report["scientific_review"] = review
    report["feedback"] = evaluation_feedback(report, manifest["baseline_cost"])
    write_json(run / "scientific-review.json", review)
    write_json(run / "report.json", report)
    return review


def complete_verification(bundle, expected_sha256, run_path, *, progress=None):
    """Maintainer-only, one bounded continuation of the frozen candidate.

    The generator is never rerun. A new challenge consumes the same persistent
    campaign budget, and the completed result replaces the unresolved label.
    """
    import fcntl

    folder, _, manifest = load_training(bundle, expected_sha256)
    run = Path(run_path).resolve()
    report_path = run / "report.json"
    with (run / "completion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = json.loads(report_path.read_text())
        if report.get("bundle_sha256") != expected_sha256:
            raise ValueError("Completion must use the original frozen bundle")
        feedback = evaluation_feedback(report, manifest["baseline_cost"])
        if not feedback["training_signal"]["deferred_eligible"]:
            raise ValueError("Only unresolved, frozen submissions can receive completion")
        frozen = run / "circuit.ops"
        if frozen.is_symlink() or digest(frozen) != report["h_ops"]:
            raise ValueError("Frozen artifact changed before completion")
        for name, expected in report["candidate_files"].items():
            if digest(run / "candidate" / name) != expected:
                raise ValueError("Generator snapshot changed before completion")
        target = run / "completion"
        target.mkdir(exist_ok=False)  # Reservation persists even if interrupted.
        write_json(target / "original-report.json", report)
        timeout = manifest["verification"]["deferred_timeout_seconds"]
        report["completion"] = {"status": "running", "budget_seconds": timeout}
        write_json(report_path, report)
        activity = EvaluationProgress(progress)
        activity.run, activity.deadline = target, time.monotonic() + timeout
        try:
            worker, checked = run_training_verifier(
                folder, expected_sha256, manifest, frozen, target, timeout, activity
            )
            report.pop("failure_stage", None)
            report.pop("detail", None)
            report.update(checked)
            if checked["correctness_status"] in ACCEPTED:
                shutil.copyfile(target / "verified-gates.json", run / "verified-gates.json")
            report["completion"] |= {"status": "finished", "verification_process": worker}
        except Exception as exc:  # noqa: BLE001 -- infrastructure failure never grants reward
            report.update(correctness_status="error", reward=0.0, detail=str(exc))
            report["completion"]["status"] = "error"
        report["completion"]["seconds"] = time.monotonic() - activity.started
        report["completion"]["last_progress"] = activity.last
        report["feedback"] = evaluation_feedback(report, manifest["baseline_cost"])
        write_json(report_path, report)
        activity.stage("finished")
        return submission_response(run, report)


def generator_failure(run):
    """Return the candidate's own bounded error output to the submission client."""
    with (run / "generator.log").open("rb") as log:
        log.seek(0, 2)
        log.seek(max(0, log.tell() - 4000))
        detail = log.read().decode("utf-8", errors="replace").strip()
    return f"Generator failed: {detail or 'see generator.log'}"


def submission_response(run, report):
    return {
        key: report.get(key)
        for key in (
            "correctness_status",
            "reward",
            "cost",
            "verification_level",
            "verification_label",
            "scientific_review",
            "scope",
            "feedback",
            "detail",
            "seconds",
            "timing",
            "verification_schedule",
            "completion",
        )
    } | {"run": str(run)}


def training_worker(bundle, expected_sha256, *, progress=None):
    """Trainer-launched JSONL service: the only learner input is Python source."""
    import tempfile

    from qpe_lab.circuits.ops import _reject_constant, _unique_object

    load_training(bundle, expected_sha256)  # Fail before accepting any request.
    while raw := sys.stdin.buffer.readline(1024**2 + 1):
        result = {"correctness_status": "error", "reward": 0.0}
        try:
            if len(raw) > 1024**2:
                raise ValueError("Submission line exceeds 1 MiB")
            request = json.loads(
                raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
            if (
                not isinstance(request, dict)
                or set(request) != {"source"}
                or not isinstance(request["source"], str)
            ):
                raise ValueError("Expected exactly one Python source field")
            with tempfile.TemporaryDirectory(
                dir=project_root() / "runs", prefix="submission-"
            ) as temp:
                candidate = Path(temp) / "generator.py"
                candidate.write_text(request["source"])
                run, result = evaluate_training(
                    bundle, expected_sha256, candidate, progress=progress
                )
                result = submission_response(run, result)
        except Exception as exc:  # noqa: BLE001 -- untrusted service input
            result["detail"] = str(exc)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if len(raw) > 1024**2:
            return 1
    return 0
