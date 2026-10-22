"""Prepare, seal and load frozen task bundles and judge/runtime identities."""

import base64
import hashlib
import importlib.metadata
import json
import platform
import stat
import subprocess
import sys
from pathlib import Path

from qpe_lab.contracts import (
    digest,
    project_root,
    sampling_policy,
    write_json,
)
from qpe_lab.tasks.task import read_task


def provenance(root, *, include_lean=False):
    upstream = root / "third_party/uncompute"
    pin = json.loads((root / "config/upstream.json").read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"], text=True
    ).strip()
    if revision != pin["commit"] or dirty or digest(upstream / "uv.lock") != pin["uv_lock_sha256"]:
        raise ValueError("Upstream version/working tree changed")
    sources = {
        str(p.relative_to(root)): digest(p) for p in sorted((root / "src/qpe_lab").rglob("*.py"))
    }
    sources["scripts/run_dfthc.py"] = digest(root / "scripts/run_dfthc.py")
    runtime = {}
    for name in ("pygridsynth", "mpmath", "numpy", "pyzx", "typing_extensions"):
        distribution = importlib.metadata.distribution(name)
        runtime[name] = {
            "version": distribution.version,
            "files": {
                str(p): digest(distribution.locate_file(p))
                for p in distribution.files
                if str(p).endswith((".py", ".so"))
            },
        }
    lean = root / ".tools/lean-4.35.0-linux"
    if include_lean:
        if not (lean / "bin/lake").is_file():
            raise ValueError("Install the pinned Lean toolchain before enabling its backend")
        # Pin the complete executable/import runtime, excluding static archives
        # and human-readable sources which the proof-check path never loads.
        identity = hashlib.sha256()
        for p in sorted(
            p
            for top in (lean / "bin", lean / "lib")
            for p in top.rglob("*")
            if p.is_file() and p.suffix != ".a"
        ):
            identity.update(str(p.relative_to(lean)).encode() + b"\0")
            identity.update(bytes.fromhex(digest(p)))
        runtime["lean"] = {
            "version": "4.35.0",
            "runtime_sha256": identity.hexdigest(),
            "sandbox_sha256": digest(root / ".tools/bubblewrap-0.11.0/bwrap"),
        }
    return {
        "upstream": pin,
        "python": sys.version,
        "source_hashes": sources,
        "runtime_packages": runtime,
        "dependency_lock_sha256": digest(root / "uv.lock"),
        "platform": platform.platform(),
        "bubblewrap_sha256": digest(root / ".tools/bubblewrap/usr/bin/bwrap"),
        "kernel_sha256": digest(upstream / "uncompute/evalcore/_kernel.so"),
    }


def judge_fingerprint(prov):
    return hashlib.sha256(
        json.dumps(prov, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze_judge(task_path, output, *, include_lean=False):
    """Pin the judge/task before a search; never refresh an existing pin silently."""
    output = Path(output)
    if output.exists():
        raise ValueError("Judge pin already exists; use it or explicitly choose a new path")
    task = read_task(task_path)
    prov = provenance(project_root(), include_lean=include_lean)
    pin = {
        "schema": "qpe.judge.v1",
        "judge_sha256": judge_fingerprint(prov),
        "task_sha256": task.task_hash,
        "table_sha256": task.table_hash,
        "environment": prov,
        "source_snapshot_b64": {
            name: base64.b64encode((project_root() / name).read_bytes()).decode()
            for name in prov["source_hashes"]
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, pin)
    return pin


def check_judge(pin, task, prov):
    if (
        pin.get("schema") != "qpe.judge.v1"
        or pin.get("judge_sha256") != judge_fingerprint(prov)
        or pin.get("task_sha256") != task.task_hash
        or pin.get("table_sha256") != task.table_hash
    ):
        raise ValueError("Frozen judge, runtime, or task changed; candidate is not comparable")


def load_training(bundle, expected_sha256):
    """Resolve a trainer-owned pin. The caller must not take it from the learner."""
    from qpe_lab.circuits.ops import _reject_constant, _unique_object

    folder = Path(bundle).resolve()
    path = folder / "manifest.json"
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or digest(path) != expected_sha256
    ):
        raise ValueError("Training bundle digest differs from the trainer's fixed pin")
    manifest = json.loads(
        path.read_bytes(), object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    names = {
        "task.json",
        "model.json",
        "baseline.ops",
        "baseline-verification.json",
        "reference.json",
        "judge.json",
    }
    if (
        manifest.get("schema") != "qpe.training.v3"
        or manifest.get("scope") not in ("core", "walk")
        or set(manifest["files"]) != names
    ):
        raise ValueError("Invalid training bundle manifest")
    for name, expected in manifest["files"].items():
        source = folder / name
        if (
            source.is_symlink()
            or not stat.S_ISREG(source.lstat().st_mode)
            or digest(source) != expected
        ):
            raise ValueError(f"Frozen training input changed: {name}")
    task = read_task(folder / "task.json")
    if task.contract["objective"] != "tq_cost":
        raise ValueError("Training requires the T-times-physical-qubits objective")
    pin = json.loads((folder / "judge.json").read_text())
    check_judge(
        pin,
        task,
        provenance(
            project_root(),
            include_lean="lean" in manifest.get("verification", {}).get("backends", []),
        ),
    )
    if pin["judge_sha256"] != manifest["judge_sha256"]:
        raise ValueError("Training judge does not match its manifest")
    if manifest["reward"] != {
        "formula": "C0/(C0+TQ)",
        "C0": manifest["baseline_cost"]["tq_cost"],
        "rejected": 0,
    }:
        raise ValueError("Invalid frozen reward policy")
    if manifest.get("verification") != sampling_policy(
        manifest.get("verification", {}).get("acceptance"),
        lean="lean" in manifest.get("verification", {}).get("backends", []),
    ):
        raise ValueError("Invalid frozen verification policy")
    return folder, task, manifest


def prepare_training(task_path, circuit_path, output, *, scope, include_lean=False):
    """Copy the fixed inputs; reference verification is the caller's next step."""
    import shutil

    from qpe_lab.circuits.ops import read_walk
    from qpe_lab.contracts import WalkTask

    task = read_task(task_path)
    if scope not in ("core", "walk"):
        raise ValueError("Training scope must be core or walk")
    doc = read_walk(circuit_path)
    if (
        not isinstance(task, WalkTask)
        or doc.get("implementation", {}).get("kind") != "paper_strict"
    ):
        raise ValueError("Training freeze currently requires a strict DFTHC walk")
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    spec = json.loads(task.path.read_text())
    original_task_hash = task.task_hash
    source = (task.path.parent / spec["table"]).resolve()
    spec["table"] = "model.json"
    spec["optimization"]["objective"] = "tq_cost"
    shutil.copyfile(source, folder / "model.json")
    write_json(folder / "task.json", spec)
    task = read_task(folder / "task.json")
    before = provenance(project_root(), include_lean=include_lean)
    shutil.copyfile(circuit_path, folder / "baseline.ops")
    doc = read_walk(folder / "baseline.ops")
    return folder, task, doc, original_task_hash, before


def seal_training(
    folder,
    task,
    original_task_hash,
    before,
    reference,
    report,
    cost,
    error,
    scope,
    acceptance="exact_or_sampled",
):
    """Seal a reference already checked by the maintainer evaluation pipeline."""
    (folder / "reference.json").write_text(
        json.dumps(
            reference,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    write_json(folder / "baseline-verification.json", report)
    include_lean = "lean" in before["runtime_packages"]
    pin = freeze_judge(task.path, folder / "judge.json", include_lean=include_lean)
    if pin["judge_sha256"] != judge_fingerprint(before):
        raise ValueError("Trusted code changed while freezing the training reference")
    manifest = {
        "schema": "qpe.training.v3",
        "scope": scope,
        "original_task_sha256": original_task_hash,
        "judge_sha256": pin["judge_sha256"],
        "files": {
            name: digest(folder / name)
            for name in (
                "task.json",
                "model.json",
                "baseline.ops",
                "baseline-verification.json",
                "reference.json",
                "judge.json",
            )
        },
        "baseline_cost": cost,
        "reference_error_bound": error,
        "error_scope": "exact core gate equivalence; borrowed-state preparation excluded"
        if scope == "core"
        else "complete walk synthesis",
        "reward": {"formula": "C0/(C0+TQ)", "C0": cost["tq_cost"], "rejected": 0},
        "limits": {
            "timeout_seconds": 600,
            "max_operations": 2_000_000,
            "max_qubits": task.max_qubits,
            "max_clifford": 1_000_000,
            "max_measurements": 100_000,
            "max_classical_xor": 1_000_000,
        },
        "verification": sampling_policy(acceptance, lean=include_lean),
        "trust_scope": "frozen concrete entries; core borrows the specified phase-gradient state; full-walk preparation is outside core reward",
    }
    write_json(folder / "manifest.json", manifest)
    return {
        "bundle": str(folder),
        "bundle_sha256": digest(folder / "manifest.json"),
        "baseline_cost": {k: cost[k] for k in ("t_count", "qubits", "tq_cost")},
        "judge_sha256": pin["judge_sha256"],
        "scope": scope,
    }
