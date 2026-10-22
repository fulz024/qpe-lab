"""Trainer-owned optimization sessions and a model-independent JSONL tool boundary.

Model orchestration stays outside this process. Give the learner this toolbox,
not a host shell: file access is virtualized, Python experiments use the same
OS isolation as submissions, and budgets/results live outside its workspace.
"""

import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

from qpe_lab.circuits.ops import _reject_constant, _unique_object
from qpe_lab.contracts import digest, project_root, write_json
from qpe_lab.evaluation.cost import evaluation_feedback
from qpe_lab.evaluation.protocol import (
    ACCEPTED,
    EvaluationProgress,
    evaluate,
    evaluate_training,
    freeze_training,
    submission_response,
)
from qpe_lab.evaluation.sandbox import generate
from qpe_lab.tasks.bundle import load_training
from qpe_lab.tasks.prompt import agent_task

MAX_TEXT = 1024**2


class AgentSession:
    """One exclusively opened session; restart preserves budget and best artifacts."""

    def __init__(
        self,
        path,
        *,
        bundle=None,
        bundle_sha256=None,
        max_submissions=None,
        seconds=None,
        initial_candidate=None,
        progress=None,
    ):
        import fcntl

        self.root = project_root()
        self.progress = progress
        self.path = Path(path).resolve()
        runs = (self.root / "runs").resolve()
        if self.path == runs or not self.path.is_relative_to(runs):
            raise ValueError("Session directory must be a dedicated folder inside runs/")
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = (self.path / "session.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.path / "session.json").exists():
                self.state = json.loads((self.path / "session.json").read_text())
                for key, value in (
                    ("bundle_sha256", bundle_sha256),
                    ("max_submissions", max_submissions),
                    ("seconds", seconds),
                ):
                    if value is not None and self.state[key] != value:
                        raise ValueError(f"Resume must preserve the existing {key}")
                if bundle is not None and Path(bundle).resolve() != Path(self.state["bundle"]):
                    raise ValueError("Resume must preserve the frozen bundle")
                if initial_candidate is not None:
                    raise ValueError("An existing session already has its initial candidate")
            else:
                self.create(bundle, bundle_sha256, max_submissions, seconds, initial_candidate)
            self.bundle, self.task, self.manifest = load_training(
                self.state["bundle"], self.state["bundle_sha256"]
            )
            self.workspace = self.path / "workspace"
            self.workspace.mkdir(exist_ok=True)
        except BaseException:
            self.lock.close()
            raise

    def close(self):
        self.lock.close()

    def save(self):
        temporary = self.path / "session.next.json"
        write_json(temporary, self.state)
        temporary.replace(self.path / "session.json")

    def create(self, bundle, pin, submissions, seconds, initial_candidate):
        submissions = 20 if submissions is None else submissions
        seconds = 1800 if seconds is None else seconds
        if type(submissions) is not int or not 1 <= submissions <= 10000:
            raise ValueError("max_submissions must be between 1 and 10000")
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be positive and finite")
        if (bundle is None) != (pin is None):
            raise ValueError("Supply both bundle and bundle_sha256, or neither")
        if bundle is None:
            # Maintainer invocation only; no learner action can reach this path.
            task = self.root / "benchmarks/dfthc/fe2s2/walk-paper-strict/walk.task.json"
            run, report = evaluate(
                task,
                self.root / "candidates/dfthc_walk/optimized.py",
                timeout=300,
                progress=self.progress,
            )
            if report["correctness_status"] not in ACCEPTED:
                raise ValueError(f"Reference construction failed; inspect {run}")
            frozen = freeze_training(task, run / "circuit.ops", self.path / "bundle")
            bundle, pin = frozen["bundle"], frozen["bundle_sha256"]
        bundle, _, manifest = load_training(bundle, pin)
        preflight = self.path / "preflight"
        preflight.mkdir(exist_ok=False)
        reference = preflight / "generator.py"
        reference.write_text("from qpe_lab import CircuitBuilder\nCircuitBuilder().emit()\n")
        reference_run, reference_report = evaluate_training(
            bundle, pin, reference, progress=self.progress
        )
        if (
            reference_report["correctness_status"] not in ACCEPTED
            or reference_report["reward"] != 0.5
        ):
            raise ValueError(f"Reference smoke test failed; inspect {reference_run}")
        initial = Path(initial_candidate or self.root / "candidates/dfthc_walk/reuse.py").resolve()
        initial_run, initial_report = evaluate_training(
            bundle, pin, initial, progress=self.progress
        )
        if initial_report["correctness_status"] not in ACCEPTED:
            raise ValueError(f"Initial candidate did not verify; inspect {initial_run}")
        self.state = {
            "schema": "qpe.agent-session.v1",
            "bundle": str(bundle),
            "bundle_sha256": pin,
            "scope": manifest["scope"],
            "max_submissions": submissions,
            "seconds": seconds,
            "expires_at": time.time() + seconds,
            "submissions": 0,
            "experiments": 0,
            "tool_calls": 0,
            "history": [
                {"run": str(reference_run), "candidate": reference.name, "role": "reference"},
                {"run": str(initial_run), "candidate": initial.name, "role": "initial"},
            ],
        }
        workspace = self.path / "workspace"
        workspace.mkdir(exist_ok=True)
        start = min(
            self.state["history"],
            key=lambda r: json.loads((Path(r["run"]) / "report.json").read_text())["cost"][
                "tq_cost"
            ],
        )
        self.state["starter_candidate"] = start["candidate"]
        for source in (Path(start["run"]) / "candidate").glob("*.py"):
            shutil.copyfile(source, workspace / source.name)
        self.save()

    def budget(self):
        return {
            "submissions_used": self.state["submissions"],
            "submissions_remaining": max(
                0, self.state["max_submissions"] - self.state["submissions"]
            ),
            "experiments_remaining": max(
                0, 4 * self.state["max_submissions"] - self.state["experiments"]
            ),
            "seconds_remaining": max(0, self.state["expires_at"] - time.time()),
        }

    def reserve(self, action):
        if self.state["expires_at"] <= time.time():
            raise ValueError("Session time budget exhausted; task/read/best remain available")
        if self.state["tool_calls"] >= max(100, 20 * self.state["max_submissions"]):
            raise ValueError("Session work-tool budget exhausted; task/read/best remain available")
        if action == "submit":
            if self.state["submissions"] >= self.state["max_submissions"]:
                raise ValueError("Session submission budget exhausted")
            self.state["submissions"] += 1
        elif action == "experiment":
            if self.state["experiments"] >= 4 * self.state["max_submissions"]:
                raise ValueError("Session experiment budget exhausted")
            self.state["experiments"] += 1
        self.state["tool_calls"] += 1
        # Reserve before execution; crashes and worker restarts do not refund.
        self.save()

    def resolve(self, text, *, writable=False):
        if not isinstance(text, str) or len(text) > 512 or "\\" in text:
            raise ValueError("Use a virtual POSIX path")
        path = PurePosixPath(text)
        if not path.is_absolute() or ".." in path.parts or len(path.parts) > 12:
            raise ValueError("Use an absolute path inside /workspace or the read-only task files")
        roots = {"/workspace": self.workspace}
        if not writable:
            roots.update(
                {
                    "/reference.json": self.bundle / "reference.json",
                    "/task/task.json": self.bundle / "task.json",
                    "/task/model.json": self.bundle / "model.json",
                    **{
                        f"/app/{name}": self.root / name
                        for name in (
                            "src",
                            "docs",
                            "candidates",
                            "benchmarks",
                            "AGENTS.md",
                            "README.md",
                        )
                    },
                }
            )
        for virtual, actual in roots.items():
            try:
                relative = path.relative_to(virtual)
            except ValueError:
                continue
            result = actual / str(relative) if actual.is_dir() else actual
            if not actual.is_dir() and str(relative) != ".":
                continue
            # No learner-created symlink/hardlink exists through this API; still
            # reject symlinks so externally prepared workspaces are not aliases.
            current = actual
            if current.is_symlink():
                raise ValueError("Symbolic links are outside the session file interface")
            for component in relative.parts:
                current /= component
                if current.is_symlink():
                    raise ValueError("Symbolic links are outside the session file interface")
            if not result.resolve().is_relative_to(
                actual.resolve() if actual.is_dir() else actual.parent.resolve()
            ):
                raise ValueError("Path leaves its visible root")
            return result
        raise ValueError("Path is not exposed by this session")

    def best_record(self):
        accepted = []
        for record in self.state["history"]:
            run = Path(record["run"])
            report = json.loads((run / "report.json").read_text())
            if report.get("bundle_sha256") != self.state["bundle_sha256"]:
                raise ValueError("Session record belongs to another bundle")
            if report["correctness_status"] in ACCEPTED:
                if digest(run / "circuit.ops") != report["h_ops"]:
                    raise ValueError("Saved candidate artifact changed")
                if digest(run / "verified-gates.json") != report["verified_gates_sha256"]:
                    raise ValueError("Saved verified gates changed")
                for name, expected in report["candidate_files"].items():
                    if digest(run / "candidate" / name) != expected:
                        raise ValueError("Saved generator source changed")
                accepted.append((record, report))
        if not accepted:
            raise ValueError("No verified starting candidate remains")
        return min(accepted, key=lambda pair: pair[1]["cost"]["tq_cost"])

    def best(self):
        record, report = self.best_record()
        folder = Path(record["run"]) / "candidate"
        return {
            "candidate": record["candidate"],
            "files": {p.name: p.read_text() for p in sorted(folder.glob("*.py"))},
            "result": submission_response(record["run"], report),
            "bundle_sha256": self.state["bundle_sha256"],
        }

    def delivery(self):
        context = agent_task(
            self.bundle,
            self.state["bundle_sha256"],
            workspace=self.workspace,
            budget=f"{self.state['max_submissions']}次正式提交，{self.state['seconds']}秒；服务执行预算",
            session=True,
        )
        best = self.best()
        context["best"] = best
        context["starter_path"] = "/workspace/" + self.state["starter_candidate"]
        context["tools"] = {
            "task": {"action": "task"},
            "read": {"action": "read", "path": "/workspace", "offset": 0, "limit": 65536},
            "write": {
                "action": "write",
                "path": "/workspace/generator.py",
                "text": "Python source",
            },
            "experiment": {
                "action": "experiment",
                "source": "print('local experiment')",
                "timeout": 30,
            },
            "submit": {"action": "submit", "path": context["starter_path"]},
            "best": {"action": "best"},
        }
        context["system_prompt"] += (
            "\n\n本轮起点已由服务复现，读取best获取完整源码。当前最佳："
            + json.dumps({k: best["result"][k] for k in ("cost", "reward", "verification_label")})
            + "\n本轮工具请求格式："
            + json.dumps(context["tools"], ensure_ascii=False)
            + "\nread的offset/limit用于字节分页；目录时用于条目分页。"
            "write只写/workspace，experiment使用其只读快照。"
            "submit也可直接提供source，不能同时提供path。实验和正式提交共享会话时间。"
        )
        return context

    def handle(self, request):
        if not isinstance(request, dict):
            raise TypeError("A tool request must be a JSON object")
        action = request.get("action")
        allowed = {
            "task": {"action"},
            "best": {"action"},
            "read": {"action", "path", "offset", "limit"},
            "write": {"action", "path", "text"},
            "experiment": {"action", "source", "timeout"},
            "submit": {"action", "source", "path"},
        }
        if action not in allowed or not set(request) <= allowed[action]:
            raise ValueError("Unknown tool or extra tool arguments")
        if action == "task":
            return self.delivery()
        if action == "best":
            return self.best()
        if action == "read":
            path = self.resolve(request.get("path"))
            offset, limit = request.get("offset", 0), request.get("limit", 65536)
            if (
                type(offset) is not int
                or offset < 0
                or type(limit) is not int
                or not 1 <= limit <= MAX_TEXT
            ):
                raise ValueError("Invalid read page")
            if path.is_dir():
                entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
                return {
                    "entries": entries[offset : offset + min(limit, 256)],
                    "total": len(entries),
                }
            with path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(limit)
            return {
                "text": data.decode("utf-8", errors="replace"),
                "next_offset": offset + len(data),
                "bytes": path.stat().st_size,
            }
        if action == "write":
            path, text = self.resolve(request.get("path"), writable=True), request.get("text")
            if not isinstance(text, str) or len(text.encode()) > MAX_TEXT or path == self.workspace:
                raise ValueError("Write a text file of at most 1 MiB")
            files = [p for p in self.workspace.rglob("*") if p.is_file()]
            size = sum(p.stat().st_size for p in files) - (
                path.stat().st_size if path.is_file() else 0
            )
            if (
                len(files) + int(not path.exists()) > 128
                or size + len(text.encode()) > 8 * MAX_TEXT
            ):
                raise ValueError("Workspace quota: 128 files, 8 MiB total")
            self.reserve(action)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return {"path": request["path"], "bytes": len(text.encode())}
        if action == "experiment":
            source, timeout = request.get("source"), request.get("timeout", 30)
            if not isinstance(source, str) or len(source.encode()) > MAX_TEXT:
                raise ValueError("Experiment source must be at most 1 MiB")
            if (
                type(timeout) not in (int, float)
                or not math.isfinite(timeout)
                or not 0 < timeout <= 60
            ):
                raise ValueError("Experiment timeout must be at most 60 seconds")
            self.reserve(action)
            with tempfile.TemporaryDirectory(dir=self.path, prefix="experiment-") as temp:
                folder = Path(temp)
                code, out = folder / "code", folder / "output"
                code.mkdir()
                out.mkdir()
                candidate = code / "generator.py"
                candidate.write_text(source, encoding="utf-8")
                activity = EvaluationProgress(self.progress)
                limit = min(timeout, max(0.01, self.state["expires_at"] - time.time()))
                activity.deadline = time.monotonic() + limit
                activity.stage("experiment")
                job = generate(
                    self.root,
                    candidate,
                    self.task,
                    out,
                    limit,
                    reference=self.bundle / "reference.json",
                    workspace=self.workspace,
                    heartbeat=activity.heartbeat,
                )
                activity.stage("finished")
                log = folder / "generator.log"
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 32768))
                    text = stream.read().decode("utf-8", errors="replace")
                return {"job": job, "output": text, "reward": None, "purpose": "experiment_only"}
        if ("source" in request) == ("path" in request):
            raise ValueError("Submit exactly one of source or path")
        previous = self.best_record()[1]["cost"]
        with tempfile.TemporaryDirectory(dir=self.path, prefix="submission-") as temp:
            if "source" in request:
                source = request["source"]
                if not isinstance(source, str) or len(source.encode()) > MAX_TEXT:
                    raise ValueError("Submission source must be at most 1 MiB")
                candidate = Path(temp) / "generator.py"
                candidate.write_text(source, encoding="utf-8")
            else:
                candidate = self.resolve(request["path"], writable=True)
                if not candidate.is_file() or candidate.suffix != ".py":
                    raise ValueError("Submit a Python file from /workspace")
            self.reserve(action)
            run, report = evaluate_training(
                self.bundle,
                self.state["bundle_sha256"],
                candidate,
                deadline=self.state["expires_at"],
                progress=self.progress,
            )
            report["feedback"] = evaluation_feedback(
                report, self.manifest["baseline_cost"], previous
            )
            report["session"] = str(self.path)
            write_json(run / "report.json", report)
            self.state["history"].append(
                {"run": str(run), "candidate": candidate.name, "role": "learner"}
            )
            self.save()
            return submission_response(run, report)


def session_worker(path, **options):
    """Initial ready event, then one JSON reply per request. No model credentials."""
    session = AgentSession(path, **options)
    try:
        print(
            json.dumps(
                {"event": "ready", "task": session.delivery(), "budget": session.budget()},
                ensure_ascii=False,
            ),
            flush=True,
        )
        while raw := sys.stdin.buffer.readline(MAX_TEXT + 1):
            try:
                if len(raw) > MAX_TEXT:
                    raise ValueError("Tool request exceeds 1 MiB")
                request = json.loads(
                    raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
                )
                result = {"ok": True, "result": session.handle(request)}
            except Exception as exc:  # noqa: BLE001 -- untrusted requests get no authority
                result = {"ok": False, "detail": str(exc)}
            result["budget"] = session.budget()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if len(raw) > MAX_TEXT:
                return 1
        return 0
    finally:
        session.close()
