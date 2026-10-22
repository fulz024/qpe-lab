"""Real Python generator tests, including the previously identified execution gaps."""

import json
import tempfile
import unittest
from pathlib import Path

from test_lookup import make_task

from qpe_lab.contracts import project_root
from qpe_lab.evaluation.protocol import evaluate
from qpe_lab.tasks.task import load_walk


class RunnerTests(unittest.TestCase):
    def setUp(self):
        # Keep fixtures under ignored runs, which is accessible from both host and WSL.
        self.temp = tempfile.TemporaryDirectory(dir=project_root() / "runs", prefix="runner-test-")
        self.folder = Path(self.temp.name)
        self.task = make_task(self.folder / "task")
        (self.folder / "candidate").mkdir()
        self.candidate = self.folder / "candidate/generator.py"

    def tearDown(self):
        self.temp.cleanup()

    def run_candidate(self, code, timeout=20):
        self.candidate.write_text(code)
        return evaluate(self.task.path, self.candidate, timeout=timeout)

    def test_regular_generator_and_kernel(self):
        streams = []
        for module in ("qpe_lab", "qpe_lab.circuits.builder"):
            with self.subTest(module=module):
                folder, report = self.run_candidate(
                    f"from {module} import build_lookup\nfrom qpe_lab import load_task\n"
                    "b,_,_=build_lookup(load_task())\nb.emit()\n"
                )
                self.assertEqual(report["correctness_status"], "exact_verified", report)
                self.assertEqual(report["kernel_crosscheck"]["verdict"], "OK")
                self.assertTrue((folder / "candidate/generator.py").is_file())
                self.assertIn("generator.py", report["candidate_files"])
                streams.append(report["h_ops"])
        self.assertEqual(
            len(set(streams)),
            1,
            "Public and circuit-layer builders must generate identical streams",
        )

    def test_slow_generator_is_killed_before_emitting(self):
        _, report = self.run_candidate("import time\ntime.sleep(10)\n", timeout=0.3)
        self.assertEqual(report["correctness_status"], "timeout", report)
        self.assertLess(report["seconds"], 3)

    def test_generator_cannot_write_grader_or_escape_mounts(self):
        canary = self.folder / "outside-output.txt"
        host_root = str(project_root())
        code = f"""import pathlib, socket
from qpe_lab import build_lookup
blocked = []
for path in ['/opt/uncompute/tasks/chemistry-fe2s2/grader.py', '/task/table.json', {str(canary)!r}]:
    try:
        with open(path, 'ab') as f: f.write(b'')
    except OSError: blocked.append(path)
assert len(blocked) == 3, blocked
assert not pathlib.Path({host_root!r} + '/config/upstream.json').exists()
sock = socket.socket(); sock.settimeout(0.2)
try:
    sock.connect(('1.1.1.1', 443))
except OSError: pass
else: raise AssertionError('network is reachable')
finally: sock.close()
b,_,_=build_lookup(); b.emit()
"""
        _, report = self.run_candidate(code)
        self.assertEqual(report["correctness_status"], "exact_verified", report)
        self.assertFalse(canary.exists())

    def test_wrong_phase_never_gets_exact_status(self):
        _, report = self.run_candidate(
            "from qpe_lab import build_lookup\nb,_,target=build_lookup()\nb.z(target[0])\nb.emit()\n"
        )
        self.assertEqual(report["correctness_status"], "fail", report)
        self.assertIn("phase", report["detail"])

    def test_nonzero_exit_is_not_accepted(self):
        _, report = self.run_candidate(
            "from qpe_lab import build_lookup\nb,_,_=build_lookup()\nb.emit()\nraise SystemExit(7)\n"
        )
        self.assertEqual(report["correctness_status"], "error")

    def test_candidate_stdout_cannot_supply_verdict(self):
        _, report = self.run_candidate(
            "print(" + repr(json.dumps({"correctness_status": "exact_verified"})) + ")\n"
        )
        self.assertEqual(report["correctness_status"], "error")


class TrainingRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qpe_lab import build_walk
        from qpe_lab.evaluation.protocol import freeze_training

        cls.temp = tempfile.TemporaryDirectory(dir=project_root() / "runs", prefix="training-test-")
        cls.folder = Path(cls.temp.name)
        task = load_walk(project_root() / "benchmarks/dfthc/toy/walk-paper-strict/walk.task.json")
        build_walk(
            task,
            implementation="paper_strict",
            outer_block_size=2,
            qrom_block_size=2,
            outer_unlookup_size=4,
            unlookup_block_size=8,
        ).emit(cls.folder / "baseline.ops")
        cls.pin = freeze_training(
            task.path, cls.folder / "baseline.ops", cls.folder / "bundle", scope="walk"
        )
        cls.bundle = cls.folder / "bundle"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_candidate(self, source):
        from qpe_lab.evaluation.protocol import evaluate_training

        with tempfile.TemporaryDirectory(dir=self.folder, prefix="candidate-") as temp:
            candidate = Path(temp) / "generator.py"
            candidate.write_text(source)
            return evaluate_training(self.bundle, self.pin["bundle_sha256"], candidate)

    def test_baseline_and_agent_written_local_optimization(self):
        _, reference = self.run_candidate(
            "from qpe_lab import CircuitBuilder\nCircuitBuilder().emit()\n"
        )
        self.assertEqual(reference["correctness_status"], "verified", reference)
        self.assertEqual(reference["reward"], 0.5)
        code = """from qpe_lab import CircuitBuilder
from qpe_lab.circuits.gadgets.synthesis import specialize_zero_targets, zero_target_ccx_clifford_t, ccx_clifford_t
b = CircuitBuilder()
_, witness = specialize_zero_targets(b.gates)
start,a,c,target,_ = next(row for row in witness if row[-1])
b.replace([{'start':start, 'stop':start+len(list(ccx_clifford_t(a,c,target))),
            'gates':list(zero_target_ccx_clifford_t(a,c,target))}])
b.emit()
"""
        run, improved = self.run_candidate(code)
        self.assertEqual(improved["correctness_status"], "verified", improved)
        self.assertEqual(improved["cost"]["t_count"], reference["cost"]["t_count"] - 3)
        self.assertGreater(improved["reward"], 0.5)
        self.assertTrue((run / "verified-gates.json").is_file())

    def test_sampling_acceptance_and_independent_scientific_review(self):
        from qpe_lab.evaluation.protocol import scientific_review

        # A wide word with a fresh internal record exercises sampled channel
        # verification; its coherent-only counterpart now has a tensor proof.
        source = """from qpe_lab import CircuitBuilder
b=CircuitBuilder()
identity=[]
for i in range(2050):
 q=b.protected_wires[i%5]
 identity += [('h',(q,),None),('h',(q,),None)]
scratch=next(q for _,qs,_ in b.gates for q in qs if q not in b.protected_wires)
record=max(v for n,_,v in b.gates if n=='hmr')+1
identity += [('hmr',(scratch,),record)]
b.submit(identity+b.gates)
b.emit()
"""
        run, report = self.run_candidate(source)
        self.assertEqual(report["correctness_status"], "verified", report)
        self.assertEqual(report["verification_level"], 2)
        self.assertEqual(report["reward"], 0.5)
        self.assertTrue(report["evidence"]["sampling"])
        review = scientific_review(self.bundle, self.pin["bundle_sha256"], run)
        self.assertEqual(review["status"], "incomplete")
        self.assertEqual(json.loads((run / "report.json").read_text())["reward"], 0.5)
        exact_run, exact = self.run_candidate(
            "from qpe_lab import CircuitBuilder\nCircuitBuilder().emit()\n"
        )
        self.assertEqual(
            scientific_review(self.bundle, self.pin["bundle_sha256"], exact_run)["status"], "passed"
        )
        reviewed = json.loads((exact_run / "report.json").read_text())
        self.assertEqual(reviewed["verification_level"], 3)
        self.assertEqual(reviewed["reward"], exact["reward"])

    def test_exact_deadline_is_not_swallowed_by_proof_search(self):
        import time

        from qpe_lab.circuits.sim import Unsupported
        from qpe_lab.evaluation.verification.engine import timed_verification

        def search():
            end = time.monotonic() + 0.2
            while time.monotonic() < end:
                try:
                    time.sleep(0.02)
                except Unsupported:
                    continue
            return "unexpected"

        with self.assertRaisesRegex(Unsupported, "time budget"):
            timed_verification(search, 0.03)

    def test_persistent_sampling_budget_and_fresh_challenges(self):
        from concurrent.futures import ThreadPoolExecutor

        from qpe_lab.evaluation.verification.sampled import reserve_sampling_attempt

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ledger.sqlite3"
            with ThreadPoolExecutor(max_workers=4) as pool:
                draws = list(
                    pool.map(lambda _: reserve_sampling_attempt(path, "bundle", "gates"), range(12))
                )
            self.assertEqual(sorted(d["attempt"] for d in draws), list(range(1, 13)))
            self.assertEqual(len({d["seed"] for d in draws}), 12)
            self.assertLess(sum(d["confidence_budget"] for d in draws), 1e-9)
            self.assertEqual(
                reserve_sampling_attempt(path, "another-bundle", "same-gates")["attempt"], 13
            )

    def test_bad_gate_and_generator_monkeypatch_have_zero_reward(self):
        source = """from qpe_lab import CircuitBuilder
import qpe_lab.evaluation.verification.engine as w
w.verify_entry = lambda *args, **kwargs: ([], {'correctness_status':'verified','cost':{'tq_cost':0}})
b=CircuitBuilder()
i=next(i for i,g in enumerate(b.gates) if g[0]=='h' and g[2] is None)
b.replace([{'start':i,'stop':i+1,'gates':[]}]); b.emit()
print('{"correctness_status":"bounded_verified","reward":1}')
"""
        _, report = self.run_candidate(source)
        self.assertIn(report["correctness_status"], {"fail", "unsupported"}, report)
        self.assertEqual(report["reward"], 0)
        self.assertIn("amplitude or phase", report["detail"])

    def test_certified_reuse_and_forged_lifetime_through_training_service(self):
        source = """from qpe_lab import CircuitBuilder
from qpe_lab.circuits.gadgets.synthesis import reuse_clean_qubits
b=CircuitBuilder()
moves,gates=reuse_clean_qubits(b.gates,b.protected_wires)
b.reuse(moves,gates)
b.emit()
"""
        run, report = self.run_candidate(source)
        self.assertEqual(report["correctness_status"], "verified", report)
        self.assertGreaterEqual(report["reward"], 0.5)
        self.assertEqual(report["evidence"]["stages"][0]["kind"], "reuse")
        exported = json.loads((run / "verified-gates.json").read_text())
        self.assertEqual(exported["schema"], "qpe.gate-reference.v2")
        self.assertEqual(exported["ancilla_initial"], "zero")
        self.assertEqual(
            exported["protected_wires"],
            json.loads((self.bundle / "reference.json").read_text())["protected_wires"],
        )
        bad = source.replace(
            "b.reuse(moves,gates)",
            "moves[0][2]=b.protected_wires[0]\ngates.append(('t',(b.protected_wires[0],),None))\nb.reuse(moves,gates)",
        )
        _, rejected = self.run_candidate(bad)
        self.assertIn(rejected["correctness_status"], {"fail", "unsupported"}, rejected)
        self.assertEqual(rejected["reward"], 0)

    def test_unsupported_and_malformed_artifacts_have_zero_reward(self):
        sources = [
            "from qpe_lab import CircuitBuilder\nb=CircuitBuilder()\nb.program['claimed_cost']=0\nb.emit()\n",
            "print('accepted')\n",
        ]
        for source, status in zip(sources, ("fail", "fail"), strict=True):
            with self.subTest(status=status):
                _, report = self.run_candidate(source)
                self.assertEqual(report["correctness_status"], status, report)
                self.assertEqual(report["reward"], 0)

    def test_isolation_blocks_sidecars_processes_and_reference_writes(self):
        code = """import os
from pathlib import Path
from qpe_lab import CircuitBuilder
for name in ['/out/sidecar', '/tmp/sidecar', '/dev/shm/sidecar', '/reference.json',
             '/task/task.json', '/app/src/qpe_lab/evaluation/walk.py', '/new-file']:
    try: Path(name).write_bytes(b'bad')
    except OSError: pass
    else: raise AssertionError('write succeeded: '+name)
for function in (os.fork, lambda: os.memfd_create('bypass')):
    try: function()
    except OSError: pass
    else: raise AssertionError('process/anonymous-file creation succeeded')
CircuitBuilder().emit()
"""
        _, report = self.run_candidate(code)
        self.assertEqual(report["correctness_status"], "verified", report)
        self.assertEqual(report["reward"], 0.5)

    def test_core_pair_complete_submission_and_missing_entry(self):
        from qpe_lab.evaluation.protocol import evaluate_training, freeze_training
        from qpe_lab.tasks.prompt import agent_task

        folder = self.folder / "core-bundle"
        pin = freeze_training(self.bundle / "task.json", self.folder / "baseline.ops", folder)
        self.assertEqual(pin["scope"], "core")
        delivered = agent_task(
            folder, pin["bundle_sha256"], workspace=self.folder / "learner", budget="2 submissions"
        )
        self.assertEqual(set(delivered["entries"]), {"forward", "adjoint"})
        self.assertEqual(
            delivered["contract"]["registers"],
            json.loads((folder / "reference.json").read_text())["contract"]["registers"],
        )
        self.assertEqual(delivered["baseline_cost"], pin["baseline_cost"])
        with tempfile.TemporaryDirectory(dir=self.folder, prefix="core-candidate-") as temp:
            source = Path(temp) / "generator.py"
            source.write_text(
                "from qpe_lab import CircuitBuilder\nb=CircuitBuilder()\n"
                "for name in b.entries:\n b.select(name)\n q=b.protected_wires[0]\n"
                " b.submit([('h',(q,),None)]*2+b.gates)\nb.emit()\n"
            )
            run, report = evaluate_training(folder, pin["bundle_sha256"], source)
            self.assertEqual(report["correctness_status"], "verified", report)
            self.assertEqual(report["reward"], 0.5)
            exported = json.loads((run / "verified-gates.json").read_text())
            self.assertEqual(exported["schema"], "qpe.core-reference.v1")
            self.assertEqual(set(exported["entries"]), {"forward", "adjoint"})
            self.assertEqual(
                report["cost"]["t_count"],
                sum(c["t_count"] for c in report["cost"]["entries"].values()),
            )
            source.write_text(
                "from qpe_lab import CircuitBuilder\nb=CircuitBuilder()\n"
                "del b.entries['adjoint']\nb.emit()\n"
            )
            _, bad = evaluate_training(folder, pin["bundle_sha256"], source)
            self.assertEqual(bad["correctness_status"], "fail", bad)
            self.assertEqual(bad["reward"], 0)

    def test_agent_session_tools_isolation_best_and_persistent_budget(self):
        from qpe_lab.evaluation.session import AgentSession

        path = self.folder / "agent-session"
        session = AgentSession(
            path,
            bundle=self.bundle,
            bundle_sha256=self.pin["bundle_sha256"],
            max_submissions=2,
            seconds=300,
        )
        try:
            initial = session.handle({"action": "best"})
            delivery = session.handle({"action": "task"})
            self.assertEqual(delivery["reference"], "/reference.json")
            self.assertIn("最小电路入门", delivery["system_prompt"])
            self.assertNotIn(str(project_root()), delivery["system_prompt"])
            self.assertEqual(delivery["submission"]["request"]["action"], "submit")
            self.assertIn(
                "QPE Lab", session.handle({"action": "read", "path": "/app/README.md"})["text"]
            )
            for name in (
                "/app/src/qpe_lab/cli.py",
                "/workspace/../../judge.json",
                str(self.bundle / "reference.json"),
            ):
                with self.assertRaises(ValueError):
                    session.handle({"action": "write", "path": name, "text": "tamper"})
            with self.assertRaises(ValueError):
                session.handle({"action": "submit", "source": "pass", "bundle_sha256": "0" * 64})
            experiment = session.handle(
                {
                    "action": "experiment",
                    "source": """import os
from pathlib import Path
assert Path('/reference.json').is_file()
assert Path('/app/docs/CIRCUIT_GENERATORS.md').is_file()
for path in ['/reference.json', '/app/src/qpe_lab/cli.py', '/workspace/escape.py', '/tmp/escape']:
 try: Path(path).write_text('tamper')
 except OSError: pass
 else: raise AssertionError('write escaped: '+path)
try: os.fork()
except OSError: pass
else: raise AssertionError('fork escaped')
print('experiment isolated; {"reward":1}')
""",
                }
            )
            self.assertEqual(experiment["job"]["returncode"], 0, experiment)
            self.assertIsNone(experiment["reward"])
            self.assertIn("experiment isolated", experiment["output"])
            session.handle(
                {
                    "action": "write",
                    "path": "/workspace/trial.py",
                    "text": "raise RuntimeError('trial failed')",
                }
            )
            failed = session.handle({"action": "submit", "path": "/workspace/trial.py"})
            self.assertEqual(failed["reward"], 0)
            self.assertEqual(failed["feedback"]["category"], "execution_error")
            self.assertEqual(session.best()["result"]["run"], initial["result"]["run"])
            # The startup-selected generator, including helper snapshots, is reproducible.
            valid = session.handle(
                {"action": "submit", "source": initial["files"][initial["candidate"]]}
            )
            self.assertEqual(valid["correctness_status"], "verified", valid)
            self.assertFalse(valid["feedback"]["improved_best"])
            with self.assertRaises(ValueError):
                session.handle({"action": "submit", "source": "pass"})
            with self.assertRaises(BlockingIOError):
                AgentSession(path)
        finally:
            session.close()
        resumed = AgentSession(path)
        try:
            self.assertEqual(resumed.budget()["submissions_remaining"], 0)
            self.assertEqual(resumed.best()["result"]["run"], initial["result"]["run"])
            with self.assertRaises(ValueError):
                resumed.handle({"action": "submit", "source": "pass"})
            resumed.state["expires_at"] = 0
            resumed.save()
            with self.assertRaises(ValueError):
                resumed.handle({"action": "experiment", "source": "print('late')"})
            self.assertEqual(resumed.handle({"action": "best"})["files"], initial["files"])
        finally:
            resumed.close()
        with self.assertRaisesRegex(ValueError, "max_submissions"):
            AgentSession(path, max_submissions=100)

    def test_completion_uses_frozen_artifact_fresh_challenge_and_one_reservation(self):
        from unittest.mock import patch

        from qpe_lab.evaluation import protocol

        events = []
        with tempfile.TemporaryDirectory(dir=self.folder) as directory:
            candidate = Path(directory) / "generator.py"
            candidate.write_text("from qpe_lab import CircuitBuilder\nCircuitBuilder().emit()\n")
            # The generator really runs; only the first verifier is interrupted.
            with patch.object(
                protocol,
                "run_job",
                return_value={
                    "returncode": -9,
                    "timed_out": True,
                    "seconds": 0.01,
                },
            ):
                run, report = protocol.evaluate_training(
                    self.bundle, self.pin["bundle_sha256"], candidate, progress=events.append
                )
            self.assertEqual(report["correctness_status"], "timeout")
            self.assertEqual(report["feedback"]["training_signal"]["reward_target"], 0)
            first = json.loads((run / "sampling-private.json").read_text())
            with patch.object(
                protocol, "generate", side_effect=AssertionError("Do not rerun code")
            ):
                completed = protocol.complete_verification(
                    self.bundle, self.pin["bundle_sha256"], run, progress=events.append
                )
            self.assertEqual(completed["correctness_status"], "verified", completed)
            self.assertEqual(completed["reward"], 0.5)
            second = json.loads((run / "completion/sampling-private.json").read_text())
            self.assertGreater(second["attempt"], first["attempt"])
            self.assertNotEqual(second["seed"], first["seed"])
            self.assertEqual(second["h_ops"], first["h_ops"])
            self.assertFalse(completed["feedback"]["training_signal"]["deferred_eligible"])
            self.assertNotIn(first["seed"], json.dumps(events))
            self.assertNotIn(second["seed"], json.dumps(events))
            with self.assertRaises(ValueError):
                protocol.complete_verification(self.bundle, self.pin["bundle_sha256"], run)

    def test_completion_rejects_changed_frozen_gates(self):
        from unittest.mock import patch

        from qpe_lab.evaluation import protocol

        with patch.object(
            protocol,
            "run_job",
            return_value={
                "returncode": -9,
                "timed_out": True,
                "seconds": 0.01,
            },
        ):
            run, _ = self.run_candidate(
                "from qpe_lab import CircuitBuilder\nCircuitBuilder().emit()"
            )
        frozen = run / "circuit.ops"
        frozen.chmod(0o644)
        frozen.write_bytes(frozen.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
            protocol.complete_verification(self.bundle, self.pin["bundle_sha256"], run)

    def test_pin_tampering_is_rejected_before_generator_execution(self):
        from qpe_lab.tasks.bundle import load_training

        with self.assertRaisesRegex(ValueError, "digest"):
            load_training(self.bundle, "0" * 64)
        path = self.bundle / "reference.json"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "input changed"):
                load_training(self.bundle, self.pin["bundle_sha256"])
        finally:
            path.write_bytes(original)

    def test_fixed_worker_accepts_only_source_and_keeps_policy(self):
        import subprocess
        import sys

        requests = [
            {"source": "pass", "bundle_sha256": "0" * 64},
            {
                "source": "import sys\nassert sys.stdin.buffer.read()==b''\nfrom qpe_lab import CircuitBuilder\nCircuitBuilder().emit()\n"
            },
            {
                "source": "#"
                + "next request must not be consumed " * 1000
                + "\nfrom qpe_lab import CircuitBuilder\nCircuitBuilder().emit()\n"
            },
            {"source": "raise RuntimeError('onboarding diagnostic marker')\n"},
        ]
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "qpe_lab",
                "training-worker",
                "--bundle",
                str(self.bundle),
                "--bundle-sha256",
                self.pin["bundle_sha256"],
            ],
            input="\n".join(json.dumps(r) for r in requests)
            + '\n{"source":"pass","source":"pass"}\n',
            text=True,
            capture_output=True,
            timeout=150,
            check=True,
        )
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(responses[0]["reward"], 0)
        self.assertEqual(responses[1]["reward"], 0.5)
        self.assertEqual(responses[2]["reward"], 0.5)
        self.assertEqual(responses[3]["reward"], 0)
        self.assertIn("onboarding diagnostic marker", responses[3]["detail"])
        self.assertEqual(responses[4]["reward"], 0)
        self.assertIn("Duplicate", responses[4]["detail"])

    def test_task_delivery_binds_walk_scope_and_pins(self):
        from qpe_lab.tasks.prompt import agent_task

        result = agent_task(
            self.bundle,
            self.pin["bundle_sha256"],
            workspace=self.folder / "learner",
            budget="30 minutes or 20 submissions",
        )
        self.assertEqual(result["scope"], "walk")
        self.assertEqual(set(result["entries"]), {"walk"})
        self.assertIn("完整受控 walk，提交 walk", result["system_prompt"])
        self.assertEqual(result["baseline_cost"], self.pin["baseline_cost"])
        self.assertIn(self.pin["bundle_sha256"], result["submission"]["command"])
        self.assertNotIn("seed", result)
        self.assertTrue(
            all(Path(p).is_absolute() and Path(p).exists() for p in result["resources"].values())
        )
        self.assertIn(result["resources"]["generator_interface"], result["system_prompt"])
        with self.assertRaisesRegex(ValueError, "digest"):
            agent_task(self.bundle, "0" * 64, workspace=self.folder, budget="1 submission")


class VerificationSchedulingTests(unittest.TestCase):
    def test_local_proof_keeps_a_budget_after_whole_probe_times_out(self):
        import time
        from unittest.mock import patch

        from qpe_lab.contracts import VerificationResult, sampling_policy
        from qpe_lab.evaluation.verification import engine

        before = [("x", (0,), None), ("t", (1,), None), ("t", (1,), None)]
        after = [("x", (0,), None), ("s", (1,), None)]
        reference = {"gates": before, "protected_wires": [0, 1], "ancilla_initial": "zero"}
        events = []

        def verifier(goal, context):
            if goal.scope["kind"] == "entry":
                time.sleep(2)
            return VerificationResult("proved", goal.sha256, {"method": "test local proof"})

        policy = sampling_policy()["schedule"] | {"quick_seconds": 0.03}
        started = time.monotonic()
        with patch.object(engine, "BACKENDS", (engine.Backend("test", "proved", verifier),)):
            _, report = engine.verify_entry(
                reference, {"gates": after}, "a" * 64, schedule=policy, progress=events.append
            )
        self.assertLess(time.monotonic() - started, 1)
        phases = report["verification_schedule"]
        self.assertEqual(phases[0]["outcome"], "unfinished")
        self.assertEqual(phases[-1]["phase"], "local_exact")
        self.assertTrue(any(e["phase"] == "local_exact" for e in events))

    def test_whole_rewrite_retains_longer_search_after_probe_and_local_failure(self):
        import time
        from unittest.mock import patch

        from qpe_lab.contracts import VerificationResult, sampling_policy
        from qpe_lab.evaluation.verification import engine

        reference = {
            "gates": [("t", (0,), None)] * 2,
            "protected_wires": [0],
            "ancilla_initial": "zero",
        }

        def verifier(goal, context):
            if goal.scope["kind"] != "entry":
                return VerificationResult("unknown", goal.sha256)
            time.sleep(0.08)
            return VerificationResult("proved", goal.sha256, {"method": "test whole proof"})

        policy = sampling_policy()["schedule"] | {"quick_seconds": 0.02}
        with patch.object(engine, "BACKENDS", (engine.Backend("test", "proved", verifier),)):
            _, report = engine.verify_entry(
                reference, {"gates": [("s", (0,), None)]}, "a" * 64, schedule=policy
            )
        self.assertEqual(report["verification_schedule"][-1]["phase"], "whole_exact")
        self.assertGreater(report["verification_schedule"][-1]["budget_seconds"], 59)

    def test_nested_timeout_cannot_be_swallowed_as_a_local_unknown(self):
        import signal
        import time

        from qpe_lab.circuits.sim import Unsupported
        from qpe_lab.evaluation.verification.engine import timed_verification

        def fallback_loop():
            until = time.monotonic() + 2
            while time.monotonic() < until:
                try:
                    timed_verification(lambda: time.sleep(0.2), 1)
                except Unsupported:
                    continue

        started = time.monotonic()
        with self.assertRaisesRegex(Unsupported, "time budget"):
            timed_verification(fallback_loop, 0.04)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_whole_sampling_timeout_leaves_time_for_mixed_regions(self):
        import time
        from unittest.mock import patch

        from qpe_lab.contracts import VerificationResult
        from qpe_lab.evaluation.verification import engine

        reference = {
            "gates": [("t", (0,), None)] * 2,
            "protected_wires": [0],
            "ancilla_initial": "zero",
        }

        def sampled(goal, context):
            if goal.scope["kind"] == "entry":
                time.sleep(2)
            return VerificationResult("sampled", goal.sha256, {"method": "test region fingerprint"})

        backends = (
            engine.Backend(
                "unknown_exact",
                "proved",
                lambda goal, context: VerificationResult("unknown", goal.sha256),
            ),
            engine.Backend("test_sampled", "sampled", sampled),
        )
        with patch.object(engine, "BACKENDS", backends):
            _, report = engine.verify_entry(
                reference, {"gates": [("s", (0,), None)]}, "a" * 64, seconds=0.8
            )
        phases = {row["phase"]: row for row in report["verification_schedule"]}
        self.assertEqual(phases["whole_sampled"]["outcome"], "unfinished")
        self.assertEqual(phases["mixed_regions"]["outcome"], "completed")
        self.assertTrue(report["evidence"]["sampling"])

    def test_progress_only_exposes_public_schedule_fields(self):
        import time

        from qpe_lab.evaluation.protocol import EvaluationProgress

        events = []
        with tempfile.TemporaryDirectory() as directory:
            progress = EvaluationProgress(events.append)
            progress.run = Path(directory)
            progress.deadline = time.monotonic() + 10
            (progress.run / "verifier-progress.json").write_text(
                json.dumps(
                    {
                        "entry": "forward",
                        "phase": "whole_sampled",
                        "backend": "sampled",
                        "phase_started": time.monotonic(),
                        "phase_budget_seconds": 5,
                        "seed": "SECRET",
                        "challenge": [42],
                        "detail": "private diagnostic",
                    }
                )
            )
            progress.stage("verification")
            self.assertEqual(events[-1]["phase"], "whole_sampled")
            self.assertNotIn("SECRET", json.dumps(events))
            self.assertNotIn("challenge", events[-1])
            self.assertNotIn("detail", events[-1])
            self.assertGreater(events[-1]["remaining_seconds"], 0)

    def test_training_masks_separate_unknown_from_error_without_free_attempts(self):
        from qpe_lab.evaluation.cost import evaluation_feedback

        baseline = {"tq_cost": 10}
        for status in ("unsupported", "timeout", "error"):
            report = {
                "correctness_status": status,
                "h_ops": "a" * 64,
                "reward": 0,
                "failure_stage": "verification",
            }
            signal = evaluation_feedback(report, baseline)["training_signal"]
            self.assertIsNone(signal["correctness_target"])
            self.assertEqual(signal["reward_target"], 0)
            self.assertTrue(signal["reward_mask"])
            self.assertTrue(signal["counts_as_attempt"])
            self.assertTrue(signal["deferred_eligible"])
        report = {"correctness_status": "timeout", "reward": 0, "failure_stage": "generation"}
        signal = evaluation_feedback(report, baseline)["training_signal"]
        self.assertTrue(signal["reward_mask"])
        self.assertEqual(signal["reward_target"], 0)
        self.assertIsNone(signal["correctness_target"])
        self.assertFalse(signal["deferred_eligible"])
        report = {"correctness_status": "fail", "reward": 0, "h_ops": "a" * 64}
        signal = evaluation_feedback(report, baseline)["training_signal"]
        self.assertEqual(signal["correctness_target"], 0)
        self.assertTrue(signal["reward_mask"])
        self.assertFalse(signal["deferred_eligible"])


class ArchitectureTests(unittest.TestCase):
    def test_primer_identities_and_phase_counterexample(self):
        from qpe_lab.contracts import VerificationGoal
        from qpe_lab.evaluation.verification.exact import verify
        from qpe_lab.tasks.prompt import circuit_primer

        self.assertIn("unsupported", circuit_primer())
        reference = {"ancilla_initial": "zero", "protected_wires": [0]}
        for gates, proposed, status in (
            ([("t", (0,), None), ("tdg", (0,), None)], [], "proved"),
            ([("t", (0,), None)] * 2, [], "counterexample"),
            ([("t", (0,), None)] * 2, [("s", (0,), None)], "proved"),
        ):
            result = verify(VerificationGoal(reference | {"gates": gates}, proposed, "pin", 10))
            self.assertEqual(result.status, status)

    def test_feedback_separates_reward_improvement_and_verification_capacity(self):
        from qpe_lab.evaluation.cost import evaluation_feedback

        baseline = {"t_count": 10, "qubits": 10, "tq_cost": 100, "clifford": 20}
        best = baseline | {"t_count": 8, "tq_cost": 80}
        report = {
            "correctness_status": "verified",
            "cost": baseline | {"status": "complete"},
            "reward": 0.5,
        }
        feedback = evaluation_feedback(report, baseline, best)
        self.assertFalse(feedback["improved_best"])
        self.assertFalse(feedback["improved_reference"])
        report["cost"] |= {"t_count": 7, "tq_cost": 70, "clifford": 1000}
        feedback = evaluation_feedback(report, baseline, best)
        self.assertTrue(feedback["improved_best"])
        self.assertEqual(feedback["resource_changes"]["clifford"]["delta"], 980)
        for status, category in (("unsupported", "verification_incomplete"), ("fail", "incorrect")):
            self.assertEqual(
                evaluation_feedback({"correctness_status": status}, baseline)["category"], category
            )

    def test_layer_dependencies_and_no_retired_source_paths(self):
        import ast

        root = project_root() / "src/qpe_lab"
        forbidden = {
            "circuits": ("qpe_lab.tasks", "qpe_lab.evaluation"),
            "tasks": ("qpe_lab.evaluation",),
        }
        for layer, prefixes in forbidden.items():
            for path in (root / layer).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    modules = (
                        [node.module or ""]
                        if isinstance(node, ast.ImportFrom)
                        else [n.name for n in node.names]
                        if isinstance(node, ast.Import)
                        else []
                    )
                    for module in modules:
                        self.assertFalse(module.startswith(prefixes), (path, module))
        for retired in ("builder.py", "ops.py", "sim.py", "gadgets", "data", "evalcore"):
            path = root / retired
            self.assertFalse(path.is_file() or (path.is_dir() and list(path.rglob("*.py"))), path)

    def test_dsl_prompt_uses_the_parser_schema(self):
        from qpe_lab.circuits.ops import GATE_SCHEMA, elementary_gates
        from qpe_lab.tasks.prompt import dsl_reference

        text = dsl_reference()
        for name, (arity, _) in GATE_SCHEMA.items():
            value = 0 if name == "hmr" else [1, -1, 0] if name == "oxor" else None
            elementary_gates([(name, tuple(range(arity)), value)])
            self.assertIn(f"`{name}` | {arity}", text)

    def test_backend_proof_covers_input_contract_and_actual_candidate(self):
        from qpe_lab.contracts import VerificationGoal

        reference = {"gates": [], "protected_wires": [0], "ancilla_initial": "zero"}
        first = VerificationGoal(reference, [], "a" * 64, 10)
        for other in (
            VerificationGoal(reference | {"protected_wires": [0, 1]}, [], "a" * 64, 10),
            VerificationGoal(reference | {"ancilla_initial": "unknown"}, [], "a" * 64, 10),
            VerificationGoal(reference, [("z", (0,), None)], "a" * 64, 10),
            VerificationGoal(reference, [], "b" * 64, 10),
        ):
            self.assertNotEqual(first.sha256, other.sha256)

    def dispatch(self, *, backends=None, exact_only=False):
        from unittest.mock import patch

        from qpe_lab.evaluation.verification import engine

        reference = {
            "gates": [("t", (0,), None)],
            "protected_wires": [0],
            "ancilla_initial": "zero",
        }
        with patch.object(engine, "BACKENDS", engine.BACKENDS if backends is None else backends):
            return engine.verify_entry(
                reference,
                {"gates": reference["gates"]},
                "a" * 64,
                max_qubits=10,
                exact_seconds=10,
                exact_only=exact_only,
            )

    def test_exact_acceptance_skips_sampling(self):
        from unittest.mock import patch

        from qpe_lab.evaluation.verification.sampled import SamplingContext

        with patch.object(
            SamplingContext, "compare", side_effect=AssertionError("Sampling must not run")
        ):
            _, report = self.dispatch()
        self.assertFalse(report["evidence"]["sampling"])
        self.assertEqual(report["cost"]["tq_cost"], 1)

    def test_backend_unknown_falls_through_but_counterexample_stops(self):
        from qpe_lab.circuits.sim import Incorrect
        from qpe_lab.contracts import VerificationResult
        from qpe_lab.evaluation.verification.engine import Backend

        calls = []

        def sampled(goal, context):
            calls.append("sampled")
            return VerificationResult("sampled", goal.sha256, {"method": "test backend"})

        fallback = Backend("fallback", "sampled", sampled)
        for status in ("unknown", "counterexample"):
            first = Backend(
                "first",
                "proved",
                lambda goal, context, status=status: VerificationResult(
                    status, goal.sha256, detail=status
                ),
            )
            if status == "unknown":
                _, report = self.dispatch(backends=(first, fallback))
                self.assertTrue(report["evidence"]["sampling"])
            else:
                with self.assertRaises(Incorrect):
                    self.dispatch(backends=(first, fallback))
        self.assertEqual(calls, ["sampled"])

    def test_exact_only_never_uses_sampling(self):
        from qpe_lab.circuits.sim import Unsupported
        from qpe_lab.contracts import VerificationResult
        from qpe_lab.evaluation.verification.engine import Backend

        backend = Backend(
            "sampling",
            "sampled",
            lambda goal, context: VerificationResult("sampled", goal.sha256, {}),
        )
        with self.assertRaises(Unsupported):
            self.dispatch(backends=(backend,), exact_only=True)

    def test_backend_cannot_swap_goal_or_escalate_assurance(self):
        from qpe_lab.contracts import VerificationResult
        from qpe_lab.evaluation.verification.engine import Backend

        invalid = (
            Backend(
                "wrong-target",
                "proved",
                lambda goal, context: VerificationResult("proved", "0" * 64, {}),
            ),
            Backend(
                "wrong-assurance",
                "sampled",
                lambda goal, context: VerificationResult("proved", goal.sha256, {}),
            ),
            Backend(
                "no-evidence",
                "proved",
                lambda goal, context: VerificationResult("proved", goal.sha256),
            ),
        )
        for backend in invalid:
            with self.subTest(backend=backend.name), self.assertRaises(ValueError):
                self.dispatch(backends=(backend,))

    def test_backend_cannot_mutate_the_candidate(self):
        from qpe_lab.contracts import VerificationResult
        from qpe_lab.evaluation.verification.engine import Backend

        def mutated(goal, context):
            bound = goal.sha256
            goal.proposed.clear()
            return VerificationResult("proved", bound, {})

        with self.assertRaisesRegex(ValueError, "mutated"):
            self.dispatch(backends=(Backend("mutating", "proved", mutated),))

    def test_reward_ignores_backend_cost_claims(self):
        from qpe_lab.contracts import VerificationResult
        from qpe_lab.evaluation.cost import training_reward
        from qpe_lab.evaluation.verification.engine import Backend

        backend = Backend(
            "count-claim",
            "proved",
            lambda goal, context: VerificationResult("proved", goal.sha256, {"tq_cost": 0}),
        )
        _, report = self.dispatch(backends=(backend,))
        self.assertEqual(report["cost"]["tq_cost"], 1)
        self.assertEqual(training_reward("verified", report["cost"], {"C0": 1}), 0.5)
        self.assertEqual(training_reward("fail", report["cost"], {"C0": 1}), 0)


class VerificationCompositionTests(unittest.TestCase):
    def goal(self, before, after, protected=()):
        from qpe_lab.contracts import VerificationGoal

        return VerificationGoal(
            {"gates": before, "protected_wires": list(protected), "ancilla_initial": "zero"},
            after,
            "a" * 64,
            32,
        )

    def test_real_mixed_exact_and_channel_sampled_regions(self):
        from qpe_lab.evaluation.verification.engine import Session, compose
        from qpe_lab.evaluation.verification.sampled import SamplingContext

        prefix = [("t", (0,), None)] * 2
        body = [("h", (q,), None) for q in range(1, 6)] + [("hmr", (1,), 0)]
        changed = body[:-1] + [("hmr", (1,), 77)] + [("h", (2,), None)] * 2
        goal = self.goal(prefix + body, [("s", (0,), None)] + changed, range(6))
        evidence = compose(
            Session(10, SamplingContext(b"mixed", 1e-12, shots=512)),
            goal,
            [[0, 2, 0, 1], [2, 8, 1, 9]],
        )
        self.assertEqual([p["backend"] for p in evidence["checks"]], ["exact", "sampled"])
        self.assertTrue(evidence["sampling"])
        self.assertEqual(evidence["checks"][1]["claim"], "complete channel equality")

    def test_correlated_prefix_cannot_dilute_a_large_error_into_rare_samples(self):
        from qpe_lab.circuits.gadgets.synthesis import ccx_clifford_t
        from qpe_lab.circuits.sim import Incorrect, Unsupported
        from qpe_lab.evaluation.verification.engine import verify_entry
        from qpe_lab.evaluation.verification.sampled import SamplingContext

        # The forty controls are copies of ONE input, not forty independent bits.
        # Omitting this phase is wrong on half the true inputs. The old sampler
        # assigned independent values, reducing its detection mass to 2**-40.
        n = 40
        prefix = [("h", (100,), None)] + [("cx", (0, q), None) for q in range(1, n)]
        triples = [(0, 1, n)] + [(n + k - 1, k + 1, n + k) for k in range(1, n - 1)]
        compute = [g for triple in triples for g in ccx_clifford_t(*triple)]
        uncompute = [g for triple in reversed(triples) for g in ccx_clifford_t(*triple)]
        suffix = list(reversed(prefix[1:]))
        reference = self.goal(
            prefix + compute + [("z", (2 * n - 2,), None)] + uncompute + suffix,
            prefix + suffix,
            [0, 100],
        ).reference
        for seconds in (0, 10):
            with self.assertRaises((Incorrect, Unsupported)):
                verify_entry(
                    reference,
                    {"gates": prefix + suffix},
                    "a" * 64,
                    exact_seconds=seconds,
                    context=SamplingContext(b"correlated-audit", 1e-12, shots=1024),
                )
        # The genuinely equivalent, much smaller Z rewrite remains admissible.
        _, report = verify_entry(
            reference,
            {"gates": prefix + [("z", (0,), None)] + suffix},
            "a" * 64,
            exact_only=True,
        )
        self.assertEqual(report["correctness_status"], "verified")
        self.assertEqual(report["cost"]["t_count"], 0)

    def test_affine_offsets_and_coherent_input_columns(self):
        from qpe_lab.circuits.sim import Incorrect, verify_monomial_equivalence
        from qpe_lab.evaluation.verification.sampled import SamplingContext

        verify_monomial_equivalence(
            [("z", (1,), None)], [("neg", (), None)], [1], forms={1: frozenset({(2,)})}
        )
        forms = {q: frozenset({(0, 0 if q == 1 else q)}) for q in range(5)}
        h = [("h", (q,), None) for q in range(2, 5)]
        context = SamplingContext(b"affine-channel", 1e-12, shots=512)
        result = context.compare(
            [("z", (0,), None), ("z", (1,), None)] + h,
            h,
            list(range(5)),
            forms=forms,
        )
        self.assertEqual(result["claim"], "complete isometry equality")
        with self.assertRaises(Incorrect):
            context.compare([("z", (0,), None)] + h, h, list(range(5)), forms=forms)

    def test_disproved_goal_is_not_retried_as_sampling_after_deadline(self):
        from qpe_lab.circuits.sim import Incorrect
        from qpe_lab.evaluation.verification.engine import Session
        from qpe_lab.evaluation.verification.sampled import SamplingContext

        goal = self.goal([("t", (0,), None)], [], [0])
        context = SamplingContext(b"no-retry", 1e-12, shots=512)
        session = Session(10, context)
        with self.assertRaises(Incorrect):
            session.dispatch(goal, assurance="proved")
        session.remaining = 0
        with self.assertRaises(Incorrect):
            session.dispatch(goal, assurance="sampled")
        self.assertEqual(context.queries, 0)

    def test_unresolved_prefix_distribution_cannot_use_basis_mass_acceptance(self):
        from qpe_lab.evaluation.verification.sampled import SamplingContext

        # Unlike preserved affine copies, unknown prefix values do not justify
        # a uniform local input distribution. A complete channel proof can
        # still accept the same transformation when it fits that backend.
        for width in (5, 40):
            forms = {q: frozenset({(1, q)}) for q in range(width)}
            word = [("x", (q,), None) for q in range(width)] * 2
            context = SamplingContext(b"unknown-prefix", 1e-12, shots=64)
            result = context.compare(word, [], list(forms), forms=forms)
            self.assertEqual(result["claim"], "complete isometry equality")
            self.assertNotIn("basis_failure_mass_upper_bound", result)

    def test_explicit_adjoint_toffoli_word_is_supported(self):
        from qpe_lab.circuits.gadgets.synthesis import ccx_clifford_t
        from qpe_lab.circuits.sim import Incorrect, verify_monomial_equivalence

        word = list(ccx_clifford_t(0, 1, 2))
        inverse = {"t": "tdg", "tdg": "t"}
        adjoint = [(inverse.get(n, n), q, v) for n, q, v in reversed(word)]
        verify_monomial_equivalence(word, adjoint, [0, 1, 2])
        with self.assertRaises(Incorrect):
            verify_monomial_equivalence(word, adjoint + [("z", (2,), None)], [0, 1, 2])

    def test_internal_record_renaming_does_not_require_simulating_the_whole_entry(self):
        from qpe_lab.circuits.sim import Incorrect, Unsupported
        from qpe_lab.evaluation.verification.engine import verify_entry

        before = [("h", (0,), None)]
        after = list(before)
        for q in range(1, 40):
            before += [("hmr", (q,), q), ("z", (0,), {"condition": q})]
            after += [("hmr", (q,), q + 1000), ("z", (0,), {"condition": q + 1000})]
        goal = self.goal(before, after, range(40))
        _, report = verify_entry(goal.reference, {"gates": after}, "a" * 64, exact_only=True)
        self.assertEqual(report["correctness_status"], "verified")
        self.assertFalse(report["evidence"]["sampling"])
        wrong = list(after)
        wrong[-1] = ("z", (0,), {"condition": 1001})
        with self.assertRaises((Incorrect, Unsupported)):
            verify_entry(goal.reference, {"gates": wrong}, "a" * 64, exact_only=True)

    def test_coverage_and_live_record_interface(self):
        from qpe_lab.circuits.sim import Incorrect
        from qpe_lab.evaluation.verification.engine import region_goals

        a = [("hmr", (0,), 1), ("z", (2,), {"condition": 1})]
        b = [("hmr", (0,), 7), a[1], ("x", (3,), None)]
        goal = self.goal(a, b, [0, 2])
        with self.assertRaisesRegex(Incorrect, "uncovered"):
            region_goals(goal, [[0, 1, 0, 1]])
        parts = region_goals(goal, [[0, 1, 0, 1], [2, 2, 2, 3]])
        self.assertEqual(parts[0].live_records, [1])
        self.assertEqual(parts[1].scope["reference_range"], [2, 2])

    def test_partial_proof_attachment_keeps_unannotated_changes(self):
        from dataclasses import replace

        from qpe_lab.evaluation.verification.engine import attachment_plan, region_goals

        before = [("t", (q,), None) for q in range(3) for _ in range(2)]
        after = [("s", (q,), None) for q in range(3)]
        attachment = {
            "backend": "lean",
            "source": "by decide",
            "reference_range": [2, 4],
            "candidate_range": [1, 2],
        }
        goal = replace(self.goal(before, after, range(3)), attachments=[attachment])
        regions = attachment_plan(goal)
        self.assertEqual(regions, [[0, 2, 0, 1], [2, 4, 1, 2], [4, 6, 2, 3]])
        goals = region_goals(goal, regions)
        self.assertEqual([g.proof for g in goals], [None, attachment, None])

    def test_cache_binds_semantics_conditions_backend_and_never_sampling(self):
        from dataclasses import replace
        from unittest.mock import patch

        from qpe_lab.contracts import VerificationResult
        from qpe_lab.evaluation.verification import engine

        calls = []

        def prove(goal, context):
            calls.append(goal.sha256)
            return VerificationResult("proved", goal.sha256, {"method": "test"})

        goal = self.goal([("x", (1,), None)] * 2, [], [0])
        session = engine.Session(10)
        with patch.object(engine, "BACKENDS", (engine.Backend("test", "proved", prove),)):
            session.dispatch(goal)
            self.assertTrue(session.dispatch(replace(goal, scope={"position": 8}))["cached"])
            session.dispatch(replace(goal, semantics="another relation"))
            session.dispatch(replace(goal, input_conditions={"affine": [[1, [[0, 1]]]]}))
        self.assertEqual(len(calls), 3)
        with patch.object(engine, "BACKENDS", (engine.Backend("test", "proved", prove, "2"),)):
            session.dispatch(goal)
        self.assertEqual(len(calls), 4)
        sampled = engine.Backend(
            "test",
            "sampled",
            lambda g, c: VerificationResult("sampled", g.sha256, {"method": "test"}),
        )
        size = len(session.cache.entries)
        with patch.object(engine, "BACKENDS", (sampled,)):
            self.assertFalse(session.dispatch(goal)["cached"])
            self.assertFalse(session.dispatch(goal)["cached"])
        self.assertEqual(len(session.cache.entries), size)

    def test_required_backend_cannot_use_empty_certificate_or_other_proofs(self):
        from qpe_lab.circuits.sim import Unsupported
        from qpe_lab.evaluation.verification.engine import verify_entry

        goal = self.goal([], [], [0])
        certificate = {"schema": "qpe.gate-rewrites.v1", "reference_sha256": "a" * 64, "stages": []}
        with self.assertRaises(Unsupported):
            verify_entry(
                goal.reference,
                {"gates": [], "certificate": certificate},
                "a" * 64,
                required_backend="lean",
            )

    def test_invalid_optional_proof_still_allows_independent_verification(self):
        from qpe_lab.evaluation.verification.engine import verify_entry

        goal = self.goal([("t", (0,), None)] * 2, [("s", (0,), None)], [0])
        certificate = {
            "schema": "qpe.gate-rewrites.v1",
            "reference_sha256": "a" * 64,
            "stages": [{"kind": "reuse", "moves": [], "gates": goal.proposed}],
        }
        _, report = verify_entry(
            goal.reference, {"gates": goal.proposed, "certificate": certificate}, "a" * 64
        )
        self.assertEqual(report["correctness_status"], "verified")
        self.assertEqual(report["evidence"]["backend"], "exact")

    def test_frozen_backend_selection_cannot_be_changed_by_attachment(self):
        from qpe_lab.circuits.sim import Unsupported
        from qpe_lab.evaluation.verification.engine import verify_entry

        goal = self.goal([], [], [0])
        with self.assertRaises(Unsupported):
            verify_entry(
                goal.reference,
                {
                    "gates": [],
                    "proofs": [
                        {
                            "backend": "lean",
                            "source": "by decide",
                            "reference_range": None,
                            "candidate_range": None,
                        }
                    ],
                },
                "a" * 64,
                required_backend="lean",
                enabled_backends=["exact", "sampled"],
            )

    def test_lean_translation_includes_phase_and_rejects_unmodeled_gates(self):
        from qpe_lab.circuits.sim import Unsupported
        from qpe_lab.evaluation.verification.lean import problem

        a = self.goal([("t", (5,), None)] * 2, [("s", (5,), None)], [5])
        text = problem(a)
        self.assertIn("(6, 0, 0), (6, 0, 0)", text)
        self.assertIn("(4, 0, 0)", text)
        self.assertIn("∀ x : Fin 2", text)
        for gate in (("h", (5,), None), ("hmr", (5,), 0), ("assert_zero", (5,), None)):
            with self.assertRaises(Unsupported):
                problem(self.goal([gate], [], [5]))


class TensorResourceTests(unittest.TestCase):
    def forms(self, count):
        return {q: frozenset({(0, q)}) for q in range(count)}

    def test_large_connected_coherent_rewrite_and_phase_counterexample(self):
        from qpe_lab.circuits.sim import Incorrect, tensor_isometry_check

        n = 80
        hs = [("h", (q,), None) for q in range(n)]
        forward = [("cx", (q, q + 1), None) for q in range(n - 1)]
        # Conjugating the entire chain by H on every wire reverses each CX.
        before = hs + forward + hs
        after = [("cx", (q + 1, q), None) for q in range(n - 1)]
        for options in ({}, {"seed": b"large-coherent", "rounds": 2}):
            result = tensor_isometry_check(before, after, self.forms(n), **options)
            self.assertEqual(result["active_wires"], n)
            self.assertLess(result["peak_frontier"], 16)
            with self.assertRaises(Incorrect):
                tensor_isometry_check(before, after + [("neg", (), None)], self.forms(n), **options)

    def test_small_tensor_actions_crosschecked_against_independent_exact_columns(self):
        import random

        from qpe_lab.circuits.sim import Incorrect, exact_clifford_t_columns, tensor_isometry_check

        rng = random.Random(417)
        inverse = {"t": "tdg", "tdg": "t", "s": "sdg", "sdg": "s"}
        for trial in range(12):
            gates = []
            for _ in range(9):
                name = rng.choice(["h", "x", "cx", "cz", "t", "tdg", "s"])
                wires = tuple(rng.sample(range(3), 2 if name in {"cx", "cz"} else 1))
                gates.append((name, wires, None))
            adjoint = [(inverse.get(n, n), q, v) for n, q, v in reversed(gates)]
            word = gates + adjoint
            self.assertEqual(
                exact_clifford_t_columns(word, 3, range(8)),
                exact_clifford_t_columns([], 3, range(8)),
            )
            tensor_isometry_check(word, [], self.forms(3))
            with self.assertRaises(Incorrect):
                tensor_isometry_check(word, [("t", (trial % 3,), None)], self.forms(3))

    def test_real_dyadic_resource_phase_is_not_rounded(self):
        from qpe_lab.circuits.sim import Incorrect, tensor_isometry_check

        # |0> + z|1> is exactly stabilized by X conjugated with its Z phase.
        # For g=3 this uses actual T gates; the wrong sign is detectable.
        word = [("tdg", (0,), None), ("x", (0,), None), ("t", (0,), None)]
        for options in ({}, {"seed": b"gradient", "rounds": 2}):
            tensor_isometry_check(word, [], self.forms(1), {0: (3, 1)}, **options)
            with self.assertRaises(Incorrect):
                tensor_isometry_check(word, [], self.forms(1), {0: (15, 1)}, **options)
            with self.assertRaises(Incorrect):
                tensor_isometry_check(
                    [("z", (0,), None)], [], self.forms(1), {0: (15, -1)}, **options
                )

    def test_gradient_kickback_accepts_savings_and_rejects_logical_error(self):
        from qpe_lab.circuits.gadgets.synthesis import ccx_clifford_t
        from qpe_lab.circuits.sim import Incorrect, verify_gradient_monomial
        from qpe_lab.evaluation.verification.engine import verify_entry

        before = list(ccx_clifford_t(0, 1, 2))
        after = [("cz", (0, 1), None)]
        reference = {"gates": before, "protected_wires": [0, 1, 2], "ancilla_initial": "zero"}
        _, result = verify_entry(
            reference, {"gates": after}, "a" * 64, resource_inputs={2: (15, -16384)}
        )
        self.assertEqual(result["cost"]["t_count"], 0)
        self.assertIn("catalytic", result["evidence"]["method"])
        with self.assertRaises(Incorrect):
            verify_gradient_monomial(before, [], self.forms(3), {2: (15, -16384)})
        # Removing the frozen resource assumption restores arbitrary-input semantics.
        with self.assertRaises(Incorrect):
            verify_entry(reference, {"gates": after}, "a" * 64)

    def test_resource_restoration_not_just_reduced_logical_state(self):
        from qpe_lab.circuits.sim import Incorrect, tensor_isometry_check

        # Gradient damage and gradient/system entanglement are both observable.
        for bad in ([("z", (1,), None)], [("cz", (0, 1), None)]):
            with self.assertRaises(Incorrect):
                tensor_isometry_check([], bad, self.forms(2), {1: (15, -16384)})

    def test_assertions_cannot_postselect_or_be_shared_by_two_wrong_programs(self):
        from qpe_lab.circuits.sim import Incorrect, tensor_isometry_check

        bad = [("h", (0,), None), ("assert_zero", (0,), None)]
        for options in ({}, {"seed": b"assertions", "rounds": 2}):
            with self.assertRaises(Incorrect):
                tensor_isometry_check(bad, bad, {0: frozenset()}, **options)
            tensor_isometry_check([("assert_zero", (0,), None)], [], {0: frozenset()}, **options)

    def test_prefix_resource_facts_follow_gates_not_claims(self):
        from qpe_lab.circuits.gadgets.synthesis import ccx_clifford_t
        from qpe_lab.evaluation.verification.engine import PrefixResources

        tracker = PrefixResources({2: (15, -16384), 3: (15, -1)})
        tracker.run(list(ccx_clifford_t(0, 1, 2)))
        self.assertIn(2, tracker.states)
        tracker.run([("assert_zero", (3,), None)])
        self.assertEqual(tracker.states[3], (15, -1))
        tracker.run([("cx", (0, 3), None)])
        self.assertNotIn(3, tracker.states)
        tracker.run([("h", (2,), None)])
        self.assertNotIn(2, tracker.states)

    def test_tensor_budget_is_unknown_never_approximate_acceptance(self):
        from qpe_lab.circuits.sim import CyclotomicNetwork, Unsupported

        net = CyclotomicNetwork(max_products=0)
        axis = net.axis()
        net.factor([axis], {0: ((0, 1),), 1: ((0, 1),)})
        with self.assertRaises(Unsupported):
            net.contract()

    def test_rare_basis_error_is_not_hidden_from_coherent_fingerprints(self):
        from qpe_lab.circuits.gadgets.synthesis import ccx_clifford_t
        from qpe_lab.circuits.sim import Incorrect, tensor_isometry_check

        n = 24
        triples = [(0, 1, n)] + [(n + k - 1, k + 1, n + k) for k in range(1, n - 1)]
        compute = [g for triple in triples for g in ccx_clifford_t(*triple)]
        uncompute = [g for triple in reversed(triples) for g in ccx_clifford_t(*triple)]
        forms = self.forms(n) | {q: frozenset() for q in range(n, 2 * n - 1)}
        with self.assertRaises(Incorrect):
            tensor_isometry_check(
                compute + [("z", (2 * n - 2,), None)] + uncompute,
                [],
                forms,
                seed=b"rare-error",
                rounds=2,
            )

    def test_reward_sampling_route_catches_rare_phase_missed_by_basis_probes(self):
        from qpe_lab.circuits.gadgets.synthesis import ccx_clifford_t
        from qpe_lab.circuits.sim import Incorrect, sample_monomial_equivalence
        from qpe_lab.evaluation.verification.sampled import SamplingContext

        n = 24
        triples = [(0, 1, n)] + [(n + k - 1, k + 1, n + k) for k in range(1, n - 1)]
        compute = [g for triple in triples for g in ccx_clifford_t(*triple)]
        uncompute = [g for triple in reversed(triples) for g in ccx_clifford_t(*triple)]
        rare = compute + [("z", (2 * n - 2,), None)] + uncompute
        forms = self.forms(n) | {q: frozenset() for q in range(n, 2 * n - 1)}
        probes = sample_monomial_equivalence(
            rare, [], list(range(n)), seed=b"rare-route", shots=4096, forms=forms
        )
        self.assertEqual(probes["random_shots"], 4096)
        with self.assertRaises(Incorrect):
            SamplingContext(b"rare-route", 1e-12).compare(rare, [], list(range(n)), forms=forms)

    def test_basis_diagnostic_pass_cannot_become_a_verification_pass(self):
        from unittest.mock import patch

        from qpe_lab.circuits.sim import Unsupported
        from qpe_lab.contracts import VerificationGoal
        from qpe_lab.evaluation.verification.sampled import SamplingContext, verify

        goal = VerificationGoal(
            {"gates": [("x", (0,), None)] * 2, "protected_wires": [0], "ancilla_initial": "zero"},
            [],
            "pin",
            10,
        )
        context = SamplingContext(b"diagnostic-only", 1e-12)
        with patch.object(context, "fingerprint", side_effect=Unsupported("test contraction cap")):
            result = verify(goal, context)
        self.assertEqual(result.status, "unknown")
        self.assertIn("basis probes found no counterexample", result.detail)
        with patch.object(
            context,
            "compare",
            return_value={
                "claim": "uniform basis-action",
                "false_accept_bound": 0,
                "confidence_failure_budget": 1e-12,
            },
        ):
            self.assertEqual(verify(goal, context).status, "unknown")

    def test_resource_assumptions_bind_cache_and_do_not_extend_unknown_semantics(self):
        from dataclasses import replace

        from qpe_lab.contracts import VerificationGoal
        from qpe_lab.evaluation.verification.exact import verify

        goal = VerificationGoal(
            {"gates": [("x", (0,), None)], "protected_wires": [0], "ancilla_initial": "zero"},
            [("neg", (), None)],
            "a" * 64,
            3,
            resource_inputs={0: (1, -1)},
        )
        self.assertEqual(verify(goal).status, "proved")
        self.assertNotEqual(
            goal.statement_sha256, replace(goal, resource_inputs={0: (1, 0)}).statement_sha256
        )
        self.assertEqual(verify(replace(goal, semantics="unregistered")).status, "unknown")

    def test_wide_channel_sums_all_measurements_without_enumeration(self):
        from qpe_lab.circuits.sim import Incorrect, tensor_channel_fingerprint

        n = 40
        before = [("h", (q,), None) for q in range(n)] + [("hmr", (q,), q) for q in range(n)]
        after = [("hmr", (q,), q + 100) for q in range(n)]
        forms = self.forms(n + 1)
        result = tensor_channel_fingerprint(
            before, after, forms, {}, seed=b"wide-channel", rounds=2
        )
        self.assertEqual(result["active_wires"], 41)
        self.assertLess(result["peak_frontier"], 16)
        with self.assertRaises(Incorrect):
            tensor_channel_fingerprint(
                before, after + [("z", (n,), None)], forms, {}, seed=b"wide-channel", rounds=2
            )

    def test_tensor_channel_feedback_xor_external_and_live_outputs(self):
        from qpe_lab.circuits.sim import (
            Incorrect,
            tensor_channel_fingerprint,
            verify_channel_equivalence,
        )

        before = [
            ("hmr", (0,), 7),
            ("oxor", (), [8, 7, 20]),
            ("h", (1,), {"condition": 8}),
            ("neg", (), {"condition": 7}),
        ]
        after = [("hmr", (0,), 70), ("oxor", (), [8, 20, 70]), ("h", (1,), {"condition": 8})]
        options = {"external": [20], "outputs": [8]}
        verify_channel_equivalence(before, after, [0, 1], range(4), **options)
        tensor_channel_fingerprint(
            before, after, self.forms(2), {}, seed=b"dynamic", rounds=2, **options
        )
        with self.assertRaises(Incorrect):
            tensor_channel_fingerprint(
                before, after[:-1], self.forms(2), {}, seed=b"dynamic", rounds=2, **options
            )
        # Observing the measured bit distinguishes HMR(|+>) and HMR(|0>).
        with self.assertRaises(Incorrect):
            tensor_channel_fingerprint(
                [("h", (0,), None), ("hmr", (0,), 7)],
                [("hmr", (0,), 7)],
                {0: frozenset()},
                {},
                seed=b"output",
                rounds=2,
                outputs=[7],
            )

    def test_dynamic_gradient_restoration_and_assertion_loss(self):
        from qpe_lab.circuits.sim import Incorrect, tensor_channel_fingerprint

        restore = [("hmr", (1,), 0), ("x", (1,), None), ("h", (1,), None)]
        resources = {1: (15, -16384)}
        tensor_channel_fingerprint([], restore, self.forms(2), resources, seed=b"restore", rounds=2)
        for bad in (
            restore[:-1],
            restore + [("z", (1,), None)],
            restore + [("assert_zero", (0,), None)],
        ):
            with self.assertRaises(Incorrect):
                tensor_channel_fingerprint(
                    [], bad, self.forms(2), resources, seed=b"restore", rounds=2
                )

    def test_conditional_tensor_gates_against_exact_kraus_oracle(self):
        from qpe_lab.circuits.sim import (
            Incorrect,
            tensor_channel_fingerprint,
            verify_channel_equivalence,
        )

        inverse = {"t": "tdg", "tdg": "t", "s": "sdg", "sdg": "s"}
        after = [("hmr", (0,), 7)]
        for name in ("h", "x", "z", "s", "sdg", "t", "tdg", "cx", "cz", "neg"):
            wires = () if name == "neg" else (1, 2) if name in {"cx", "cz"} else (1,)
            before = after + [
                (name, wires, {"condition": 7}),
                (inverse.get(name, name), wires, {"condition": 7}),
            ]
            verify_channel_equivalence(before, after, [0, 1, 2], range(8))
            tensor_channel_fingerprint(
                before, after, self.forms(3), {}, seed=b"all-conditional", rounds=2
            )
            with self.assertRaises(Incorrect):
                tensor_channel_fingerprint(
                    before,
                    after + [("z", (2,), None)],
                    self.forms(3),
                    {},
                    seed=b"all-conditional",
                    rounds=2,
                )

    def test_prefix_resource_assumptions_against_independent_dense_columns(self):
        import random

        from qpe_lab.circuits.sim import _cyclotomic_product, exact_clifford_t_columns
        from qpe_lab.evaluation.verification.engine import PrefixResources

        rng = random.Random(159)
        tracker = PrefixResources({2: (3, 4), 3: (3, 1)})
        gates = []

        def on_resource(word):
            matrix = exact_clifford_t_columns(word, 4, range(16))
            out = []
            for row in matrix:
                for logical in range(4):
                    total = [0] * 4
                    for resource in range(4):
                        power = (4 * (resource & 1) + (resource >> 1)) % 8
                        phase = tuple(
                            (-1 if power >= 4 else 1) if i == power % 4 else 0 for i in range(4)
                        )
                        for i, value in enumerate(
                            _cyclotomic_product(row[logical | (resource << 2)], phase)
                        ):
                            total[i] += value
                    out.append(total)
            return out

        for _ in range(16):
            name = rng.choice(["x", "s", "tdg", "cx", "cz"])
            wires = tuple(rng.sample(range(4), 2 if name in {"cx", "cz"} else 1))
            gate = (name, wires, None)
            gates.append(gate)
            tracker.run([gate])
            for q, (bits, phase) in tracker.states.items():
                self.assertEqual(bits, 3)
                power = phase % 8
                stabilizer = (
                    [("tdg", (q,), None)] * power
                    + [("x", (q,), None)]
                    + [("t", (q,), None)] * power
                )
                self.assertEqual(on_resource(gates), on_resource(gates + stabilizer))


class LeanBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qpe_lab.evaluation.verification.lean import toolchain

        if not (toolchain() / "bin/lake").exists():
            raise unittest.SkipTest("Install the optional pinned Lean toolchain")

    def goal(self, before, after, protected=(0,), proof="by decide"):
        from qpe_lab.contracts import VerificationGoal

        return VerificationGoal(
            {"gates": before, "protected_wires": list(protected), "ancilla_initial": "zero"},
            after,
            "a" * 64,
            8,
            proof={"backend": "lean", "source": proof},
        )

    def test_actual_kernel_proof_and_all_monomial_gates(self):
        from qpe_lab.evaluation.verification.lean import verify

        before = [
            ("x", (0,), None),
            ("cx", (0, 1), None),
            ("cz", (0, 1), None),
            ("z", (0,), None),
            ("s", (1,), None),
            ("t", (0,), None),
            ("neg", (), None),
        ]
        inverse = {"t": "tdg", "s": "sdg"}
        after = [(inverse.get(n, n), q, v) for n, q, v in reversed(before)]
        result = verify(self.goal(before + after, [], [0, 1]))
        self.assertEqual(result.status, "proved", result.detail)
        self.assertEqual(
            result.evidence["permitted_axioms"], ["propext", "Quot.sound", "Classical.choice"]
        )

    def test_false_phase_and_sorry_do_not_pass(self):
        from qpe_lab.evaluation.verification.lean import verify

        for after, proof in (
            ([("z", (0,), None)], "by decide"),
            ([("z", (0,), None)], "by sorry"),
            ([], "by sorry"),
        ):
            result = verify(self.goal([("t", (0,), None)] * 2, after, proof=proof))
            self.assertEqual(result.status, "unknown", result.detail)

    def test_wrong_theorem_is_rejected_by_comparator(self):
        from unittest.mock import patch

        from qpe_lab.evaluation.verification import lean

        goal = self.goal([("t", (0,), None)] * 2, [("s", (0,), None)])
        challenge = lean.problem(goal)
        solution = (
            "import Std\nnamespace QPE\ntheorem circuit_equivalent : True := by trivial\nend QPE\n"
        )
        with patch.object(lean, "problem", side_effect=[challenge, solution]):
            result = lean.verify(goal)
        self.assertEqual(result.status, "unknown", result.detail)

    def test_zero_input_condition_changes_the_theorem(self):
        from qpe_lab.evaluation.verification.lean import verify

        before = [("t", (1,), None)]
        good = verify(self.goal(before, [], [0]))
        bad = verify(self.goal(before, [], [1]))
        self.assertEqual(good.status, "proved", good.detail)
        self.assertEqual(bad.status, "unknown", bad.detail)

    def test_proof_compilation_cannot_read_or_write_host_canary(self):
        from qpe_lab.evaluation.verification.lean import verify

        with tempfile.TemporaryDirectory(dir=project_root() / "runs", prefix="lean-host-") as temp:
            canary = Path(temp) / "canary.txt"
            canary.write_text("maintainer canary")
            proof = (
                "by decide\n"
                "#eval show IO Unit from do\n"
                f"  let path := System.FilePath.mk {json.dumps(str(canary))}\n"
                '  if ← path.pathExists then throw (IO.userError "host file exposed")\n'
                '  try IO.FS.writeFile path "changed" catch _ => pure ()\n'
            )
            result = verify(self.goal([("t", (0,), None)] * 2, [("s", (0,), None)], proof=proof))
            self.assertEqual(result.status, "proved", result.detail)
            self.assertEqual(canary.read_text(), "maintainer canary")

    def test_generator_attachment_roundtrip_and_required_policy(self):
        from qpe_lab import CircuitBuilder
        from qpe_lab.circuits.ops import read_circuit
        from qpe_lab.contracts import digest, write_json
        from qpe_lab.evaluation.verification.engine import verify_entry

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.json"
            reference = {
                "schema": "qpe.gate-reference.v2",
                "protected_wires": [0],
                "ancilla_initial": "zero",
                "gates": [("t", (0,), None)] * 2,
            }
            write_json(path, reference)
            b = CircuitBuilder(path).submit([("s", (0,), None)]).attach_lean("by decide")
            artifact = Path(tmp) / "circuit.ops"
            b.emit(artifact)
            candidate = read_circuit(artifact)["entries"]["walk"]
            _, report = verify_entry(
                b.references["walk"], candidate, digest(path), required_backend="lean"
            )
            self.assertEqual(report["evidence"]["backend"], "lean")
            self.assertEqual(report["cost"]["t_count"], 0)


if __name__ == "__main__":
    unittest.main()
