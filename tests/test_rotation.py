"""Rotation composition, full Fock semantics, malformed artifacts, and real jobs."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from test_lookup import make_task

from qpe_lab import build_lookup, build_rotation
from qpe_lab.circuits.ops import HYBRID_MAGIC, read_hybrid
from qpe_lab.circuits.sim import DenseFockSimulator, GaussianSimulator, Incorrect, fock_lift
from qpe_lab.contracts import digest, project_root
from qpe_lab.evaluation.protocol import evaluate
from qpe_lab.evaluation.rotation import reference_modes, verify_rotation
from qpe_lab.tasks.data.angles import reconstruct, write_rotation_task
from qpe_lab.tasks.task import load_rotation


def make_rotation_task(directory):
    lookup = make_task(directory, words=(17, 43, 62, 0), word_bits=6)
    table_path = lookup.path.parent / "table.json"
    table = json.loads(table_path.read_text())
    table.update(angle_bits=3, angles_per_word=2)
    table_path.write_text(json.dumps(table))
    spec = json.loads(lookup.path.read_text())
    spec["table_sha256"] = digest(table_path)
    lookup.path.write_text(json.dumps(spec))
    return load_rotation(write_rotation_task(lookup.path, spin_copies=1))


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.task = make_rotation_task(self.root / "task")
        self.sequence = 0

    def tearDown(self):
        self.temp.cleanup()

    def check(self, circuit, task=None):
        self.sequence += 1
        artifact = self.root / f"circuit-{self.sequence}.ops"
        circuit.emit(artifact)
        return verify_rotation(
            task or self.task, artifact, self.root / f"out-{self.sequence}", kernel=False
        )

    def test_both_baselines_have_explicit_partial_synthesis_cost(self):
        for method in ("unary", "selectswap"):
            with self.subTest(method=method):
                report = self.check(build_rotation(self.task, method=method, block_size=2))
                self.assertEqual(report["correctness_status"], "exact_verified")
                self.assertEqual(report["cost"]["status"], "partial")
                self.assertIsNone(report["cost"]["toffoli"])
                self.assertIsNone(report["cost"]["synthesis_t_count"])
                self.assertEqual(report["cost"]["controlled_givens"], 6)
                self.assertTrue(report["numerical_crosscheck"]["dense_fock_checked"])
                self.assertFalse(report["numerical_crosscheck"]["phase_alignment_used"])
                self.assertLessEqual(report["cost"]["qubits"], self.task.max_qubits)

    def test_dense_consumer_for_every_angle_word_and_enable_branch(self):
        gates = build_rotation(self.task).steps[1:-1]
        for word in range(1 << self.task.word_bits):
            for enable in (False, True):
                full = DenseFockSimulator(3).run(gates, word, 3, enable=enable)
                expected = fock_lift(reference_modes(self.task, word, enable=enable))
                np.testing.assert_allclose(full, expected, atol=2e-14)
                self.assertEqual(full[0, 0], 1)
                self.assertEqual(full[-1, -1], 1)
        self.assertGreater(np.max(np.abs(reference_modes(self.task, 17) - np.eye(3))), 0.1)

    def test_uncontrolled_two_spin_task_has_no_enable_qubit(self):
        path = write_rotation_task(
            self.task.path.parent / "task.json", spin_copies=2, controlled=False
        )
        task = load_rotation(path)
        report = self.check(build_rotation(task), task=task)
        self.assertEqual(report["correctness_status"], "exact_verified")
        self.assertEqual(report["cost"]["controlled_givens"], 12)
        self.assertEqual(report["cost"]["rotation_controls"], "angle bit")
        self.assertEqual(task.reserved_qubits, 6)
        self.assertTrue(report["numerical_crosscheck"]["dense_fock_checked"])

    def test_nonadjacent_givens_against_fermionic_generator_exponential(self):
        n, p, q, turns, beta = 4, 0, 3, 3, 3
        annihilation = []
        for mode in range(n):
            matrix = np.zeros((1 << n, 1 << n))
            for state in range(1 << n):
                if state >> mode & 1:
                    parity = (state & ((1 << mode) - 1)).bit_count() % 2
                    matrix[state ^ (1 << mode), state] = (-1) ** parity
            annihilation.append(matrix)
        generator = annihilation[p].T @ annihilation[q] - annihilation[q].T @ annihilation[p]
        eigenvalues, vectors = np.linalg.eigh(1j * generator)
        theta = math.tau * turns / (1 << beta)
        expected = (vectors * np.exp(-1j * theta * eigenvalues)) @ vectors.conj().T
        dense, gaussian = DenseFockSimulator(n), GaussianSimulator(n)
        dense.givens(p, q, turns, beta)
        gaussian.givens(p, q, turns, beta)
        np.testing.assert_allclose(dense.matrix, expected, atol=2e-14)
        np.testing.assert_allclose(fock_lift(gaussian.matrix), expected, atol=2e-14)
        reverse = DenseFockSimulator(n)
        reverse.givens(q, p, -turns, beta)
        np.testing.assert_allclose(reverse.matrix, expected, atol=2e-14)

    def test_commuting_bit_order_and_integer_full_turns_are_allowed(self):
        circuit = build_rotation(self.task)
        circuit.steps[1:4] = list(reversed(circuit.steps[1:4]))
        circuit.steps[2]["turns"] += 1 << self.task.angle_bits
        self.assertEqual(self.check(circuit)["correctness_status"], "exact_verified")

    def test_missing_consumer_wrong_sign_or_control_is_rejected(self):
        for defect in ("missing", "sign", "enable", "angle", "order"):
            circuit = build_rotation(self.task)
            if defect == "missing":
                del circuit.steps[1]
            elif defect == "sign":
                circuit.steps[1]["turns"] *= -1
            elif defect == "enable":
                circuit.steps[1]["controlled"] = False
            elif defect == "angle":
                circuit.steps[1]["angle_bit"] = 0
            else:
                circuit.steps[1:-1] = list(reversed(circuit.steps[1:-1]))
            with self.subTest(defect=defect), self.assertRaisesRegex(Incorrect, "consumer"):
                self.check(circuit)

    def test_extra_phase_or_claimed_cost_fields_are_rejected(self):
        for defect in ("phase", "cost"):
            circuit = build_rotation(self.task)
            if defect == "phase":
                circuit.steps.insert(1, {"kind": "phase", "value": -1})
            else:
                circuit.steps[1]["toffoli"] = 0
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.check(circuit)

    def test_lookup_branch_phase_and_bad_unload_are_rejected(self):
        for defect in ("phase", "dirty", "identity"):
            circuit = build_rotation(self.task)
            if defect == "identity":
                wrong = replace(self.task, words=(0, 0, 0, 0))
                b, _, _ = build_lookup(wrong)
            else:
                b, address, _ = build_lookup(self.task)
                if defect == "phase":
                    b.z(address[0])
                else:
                    b.x(b.qubit())
            circuit.steps.pop()
            circuit.lookup(b)
            with self.subTest(defect=defect), self.assertRaises(Incorrect):
                self.check(circuit)

    def test_system_and_enable_qubits_count_against_budget(self):
        circuit = build_rotation(self.task, method="selectswap", block_size=2)
        low = replace(
            self.task,
            max_qubits=self.task.address_bits + self.task.word_bits + self.task.reserved_qubits,
        )
        with self.assertRaisesRegex(ValueError, "limits"):
            self.check(circuit, task=low)

    def test_invalid_task_metadata_and_artifact_fields_fail_closed(self):
        spec = json.loads(self.task.path.read_text())
        spec["mode_order"] = "interleaved"
        self.task.path.write_text(json.dumps(spec))
        with self.assertRaises(ValueError):
            load_rotation(self.task.path)
        path = self.root / "bad.ops"
        path.write_bytes(HYBRID_MAGIC + b'{"schema":"qpe.hybrid.v1","schema":"forged"}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_hybrid(path)
        circuit = build_rotation(self.task)
        circuit.steps[1]["p"] = self.task.mode_count
        with self.assertRaisesRegex(ValueError, "outside"):
            self.check(circuit)

    def test_frozen_fe2s2_spin_blocks_match_quantized_vector(self):
        task = load_rotation(project_root() / "benchmarks/dfthc/fe2s2/rotation.task.json")
        mask = (1 << task.angle_bits) - 1
        for address in (0, 209, 255):
            word = task.words[address]
            angles = [
                (word >> (i * task.angle_bits) & mask) * math.tau / (1 << task.angle_bits)
                for i in range(task.spatial_modes - 1)
            ]
            vector = reconstruct(angles)
            matrix = reference_modes(task, word)
            np.testing.assert_allclose(matrix[:20, 0], vector, atol=2e-14)
            np.testing.assert_allclose(matrix[20:, 20], vector, atol=2e-14)
            np.testing.assert_allclose(matrix[:20, 20:], 0, atol=0)
            if address == 255:
                np.testing.assert_array_equal(matrix, np.eye(40))


class RotationRunnerTests(unittest.TestCase):
    def test_python_generator_through_isolation_and_kernel(self):
        with tempfile.TemporaryDirectory(
            dir=project_root() / "runs", prefix="rotation-test-"
        ) as folder:
            root = Path(folder)
            task = make_rotation_task(root / "task")
            candidate = root / "generator.py"
            candidate.write_text("from qpe_lab import build_rotation\nbuild_rotation().emit()\n")
            run, report = evaluate(task.path, candidate)
            self.assertEqual(report["correctness_status"], "exact_verified", report)
            self.assertEqual(report["cost"]["status"], "partial")
            self.assertTrue((run / "hybrid/load.ops").is_file())
            for block in report["lookup_blocks"]:
                self.assertEqual(block["kernel_crosscheck"]["verdict"], "OK")


if __name__ == "__main__":
    unittest.main()
