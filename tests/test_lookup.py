"""Cross-check the symbolic verifier against exhaustive independent small cases."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from uncompute import Builder, Simulator
from uncompute.evalcore.loader import CompactParseError

from qpe_lab.circuits.builder import build_lookup
from qpe_lab.circuits.checks import LookupChecker
from qpe_lab.circuits.compact_reader import interpret
from qpe_lab.circuits.gadgets.qrom import lookup_xor
from qpe_lab.circuits.sim import Incorrect, Unsupported
from qpe_lab.contracts import digest
from qpe_lab.tasks.data.angles import elimination_angles, reconstruct
from qpe_lab.tasks.task import load_lookup


def make_task(directory, words=(1, 6, 3, 0), word_bits=3):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table = directory / "table.json"
    table.write_text(json.dumps({"words_hex": [hex(x) for x in words]}))
    task = directory / "task.json"
    task.write_text(
        json.dumps(
            {
                "schema": "qpe.lookup.v1",
                "name": "test-lookup",
                "table": table.name,
                "table_sha256": digest(table),
                "address_bits": (len(words) - 1).bit_length(),
                "word_bits": word_bits,
                "max_qubits": 128,
            }
        )
    )
    return load_lookup(task)


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.task = make_task(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def check(self, b):
        path = self.root / "candidate.ops"
        with path.open("wb") as f:
            b.emit(dest=f, meta=False)
        checker = LookupChecker(self.task)
        cost = interpret(path, checker, self.task.max_qubits)
        return checker, cost

    def test_two_constructions_against_all_classical_inputs(self):
        for method, block in (("unary", 1), ("selectswap", 2)):
            with self.subTest(method=method):
                b, addr, target = build_lookup(self.task, method=method, block_size=block)
                checker, cost = self.check(b)
                addresses = [a for a in range(4) for _ in range(8)]
                zs = list(range(8)) * 4
                for seed in (0, 7):
                    sim = Simulator(b, batch=32, seed=seed, strict=True)
                    sim.write(addr, addresses)
                    sim.write(target, zs)
                    sim.run()
                    self.assertEqual(sim.read(addr), addresses)
                    self.assertEqual(
                        sim.read(target), [z ^ self.task.words[a] for a, z in zip(addresses, zs)]
                    )
                    sim.assert_clean()
                    self.assertEqual(sim.avg_toffoli, cost["toffoli"])
                self.assertEqual(checker.evidence()["addresses"], 4)

    def test_empty_identity_is_rejected(self):
        b = Builder()
        b.qubit_register(b.qubits(2))
        b.qubit_register(b.qubits(3))
        with self.assertRaises(Incorrect):
            self.check(b)

    def test_lookup_component_composes_on_existing_noncontiguous_wires(self):
        b = Builder()
        wires = b.qubits(7)
        address, target = [wires[5], wires[1]], [wires[4], wires[0], wires[3]]
        b.qubit_register(address)
        b.qubit_register(target)
        # Two identical XOR lookups cancel; a third retains the task's action.
        # Scratch allocated before the component must also survive untouched.
        for method in ("unary", "selectswap", "unary"):
            lookup_xor(b, address, target, self.task.words, method=method, block_size=2)
        self.check(b)
        addresses = [a for a in range(4) for _ in range(8)]
        zs = list(range(8)) * 4
        sim = Simulator(b, batch=32, seed=7, strict=True)
        sim.write(address, addresses)
        sim.write(target, zs)
        sim.run()
        self.assertEqual(sim.read(address), addresses)
        self.assertEqual(sim.read(target), [z ^ self.task.words[a] for a, z in zip(addresses, zs)])
        sim.assert_clean()

    def test_target_dependent_phase_is_rejected(self):
        b, _, target = build_lookup(self.task)
        b.z(target[0])
        with self.assertRaisesRegex(Incorrect, "phase"):
            self.check(b)

    def test_rare_address_phase_is_rejected(self):
        b, address, target = build_lookup(self.task)
        b.ccz(address[0], address[1], target[2])
        with self.assertRaisesRegex(Incorrect, "phase"):
            self.check(b)

    def test_dirty_workspace_and_reset_are_rejected(self):
        for reset in (False, True):
            b, _, _ = build_lookup(self.task)
            scratch = b.qubit()
            b.x(scratch)
            if reset:
                b.r(scratch)
            with self.subTest(reset=reset), self.assertRaises(Incorrect):
                self.check(b)

    def test_measurement_of_logical_output_is_unsupported(self):
        b, _, target = build_lookup(self.task)
        b.hmr(target[0])
        with self.assertRaises(Unsupported):
            self.check(b)

    def test_phase_debt_with_target_variables(self):
        for correction in (True, False):
            b, _, target = build_lookup(self.task)
            scratch = b.qubit()
            b.ccx(target[0], target[1], scratch)
            outcome = b.hmr(scratch)
            if correction:
                b.cz(target[0], target[1], cond=outcome)
            if correction:
                self.check(b)
            else:
                with self.assertRaisesRegex(Incorrect, "measurement"):
                    self.check(b)

    def test_nonlinear_target_logic_can_cancel(self):
        b, _, target = build_lookup(self.task)
        b.ccx(target[0], target[1], target[2])
        b.ccx(target[0], target[1], target[2])
        self.check(b)

    def test_overwritten_outcome_does_not_erase_debt(self):
        checker = LookupChecker(self.task)
        checker.initialize([(0, [0, 1]), (0, [2, 3, 4])])
        checker.apply("CX", 0, 0, 5, 0, None)
        checker.apply("HMR", 0, 0, 5, 0, None)
        checker.apply("HMR", 0, 0, 6, 0, None)
        self.assertTrue(checker.debts)

    def test_register_alias_rejected(self):
        checker = LookupChecker(self.task)
        with self.assertRaises(Incorrect):
            checker.initialize([(0, [0, 1]), (0, [1, 2, 3])])

    def test_table_hash_and_total_domain(self):
        (self.root / "table.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_lookup(self.task.path)

    def test_trailing_bytes_rejected(self):
        b, _, _ = build_lookup(self.task)
        path = self.root / "bad.ops"
        with path.open("wb") as f:
            b.emit(dest=f, meta=False)
        with path.open("ab") as f:
            f.write(b"not part of the circuit")
        with self.assertRaises(CompactParseError):
            interpret(path, LookupChecker(self.task), 128)

    def test_givens_sign_and_order(self):
        for v in ([1, 0, 0], [-1, 0, 0], [0, -1, 0], [-2, 3, -4], [0, 0, -1]):
            angles = elimination_angles(v)
            expected = np.asarray(v) / np.linalg.norm(v)
            np.testing.assert_allclose(reconstruct(angles), expected, atol=1e-14)
            # Independently form the forward rotation matrices.
            matrix = np.eye(len(v))
            for j, theta in zip(range(len(v) - 1, 0, -1), angles):
                rotation = np.eye(len(v))
                c, s = np.cos(theta), np.sin(theta)
                rotation[j - 1 : j + 1, j - 1 : j + 1] = [[c, s], [-s, c]]
                matrix = rotation @ matrix
            np.testing.assert_allclose(matrix @ expected, np.eye(len(v))[0], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
