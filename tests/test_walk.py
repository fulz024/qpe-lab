"""SOS physics, quantum controls, negative cases, and isolated Python submission."""

import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from qpe_lab import Builder, build_walk
from qpe_lab.circuits.gadgets.givens import phase_gradient_givens, prepare_phase_gradient
from qpe_lab.circuits.gadgets.prepare import (
    alias_counts,
    alias_prepare,
    alias_table,
    compare_greater_equal,
    measured_compare,
    prepare_weights,
    uniform_range,
)
from qpe_lab.circuits.gadgets.qrom import (
    coherent_lookup,
    deferred_erase,
    deferred_lookup,
    dynamic_lookup,
    lookup_xor,
)
from qpe_lab.circuits.gadgets.select import qrom_select
from qpe_lab.circuits.gadgets.synthesis import (
    DynamicCircuit,
    GateExpansion,
    adjoint,
    ccx_clifford_t,
    compile_rotations,
    expand_clifford_t,
    expand_walk,
    rotation_residual,
    rotation_sequence,
)
from qpe_lab.circuits.ops import WALK_MAGIC, read_walk
from qpe_lab.circuits.sim import (
    DenseFockSimulator,
    Incorrect,
    SOSSimulator,
    SparseStatevectorSimulator,
    StatevectorSimulator,
    SymbolicSimulator,
    jordan_wigner,
)
from qpe_lab.contracts import digest, project_root
from qpe_lab.evaluation.protocol import evaluate
from qpe_lab.evaluation.walk import reference_generators, verify_walk
from qpe_lab.tasks.bundle import check_judge, freeze_judge, provenance
from qpe_lab.tasks.data.hamiltonian import (
    centered_sos,
    make_model,
    read_fcidump,
    with_alias_preparation,
    with_paper_preparation,
    write_walk_task,
)
from qpe_lab.tasks.task import load_walk


def measurement_kraus(gates, num_qubits, branch):
    """Unnormalized full input matrix for one fixed measurement record."""
    size = 1 << num_qubits
    state, record, outcomes = np.eye(size, dtype=complex), {}, iter(branch)
    for name, wires, value in gates:
        if name == "oxor":
            target, left, right = value
            record[target] = (0 if left == -1 else record[left]) ^ record[right]
            continue
        if isinstance(value, dict):
            if not record[value["condition"]]:
                continue
            value = None
        if name == "hmr":
            outcome = next(outcomes)
            assert outcome in (0, 1)
            record[value] = outcome
            low = np.array([i for i in range(size) if not (i >> wires[0] & 1)])
            high = low | (1 << wires[0])
            state[low] = (state[low] + (-1) ** outcome * state[high]) / np.sqrt(2)
            state[high] = 0
        else:
            state = StatevectorSimulator(num_qubits).run([(name, wires, value)], state)
    assert next(outcomes, None) is None
    return state, record


def toy_model():
    return make_model(
        np.array([[[0.6, 0.8], [0.8, -0.6]]]),
        np.array([[[0.7, -0.2], [-0.3, 0.5]]]),
        np.array([[0.25, -0.4]]),
        np.array([-0.35, 0.6]),
        np.array([[0.8, -0.6], [0.6, 0.8]]),
        energy_offset=-1.2,
        beta=5,
        coefficient_bits=12,
    )


class WalkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.task = load_walk(write_walk_task(toy_model(), self.directory, "test-sos"))

    def tearDown(self):
        self.temp.cleanup()

    def check(self, circuit, task=None):
        path = self.directory / "circuit.ops"
        circuit.emit(path)
        return verify_walk(task or self.task, path)

    def test_paper_constant_gradient_rotation_and_flagged_uniform(self):
        from qpe_lab.circuits.gadgets.givens import constant_gradient_rotation
        from qpe_lab.circuits.gadgets.prepare import flagged_uniform, flagged_uniform_parameters

        for turns in (0, 1, 3, 7, 11, 15):
            circuit = DynamicCircuit(1)
            gradient = circuit.qubits(4)
            initialization = prepare_phase_gradient(circuit, gradient)
            constant_gradient_rotation(circuit, 0, gradient, turns)
            circuit.gates.extend(adjoint(initialization))
            initial = dict(enumerate(np.eye(2, dtype=complex)))
            expected = StatevectorSimulator(1).run(
                [("ry", (0,), 4 * math.pi * turns / 16)], np.eye(2, dtype=complex)
            )
            for seed in (0, 1, 7):
                output = SparseStatevectorSimulator(circuit.peak_qubits).run(
                    circuit.gates, initial, seed=seed
                )
                np.testing.assert_allclose(
                    [output.get(i, np.zeros(2)) for i in (0, 1)], expected, atol=1e-12
                )
                self.assertLess(
                    sum(np.linalg.norm(v) ** 2 for i, v in output.items() if i >= 2), 1e-24
                )
        for size, width, precision in ((3, 2, 3), (5, 3, 4), (90, 7, 7)):
            circuit = DynamicCircuit(width + 2)
            gradient = circuit.qubits(precision)
            initialization = prepare_phase_gradient(circuit, gradient)
            flagged_uniform(circuit, list(range(width)), size, width, width + 1, gradient)
            output = SparseStatevectorSimulator(circuit.peak_qubits).run(
                circuit.gates, {0: np.array([1.0 + 0j])}
            )
            p = sum(abs(v[0]) ** 2 for i, v in output.items() if i >> (width + 1) & 1)
            expected = flagged_uniform_parameters(size, width, precision)["success_probability"]
            self.assertAlmostEqual(p, expected, places=11)
            probabilities = [
                sum(abs(v[0]) ** 2 for i, v in output.items() if i & ((1 << width) - 1) == j)
                for j in range(size)
            ]
            np.testing.assert_allclose(probabilities, p / size, atol=1e-12)
            flagged_uniform(
                circuit, list(range(width)), size, width, width + 1, gradient, inverse=True
            )
            circuit.gates.extend(adjoint(initialization))
            output = SparseStatevectorSimulator(circuit.peak_qubits).run(
                circuit.gates, {0: np.array([1.0 + 0j])}, seed=7
            )
            self.assertLess(abs(output[0][0] - 1), 1e-11)
            self.assertLess(sum(abs(v[0]) ** 2 for i, v in output.items() if i), 1e-22)

    def test_paper_strict_physical_walk_and_projector(self):
        from unittest.mock import patch

        from qpe_lab.evaluation.walk import verify_strict_paper_tables

        model = make_model(
            np.array([[[0.6, 0.8]]]),
            np.array([[[0.7]]]),
            np.array([[0.4]]),
            np.array([-0.35, 0.6]),
            np.array([[0.8, -0.6], [0.6, 0.8]]),
            energy_offset=-1.2,
            beta=2,
            coefficient_bits=8,
        )
        task = load_walk(
            write_walk_task(
                with_paper_preparation(model, 1, uniform_bits=3),
                self.directory / "strict",
                "paper-strict-small",
            )
        )
        circuit = build_walk(
            task,
            implementation="paper_strict",
            outer_block_size=2,
            qrom_block_size=2,
            outer_unlookup_size=2,
            unlookup_block_size=4,
        )
        report = self.check(circuit, task)
        self.assertTrue(report["physical_dynamic_crosscheck"]["checked"])
        self.assertLess(report["physical_dynamic_crosscheck"]["max_error"], 1e-10)
        self.assertAlmostEqual(report["normalization"]["uniform_success_probability"], 15 / 16)
        with self.assertRaises(ValueError):
            build_walk(task, implementation="paper")
        corrupt = copy.deepcopy(circuit)
        corrupt.program["implementation"]["kind"] = "paper"
        with self.assertRaises(Incorrect):
            self.check(corrupt, task)
        gates, expansion = expand_walk(task, circuit.program)
        expansion.gates = list(gates)
        _, start, end = next(block for block in expansion.blocks if block[0] == "angle_erase")
        victim = next(
            i
            for i in range(start, end)
            if expansion.gates[i][0] == "cz" and isinstance(expansion.gates[i][2], dict)
        )
        # Preserve block positions while deleting a physical phase correction.
        expansion.gates[victim] = ("assert_zero", (expansion.registers["inner_uniform"][0],), None)
        with (
            patch(
                "qpe_lab.circuits.gadgets.synthesis.expand_walk",
                return_value=(iter(expansion.gates), expansion),
            ),
            self.assertRaises(Incorrect),
        ):
            verify_strict_paper_tables(task, circuit.program)

    def test_paper_measured_comparator_symbolic_all_outcomes(self):
        circuit = DynamicCircuit(5)
        measured_compare(circuit, [0, 1], [2, 3], 4)
        checker = SymbolicSimulator(4, 1)
        checker.initialize([(0, [0, 1, 2, 3]), (0, [4])])
        checker.apply_elementary(circuit.gates)
        checker.assert_clean()
        mask = sum(1 << i for i in range(16) if (i & 3) >= (i >> 2))
        self.assertEqual(checker.value(4), checker.xor(checker.initial[4], {0: mask}))
        for q in range(4):
            self.assertEqual(checker.value(q), checker.initial[q])
        self.assertEqual(sum(g[0] == "ccx" for g in circuit.gates), 2)

    def test_audit_strict_joint_proof_checks_second_invocation(self):
        from unittest.mock import patch

        from qpe_lab.evaluation.walk import verify_strict_paper_tables

        task = load_walk(project_root() / "benchmarks/dfthc/toy/walk-paper-strict/walk.task.json")
        circuit = build_walk(task, implementation="paper_strict")
        gates, expansion = expand_walk(task, circuit.program)
        expansion.gates = list(gates)
        erasures = [block for block in expansion.blocks if block[0] == "angle_erase"]
        self.assertEqual(len(erasures), 2)
        _, start, end = erasures[1]
        victim = next(
            i
            for i in range(start, end)
            if expansion.gates[i][0] == "cz" and isinstance(expansion.gates[i][2], dict)
        )
        expansion.gates[victim] = ("assert_zero", (expansion.registers["inner_uniform"][0],), None)
        with (
            patch(
                "qpe_lab.circuits.gadgets.synthesis.expand_walk",
                return_value=(iter(expansion.gates), expansion),
            ),
            self.assertRaises(Incorrect),
        ):
            verify_strict_paper_tables(task, circuit.program)

    def test_audit_phase_decode_accounts_for_uniform_success(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch

        from qpe_lab.cli import main

        task = load_walk(project_root() / "benchmarks/dfthc/toy/walk-paper-strict/walk.task.json")
        # The existing physical small-walk test independently checks p=15/16.
        norm2 = sum(sum(t["weight"] for t in g["terms"]) ** 2 for g in task.model["generators"]) / (
            1 << (2 * task.model["coefficient_bits"])
        )
        for phase in (0.0, math.pi / 2, math.pi):
            with self.subTest(phase=phase):
                output = io.StringIO()
                with (
                    patch(
                        "sys.argv",
                        [
                            "qpe_lab",
                            "decode-phase",
                            "--task",
                            str(task.path),
                            "--phase",
                            str(phase),
                        ],
                    ),
                    redirect_stdout(output),
                ):
                    self.assertEqual(main(), 0)
                self.assertAlmostEqual(
                    json.loads(output.getvalue())["energy"],
                    task.model["energy_offset"] + norm2 / (2 * (15 / 16)) * (1 + math.cos(phase)),
                    places=12,
                )

    def test_strict_constant_specialization_physical_walk(self):
        self.check_strict_optimized_walk("general")

    def test_strict_phase_adder_physical_walk(self):
        self.check_strict_optimized_walk("phase")

    def test_strict_boolean_phase_repair_physical_walk(self):
        self.check_strict_optimized_walk("phase", 2, 2, "zero_address", "parity", True)

    def test_strict_zero_target_specialization_without_phase_folding(self):
        from collections import Counter

        from qpe_lab.circuits.gadgets.synthesis import count_walk

        task = load_walk(project_root() / "benchmarks/dfthc/toy/walk-paper-strict/walk.task.json")
        doc = build_walk(task, implementation="paper_strict", specialize_clean_targets=True).program
        cost = count_walk(task, doc)
        self.assertNotIn("phase_fusion", cost)
        self.assertGreater(cost["zero_target_rewrites"]["replaced_blocks"], 0)
        self.assertEqual(cost["zero_target_rewrites"]["t_after"], cost["t_count"])
        compiled = list(expand_clifford_t(task, doc, cost["synthesis"]))
        self.assertEqual(dict(Counter(g[0] for g in compiled)), cost["compiled_elementary_gates"])

    def check_strict_optimized_walk(
        self,
        gradient_adder,
        phase_unlookup_bits=None,
        outer_phase_unlookup_bits=None,
        sf_lookup_control="explicit",
        fuse_phases=False,
        specialize_clean_targets=False,
    ):
        from collections import Counter

        task = load_walk(project_root() / "benchmarks/dfthc/toy/walk-paper-strict/walk.task.json")
        settings = {
            "implementation": "paper_strict",
            "outer_block_size": 2,
            "qrom_block_size": 2,
            "outer_unlookup_size": 4,
            "unlookup_block_size": 8,
        }
        for controlled in (True, False):
            with self.subTest(controlled=controlled):
                target = replace(task, controlled=controlled)
                baseline = build_walk(target, **settings)
                original, layout = expand_walk(target, baseline.program)
                counts = Counter(g[0] for g in original)
                optimized = build_walk(
                    target,
                    **settings,
                    specialize_constants=True,
                    gradient_adder=gradient_adder,
                    phase_unlookup_bits=phase_unlookup_bits,
                    outer_phase_unlookup_bits=outer_phase_unlookup_bits,
                    sf_lookup_control=sf_lookup_control,
                    fuse_phases=fuse_phases,
                    specialize_clean_targets=specialize_clean_targets,
                )
                report = self.check(optimized, target)
                self.assertEqual(report["correctness_status"], "bounded_verified")
                self.assertTrue(report["physical_dynamic_crosscheck"]["checked"])
                self.assertLess(report["physical_dynamic_crosscheck"]["max_error"], 1e-10)
                savings = 14 + (
                    8 * (target.model["spatial_modes"] - 1) if gradient_adder == "phase" else 0
                )
                if phase_unlookup_bits is None:
                    self.assertEqual(report["cost"]["discrete_toffoli"], counts["ccx"] - savings)
                else:
                    self.assertEqual(report["evidence"]["inner_phase_repair"]["plane_bits"], 2)
                self.assertLess(report["cost"]["qubits"], layout.peak_qubits)
                proof = report["evidence"]["joint_table_proof"]
                self.assertEqual(proof["invocations_verified"], 2)
                self.assertEqual(proof["sigma_values"], 1 << target.model["paper"]["alias_bits"])
                self.assertGreater(proof["metadata_bits"], 0)
                self.assertTrue(report["evidence"]["outer_table_proof"]["verified"])
                self.assertEqual(
                    report["evidence"]["outer_table_proof"]["sigma_values"],
                    1 << target.model["paper"]["alias_bits"],
                )
                self.assertEqual(
                    report["evidence"]["outer_phase_repair"]["plane_bits"],
                    outer_phase_unlookup_bits,
                )
                self.assertEqual(report["evidence"]["sf_lookup_control"]["kind"], sf_lookup_control)
                if fuse_phases:
                    fusion = report["cost"]["phase_fusion"]
                    self.assertTrue(fusion["verified"])
                    self.assertEqual(fusion["t_after"], report["cost"]["t_count"])
                    compiled = list(
                        expand_clifford_t(target, optimized.program, report["cost"]["synthesis"])
                    )
                    self.assertEqual(
                        dict(Counter(g[0] for g in compiled)),
                        report["cost"]["compiled_elementary_gates"],
                    )
                if specialize_clean_targets:
                    proof = report["cost"]["zero_target_rewrites"]
                    self.assertTrue(proof["verified"])
                    self.assertEqual(proof["initial_known_bits"], 0)
                    self.assertGreater(proof["replaced_blocks"], 0)
                    self.assertEqual(
                        proof["t_before"] - proof["t_after"], 3 * proof["replaced_blocks"]
                    )

    def test_strict_outer_alias_proof_rejects_defects(self):
        from unittest.mock import patch

        from qpe_lab.evaluation.walk import verify_strict_outer_alias

        task = load_walk(project_root() / "benchmarks/dfthc/toy/walk-paper-strict/walk.task.json")
        for planes in (None, 0, 1, 2):
            with self.subTest(planes=planes):
                doc = build_walk(
                    task,
                    implementation="paper_strict",
                    outer_block_size=2,
                    outer_unlookup_size=2,
                    outer_phase_unlookup_bits=planes,
                ).program
                proof = verify_strict_outer_alias(task, doc)
                self.assertTrue(proof["verified"])
                self.assertEqual(proof["addresses"], 1 << task.label_bits)
                self.assertEqual(proof["sigma_values"], 1 << task.model["paper"]["alias_bits"])
                _, expansion = expand_walk(task, doc)
                _, start, end = next(b for b in expansion.blocks if b[0] == "outer_adjoint")
                end = next(
                    i
                    for i in range(start, end)
                    if expansion.gates[i][0] == "h"
                    and expansion.gates[i][1][0] in expansion.registers["outer_sigma"]
                )
                victim = next(
                    i
                    for i in reversed(range(start, end))
                    if expansion.gates[i][0] in ("z", "cz")
                    and isinstance(expansion.gates[i][2], dict)
                )
                expansion.gates[victim] = (
                    "assert_zero",
                    tuple(expansion.registers["inner_uniform"]),
                    None,
                )
                with (
                    patch(
                        "qpe_lab.circuits.gadgets.synthesis.expand_walk",
                        return_value=(iter(expansion.gates), expansion),
                    ),
                    self.assertRaises(Incorrect),
                ):
                    verify_strict_outer_alias(task, doc)

    def test_strict_outer_alias_detects_nonboundary_sigma_phase(self):
        from unittest.mock import patch

        from uncompute.gadgets.arith import and_chain, release_chain

        from qpe_lab.evaluation.walk import verify_strict_outer_alias

        task = load_walk(project_root() / "benchmarks/dfthc/fe2s2/walk-paper-strict/walk.task.json")
        doc = build_walk(
            task,
            implementation="paper_strict",
            outer_block_size=2,
            outer_unlookup_size=8,
            outer_phase_unlookup_bits=0,
        ).program
        proof = verify_strict_outer_alias(task, doc)
        self.assertEqual(proof["label_sigma_combinations"], 262144)
        self.assertEqual(proof["symbolic_sigma_bits"], 6)
        bits = task.model["paper"]["alias_bits"]
        boundary = {0, (1 << bits) - 1}
        for keep in doc["alias"]["outer"]["keep"]:
            boundary.update((keep, max(0, keep - 1), min((1 << bits) - 1, keep + 1)))
        hidden = next(value for value in range(1 << bits) if value not in boundary)
        _, expansion = expand_walk(task, doc)
        injected = DynamicCircuit(expansion.peak_qubits)
        flag, trail = and_chain(
            injected,
            [(q, bool(hidden >> i & 1)) for i, q in enumerate(expansion.registers["outer_sigma"])],
        )
        injected.z(flag)
        release_chain(injected, trail)
        _, start, end = next(b for b in expansion.blocks if b[0] == "outer_adjoint")
        position = next(
            i
            for i in range(start, end)
            if expansion.gates[i][0] == "h"
            and expansion.gates[i][1][0] in expansion.registers["outer_sigma"]
        )
        # Preserve fresh outcome ids when inserting an independently built gadget.
        offset = expansion.outcomes
        extra = [
            (
                name,
                wires,
                value + offset
                if name == "hmr"
                else {"condition": value["condition"] + offset}
                if isinstance(value, dict)
                else value,
            )
            for name, wires, value in injected.gates
        ]
        expansion.gates[position:position] = extra
        expansion.peak_qubits = injected.peak_qubits
        expansion.blocks = [
            (
                name,
                start + (len(extra) if start >= position else 0),
                end + (len(extra) if end >= position else 0),
            )
            for name, start, end in expansion.blocks
        ]
        with (
            patch(
                "qpe_lab.circuits.gadgets.synthesis.expand_walk",
                return_value=(iter(expansion.gates), expansion),
            ),
            self.assertRaisesRegex(Incorrect, "phase"),
        ):
            verify_strict_outer_alias(task, doc)
        oversized = copy.deepcopy(task.model)
        oversized["paper"]["alias_bits"] = 24
        with (
            patch("qpe_lab.circuits.gadgets.synthesis.expand_walk") as expand,
            self.assertRaisesRegex(Incorrect, "symbolic-sigma limit"),
        ):
            verify_strict_outer_alias(replace(task, model=oversized), doc)
        expand.assert_not_called()

    def test_strict_joint_proof_detects_nonboundary_sigma_phase(self):
        from unittest.mock import patch

        from uncompute.gadgets.arith import and_chain, release_chain

        from qpe_lab.evaluation.walk import verify_strict_paper_tables

        task = load_walk(project_root() / "benchmarks/dfthc/fe2s2/walk-paper-strict/walk.task.json")
        doc = build_walk(
            task,
            implementation="paper_strict",
            outer_block_size=2,
            qrom_block_size=8,
            outer_unlookup_size=8,
            unlookup_block_size=32,
            specialize_constants=True,
            gradient_adder="phase",
            phase_unlookup_bits=2,
            outer_phase_unlookup_bits=0,
            sf_lookup_control="zero_address",
        ).program
        _, original = expand_walk(task, doc)
        bits = task.model["paper"]["alias_bits"]
        keep = doc["alias"]["inner"][0]["keep"][0]
        boundary = {0, (1 << bits) - 1, keep, max(0, keep - 1), min((1 << bits) - 1, keep + 1)}
        hidden = next(value for value in range(1 << bits) if value not in boundary)
        regions = [(start, end) for name, start, end in original.blocks if name == "inner_adjoint"]
        self.assertEqual(len(regions), 2)
        for start, end in regions:
            expansion = copy.deepcopy(original)
            regs = expansion.registers
            injected = DynamicCircuit(expansion.peak_qubits)
            flag, trail = and_chain(
                injected,
                [(q, False) for q in regs["term"] + regs["label"]]
                + [(q, bool(hidden >> i & 1)) for i, q in enumerate(regs["inner_sigma"])],
            )
            injected.z(flag)
            release_chain(injected, trail)
            position = next(
                i
                for i in range(start, end)
                if expansion.gates[i][0] == "h"
                and expansion.gates[i][1][0] in regs["term"] + regs["inner_sigma"]
            )
            offset = expansion.outcomes
            extra = [
                (
                    name,
                    wires,
                    value + offset
                    if name == "hmr"
                    else {"condition": value["condition"] + offset}
                    if isinstance(value, dict)
                    else value,
                )
                for name, wires, value in injected.gates
            ]
            expansion.gates[position:position] = extra
            expansion.peak_qubits = injected.peak_qubits
            expansion.blocks = [
                (
                    name,
                    lo + (len(extra) if lo >= position else 0),
                    hi + (len(extra) if hi >= position else 0),
                )
                for name, lo, hi in expansion.blocks
            ]
            with (
                patch(
                    "qpe_lab.circuits.gadgets.synthesis.expand_walk",
                    return_value=(iter(expansion.gates), expansion),
                ),
                self.assertRaisesRegex(Incorrect, "Wrong sign"),
            ):
                verify_strict_paper_tables(task, doc)
        oversized = copy.deepcopy(task.model)
        oversized["paper"]["alias_bits"] = 24
        with (
            patch("qpe_lab.circuits.gadgets.synthesis.expand_walk") as expand,
            self.assertRaisesRegex(Incorrect, "address/sigma limits"),
        ):
            verify_strict_paper_tables(replace(task, model=oversized), doc)
        expand.assert_not_called()

    def test_zero_address_sf_lookup_requires_both_corrections(self):
        from unittest.mock import patch

        from qpe_lab.evaluation.walk import verify_strict_paper_tables

        task = load_walk(project_root() / "benchmarks/dfthc/toy/walk-paper-strict/walk.task.json")
        doc = build_walk(
            task,
            implementation="paper_strict",
            specialize_constants=True,
            sf_lookup_control="zero_address",
        ).program
        self.assertTrue(verify_strict_paper_tables(task, doc)["verified"])
        _, original = expand_walk(task, doc)
        sf = original.registers["factor_metadata"][0]
        for block in ("angle_load", "angle_erase"):
            regions = [(start, end) for name, start, end in original.blocks if name == block]
            self.assertEqual(len(regions), 2)
            for start, end in regions:
                expansion = copy.deepcopy(original)
                victim = next(
                    i
                    for i in reversed(range(start, end))
                    if (
                        block == "angle_load"
                        and expansion.gates[i][0] == "cx"
                        and expansion.gates[i][1][0] == sf
                    )
                    or (
                        block == "angle_erase"
                        and expansion.gates[i][0] == "z"
                        and expansion.gates[i][1] == (sf,)
                        and isinstance(expansion.gates[i][2], dict)
                    )
                )
                expansion.gates.pop(victim)
                expansion.blocks = [
                    (name, lo - int(lo > victim), hi - int(hi > victim))
                    for name, lo, hi in expansion.blocks
                ]
                with (
                    patch(
                        "qpe_lab.circuits.gadgets.synthesis.expand_walk",
                        return_value=(iter(expansion.gates), expansion),
                    ),
                    self.assertRaises(Incorrect),
                ):
                    verify_strict_paper_tables(task, doc)

    def test_strict_specialization_option_is_bounded_and_typed(self):
        task = load_walk(project_root() / "benchmarks/dfthc/toy/walk-paper-strict/walk.task.json")
        circuit = build_walk(task, implementation="paper_strict", specialize_constants=True)
        path = self.directory / "invalid.ops"
        for bad in (1, "true", None, {}):
            corrupt = copy.deepcopy(circuit)
            corrupt.program["implementation"]["specialize_constants"] = bad
            corrupt.emit(path)
            with self.assertRaises(ValueError):
                read_walk(path)
        with self.assertRaises(ValueError):
            build_walk(self.task, specialize_constants=True)
        for bad in (1, True, None, {}, "uncontrolled"):
            corrupt = copy.deepcopy(circuit)
            corrupt.program["implementation"]["gradient_adder"] = bad
            corrupt.emit(path)
            with self.assertRaises(ValueError):
                read_walk(path)
            with self.assertRaises(ValueError):
                build_walk(task, implementation="paper_strict", gradient_adder=bad)
        with self.assertRaises(ValueError):
            build_walk(self.task, gradient_adder="phase")
        for bad in (True, None, 0, "free", {}):
            corrupt = copy.deepcopy(circuit)
            corrupt.program["implementation"]["sf_lookup_control"] = bad
            corrupt.emit(path)
            with self.assertRaises(ValueError):
                read_walk(path)
            with self.assertRaises(ValueError):
                build_walk(task, implementation="paper_strict", sf_lookup_control=bad)
        with self.assertRaises(ValueError):
            build_walk(self.task, sf_lookup_control="zero_address")
        for bad in (1, "true", None, {}):
            corrupt = copy.deepcopy(circuit)
            corrupt.program["implementation"]["fuse_phases"] = bad
            corrupt.emit(path)
            with self.assertRaises(ValueError):
                read_walk(path)
            with self.assertRaises(ValueError):
                build_walk(task, implementation="paper_strict", fuse_phases=bad)
        ideal = replace(
            task, optimization={**task.contract, "cost_model": "ideal", "walk_error_budget": 0}
        )
        for mode in (True, "affine", "parity"):
            with self.assertRaises(ValueError):
                build_walk(self.task, fuse_phases=mode)
            with self.assertRaises(ValueError):
                build_walk(ideal, implementation="paper_strict", fuse_phases=mode)
            valid = build_walk(task, implementation="paper_strict", fuse_phases=mode)
            valid.emit(path)
            self.assertEqual(read_walk(path)["implementation"]["fuse_phases"], mode)
        for bad in (1, "true", None, {}):
            corrupt = copy.deepcopy(circuit)
            corrupt.program["implementation"]["specialize_clean_targets"] = bad
            corrupt.emit(path)
            with self.assertRaises(ValueError):
                read_walk(path)
            with self.assertRaises(ValueError):
                build_walk(task, implementation="paper_strict", specialize_clean_targets=bad)
        with self.assertRaises(ValueError):
            build_walk(self.task, specialize_clean_targets=True)
        with self.assertRaises(ValueError):
            build_walk(ideal, implementation="paper_strict", specialize_clean_targets=True)
        for option in ("phase_unlookup_bits", "outer_phase_unlookup_bits"):
            for bad in (True, None, -1, 3, "2", {}):
                corrupt = copy.deepcopy(circuit)
                corrupt.program["implementation"][option] = bad
                corrupt.emit(path)
                with self.assertRaises(ValueError):
                    read_walk(path)
                if bad is not None:
                    with self.assertRaises(ValueError):
                        build_walk(task, implementation="paper_strict", **{option: bad})
            with self.assertRaises(ValueError):
                build_walk(self.task, **{option: 2})

    def test_boolean_phase_repair_exact_full_domain(self):
        from uncompute.gadgets.qrom import xor_outcomes, xor_word

        from qpe_lab.circuits.gadgets.qrom import ordered_walk, plane_phase_table

        for width, plane_bits, window in ((2, 2, 1), (4, 2, 4), (5, 2, 4), (5, 1, 8), (5, 0, 8)):
            for used in (0, 1, (1 << width) - 3, 1 << width):
                circuit = DynamicCircuit(width + 1)
                address = list(range(width))
                scratch = circuit.qubits(3)
                rng = np.random.default_rng(321)
                words = list(map(int, rng.integers(0, 8, used))) + [0] * ((1 << width) - used)
                ordered_walk(
                    circuit,
                    address,
                    len(words),
                    lambda i, ctrl, c=circuit, w=words, s=scratch: xor_word(c, ctrl, w[i], s),
                )
                circuit.cx(scratch[0], width)
                draws = [circuit.hmr(q) for q in scratch]
                for q in reversed(scratch):
                    circuit.recycle(q)
                phases = {}
                for row, word in enumerate(words):
                    terms = [draws[i] for i in range(3) if word >> i & 1]
                    # A syntactically different expression for the same parity.
                    terms += [draws[0], draws[0]]
                    phases[row] = xor_outcomes(terms)
                signs = [i for i in range(used) if i % 3 == 1]
                start = len(circuit.gates)
                plane_phase_table(
                    circuit, address, phases, window=window, plane_bits=plane_bits, signs=signs
                )
                self.assertLessEqual(
                    sum(g[0] == "oxor" for g in circuit.gates[start:]),
                    plane_bits * (1 << width) // 2,
                )

                def check(gates, width=width, address=address, signs=signs, words=words):
                    checker = SymbolicSimulator(width, 1)
                    checker.initialize([(0, address), (0, [width])])
                    checker.apply_elementary(gates)
                    sign_mask = sum(1 << i for i in signs)
                    self.assertEqual(checker.phase, {0: sign_mask} if sign_mask else {})
                    checker.phase = {}
                    checker.assert_clean()
                    flipped = sum(1 << i for i, word in enumerate(words) if word & 1)
                    expected = checker.xor(checker.initial[width], {0: flipped} if flipped else {})
                    self.assertEqual(checker.value(width), expected)
                    for q in address:
                        self.assertEqual(checker.value(q), checker.initial[q])

                check(circuit.gates)
                if used == 1 << width:
                    victim = next(
                        i
                        for i in range(start, len(circuit.gates))
                        if circuit.gates[i][0] in ("z", "cz")
                        and isinstance(circuit.gates[i][2], dict)
                    )
                    corrupt = circuit.gates[:victim] + circuit.gates[victim + 1 :]
                    with self.assertRaises((Incorrect, AssertionError)):
                        check(corrupt)

    def test_strict_select_real_width_upstream_oracle(self):
        from itertools import product
        from unittest.mock import patch

        task = load_walk(project_root() / "benchmarks/dfthc/fe2s2/walk-paper-strict/walk.task.json")
        n, modes = task.model["spatial_modes"], task.mode_count
        mask = (1 << modes) - 1
        system = np.array(
            [0, mask]
            + [1 << bit for bit in range(modes)]
            + [mask ^ (1 << bit) for bit in range(modes)],
            dtype=np.uint64,
        )
        system = np.concatenate(
            (system, np.random.default_rng(71).integers(0, 1 << modes, 64, dtype=np.uint64))
        )
        inputs = []
        # Valid loaded metadata: SF identity/nonidentity, D, and Q. The full
        # table proof establishes these invariants at every actual address.
        for sf, qtype, active in ((1, 0, 0), (1, 0, 1), (0, 0, 0), (0, 1, 0)):
            for si, so, good, alias, sign in product((0, 1), repeat=5):
                flags = (sf, qtype, active, si, so, good, 1, alias, sign)
                word = sum(value << (modes + bit) for bit, value in enumerate(flags))
                inputs.append(system | np.uint64(word))
        inputs = np.concatenate(inputs)
        sf, qtype, active, si, so, good, _, alias, sign = [
            (inputs >> (modes + bit)) & 1 for bit in range(9)
        ]
        spin = np.where(sf, si, so).astype(np.uint64)
        target = spin * n
        old = (inputs >> target) & 1
        flip = good & (sf ^ 1)
        expected = inputs ^ (flip << target)
        prefix_parity = np.zeros(len(inputs), dtype=np.uint64)
        for bit in range(n):
            prefix_parity ^= (inputs >> bit) & 1

        for specialized in (False, True):
            candidate = build_walk(
                task,
                implementation="paper_strict",
                specialize_constants=specialized,
                outer_block_size=2,
                qrom_block_size=8,
                outer_unlookup_size=8,
                unlookup_block_size=32,
            )
            # Isolate the Pauli/Boolean SELECT core by replacing only the
            # separately verified Givens network with identity in BOTH layouts.
            with patch("qpe_lab.circuits.gadgets.givens.phase_gradient_givens"):
                _, expansion = expand_walk(task, candidate.program)
            regs = expansion.registers
            logical = list(regs["system"]) + [
                regs["factor_metadata"][0],
                regs["factor_metadata"][1],
                regs["sf_active"][0],
                regs["spin_inner"][0],
                regs["spin_outer"][0],
                regs["outer_success"][0],
                regs["inner_success"][0] if regs["inner_success"] else None,
                regs["inner_flag"][0],
                regs["inner_word"][-1],
            ]
            blocks = [b for b in expansion.blocks if b[0] == "majorana_and_rotations"]
            erasures = [b for b in expansion.blocks if b[0] == "angle_erase"]
            self.assertEqual(len(blocks), 2)
            for inverse, (_, start, end) in enumerate(blocks):
                with self.subTest(specialized=specialized, inverse=bool(inverse)):
                    gates = expansion.gates[start:end]
                    active_measurement = expansion.gates[
                        erasures[inverse][1] + (n - 1) * task.model["angle_bits"]
                    ]
                    self.assertEqual(active_measurement[0], "hmr")
                    logical[modes + 2] = active_measurement[1][0]
                    mapped = {q: bit for bit, q in enumerate(logical) if q is not None}
                    for _, wires, _ in gates:
                        for q in wires:
                            if q not in mapped:
                                mapped[q] = len(logical) + sum(
                                    value >= len(logical) for value in mapped.values()
                                )
                    core = DynamicCircuit(len(logical))
                    core.gates = [
                        (name, tuple(mapped[q] for q in wires), value)
                        for name, wires, value in gates
                    ]
                    phase = (
                        (good & active & old)
                        ^ (flip & si & (old ^ (1 - inverse)))
                        ^ (good & qtype & si)
                        ^ ((sf ^ 1) & so & prefix_parity)
                        ^ (alias & sign)
                    )
                    self.check_upstream_arithmetic(
                        core, inputs, expected, sum(g[0] == "ccx" for g in gates), phase
                    )
                    if specialized and inverse == 0:
                        corrupt = copy.deepcopy(core)
                        del corrupt.gates[
                            next(
                                i
                                for i, g in enumerate(corrupt.gates)
                                if g[0] == "cz" and g[2] is None
                            )
                        ]
                        with self.assertRaises(AssertionError):
                            self.check_upstream_arithmetic(
                                corrupt,
                                inputs,
                                expected,
                                sum(g[0] == "ccx" for g in gates),
                                phase,
                            )

    def check_upstream_arithmetic(self, circuit, inputs, expected, toffoli, phase=None):
        """Run the emitted discrete gates through the pinned upstream simulator.

        Packed basis inputs do not bypass execution or cleanliness checks.
        strict=True additionally checks every measurement outcome algebraically.
        No expected values are derived from the circuit under test.
        """
        from uncompute.ops import NO_QUBIT, OpKind
        from uncompute.sim import Simulator

        ops = []
        for name, wires, value in circuit.gates:
            if name not in ("x", "cx", "ccx", "z", "cz", "hmr", "assert_zero"):
                raise ValueError(f"Unsupported arithmetic test gate: {name}")
            op = [NO_QUBIT] * 9
            op[0] = OpKind.R if name == "assert_zero" else OpKind[name.upper()]
            op[3] = wires[-1]
            if len(wires) >= 2:
                op[2] = wires[-2]
            if len(wires) == 3:
                op[1] = wires[0]
            if name == "hmr":
                op[4] = value
            elif isinstance(value, dict):
                op[5] = value["condition"]
            elif value is not None:
                raise ValueError(f"Unsupported arithmetic gate payload: {value}")
            ops.append(tuple(op))

        def packed(words, bit):
            return int.from_bytes(
                np.packbits((words >> bit) & 1, bitorder="little").tobytes(), "little"
            )

        sim = Simulator(batch=len(inputs), seed=23, strict=True)
        sim.qubits = [packed(inputs, q) for q in range(circuit.logical_qubits)]
        sim.register_qubits = set(range(circuit.logical_qubits))
        sim.apply(ops)
        sim.assert_clean(phase=0 if phase is None else packed(phase, 0))
        self.assertTrue(
            all(sim.qubits[q] == packed(expected, q) for q in range(circuit.logical_qubits)),
            "Arithmetic output or a preserved input differs from integer reference",
        )
        self.assertEqual(sim.stats.toffoli, toffoli * len(inputs))

    def test_paper_actual_width_comparator_upstream_exhaustive(self):
        # Fe2S2 uses 11-bit keep/sigma words. Include BOTH initial target bits:
        # all 2**23 inputs, not just the comparison boundaries or target |0>.
        bits, batch = 11, 1 << 17
        circuit = DynamicCircuit(2 * bits + 1)
        measured_compare(circuit, list(range(bits)), list(range(bits, 2 * bits)), 2 * bits)
        mask = (1 << bits) - 1
        for start in range(0, 1 << circuit.logical_qubits, batch):
            inputs = np.arange(start, start + batch, dtype=np.uint32)
            left, right = inputs & mask, (inputs >> bits) & mask
            expected = inputs ^ ((left >= right).astype(np.uint32) << (2 * bits))
            self.check_upstream_arithmetic(circuit, inputs, expected, bits)

    def test_paper_actual_width_adder_upstream_random_and_carries(self):
        from uncompute.gadgets.arith import add

        # This is the same controlled modular adder used by phase_gradient_givens.
        # Test the real beta=15, including the discarded overflow carry.
        bits, batch = 15, 1 << 17
        mask = (1 << bits) - 1
        circuit = DynamicCircuit(2 * bits + 1)
        add(circuit, list(range(bits)), list(range(bits, 2 * bits)), ctrl=2 * bits)

        def check(inputs):
            target, source, control = inputs & mask, (inputs >> bits) & mask, inputs >> (2 * bits)
            expected = (inputs & ~np.uint32(mask)) | ((target + source * control) & mask)
            self.check_upstream_arithmetic(circuit, inputs, expected, 2 * bits - 1)

        # Each operand/control triple is an independent uniform 31-bit draw.
        # Same random sample count as the upstream chemistry-fe2s2 grader.
        rng = np.random.default_rng(0xDF7C)
        for _ in range((1 << 24) // batch):
            check(rng.integers(0, 1 << circuit.logical_qubits, batch, dtype=np.uint32))

        # For every target word, force every carry length, wraparound, zero add,
        # and disabled control; random probes alone are unlikely to hit all.
        sources = {0, 1, mask}
        for bit in range(bits):
            sources.update((1 << bit, (1 << bit) - 1))
        target = np.arange(1 << bits, dtype=np.uint32)
        for source in sorted(sources):
            for control in (0, 1):
                check(target | (source << bits) | (control << (2 * bits)))

    def test_paper_actual_width_upstream_rejects_phase_and_output_mutations(self):
        from uncompute.sim import CleanlinessError

        bits = 11
        circuit = DynamicCircuit(2 * bits + 1)
        measured_compare(circuit, list(range(bits)), list(range(bits, 2 * bits)), 2 * bits)
        # Deterministic varied inputs include both initial target values.
        inputs = np.random.default_rng(47).integers(0, 1 << 23, 4096, dtype=np.uint32)
        mask = (1 << bits) - 1
        expected = inputs ^ (
            ((inputs & mask) >= ((inputs >> bits) & mask)).astype(np.uint32) << (2 * bits)
        )
        corrupt = copy.deepcopy(circuit)
        victim = next(
            i
            for i, (name, _, value) in enumerate(corrupt.gates)
            if name == "cz" and isinstance(value, dict)
        )
        del corrupt.gates[victim]
        with self.assertRaises(CleanlinessError):
            self.check_upstream_arithmetic(corrupt, inputs, expected, bits)
        corrupt = copy.deepcopy(circuit)
        corrupt.x(2 * bits)
        with self.assertRaisesRegex(AssertionError, "Arithmetic output"):
            self.check_upstream_arithmetic(corrupt, inputs, expected, bits)

    def test_phase_gradient_specialized_adder_upstream(self):
        from uncompute.sim import CleanlinessError

        from qpe_lab.circuits.gadgets.givens import phase_gradient_add

        def make_circuit(bits):
            circuit = DynamicCircuit(2 * bits + 1)
            phase_gradient_add(circuit, list(range(bits)), list(range(bits, 2 * bits)), 2 * bits)
            # The omitted high target is a spectator in the reduced operation.
            self.assertFalse(any(bits - 1 in wires for _, wires, _ in circuit.gates))
            return circuit

        def check(circuit, bits, inputs):
            mask = (1 << (bits - 1)) - 1
            source = (inputs >> bits) & ((1 << bits) - 1)
            control = inputs >> (2 * bits)
            total = (inputs & mask) + source * control
            expected = (inputs & ~np.uint32(mask)) | (total & mask)
            phase = (total >> (bits - 1)) & 1
            self.check_upstream_arithmetic(circuit, inputs, expected, max(0, 2 * bits - 3), phase)

        # Exhaust the reduced unitary's input basis, preserving relative phases.
        for bits in range(1, 6):
            check(
                make_circuit(bits),
                bits,
                np.arange(1 << (2 * bits + 1), dtype=np.uint32),
            )
        bits, batch = 15, 1 << 17
        circuit = make_circuit(bits)
        rng = np.random.default_rng(0xADD15)
        for _ in range((1 << 24) // batch):
            check(circuit, bits, rng.integers(0, 1 << 31, batch, dtype=np.uint32))
        # Every lower target and all carry lengths, high phase bits and controls.
        sources = {0, 1, (1 << bits) - 1}
        for bit in range(bits):
            sources.update((1 << bit, (1 << bit) - 1))
        target = np.arange(1 << bits, dtype=np.uint32)
        for source in sorted(sources):
            for control in (0, 1):
                check(circuit, bits, target | (source << bits) | (control << (2 * bits)))
        probe = rng.integers(0, 1 << 31, 4096, dtype=np.uint32)
        for measured in (False, True):
            corrupt = copy.deepcopy(circuit)
            del corrupt.gates[
                next(
                    i
                    for i, (name, _, value) in enumerate(corrupt.gates)
                    if name == "cz" and isinstance(value, dict) == measured
                )
            ]
            with self.assertRaises(CleanlinessError):
                check(corrupt, bits, probe)

    def test_phase_gradient_specialized_rotation_all_columns(self):
        for bits in (1, 2, 3):
            for inverse in (False, True):
                circuit = DynamicCircuit(bits + 2)
                gradient = circuit.qubits(bits)
                initialization = prepare_phase_gradient(circuit, gradient)
                phase_gradient_givens(
                    circuit,
                    0,
                    1,
                    list(range(2, bits + 2)),
                    gradient,
                    inverse=inverse,
                    adder="phase",
                )
                circuit.gates.extend(adjoint(initialization))
                for q in reversed(gradient):
                    circuit.recycle(q)
                dimension = 1 << circuit.logical_qubits
                columns = np.zeros((1 << circuit.peak_qubits, dimension), complex)
                columns[np.arange(dimension), np.arange(dimension)] = 1
                expected = np.zeros_like(columns)
                for angle in range(1 << bits):
                    reference = GateExpansion(2)
                    gates = list(reference.givens(0, 1, -angle if inverse else angle, bits))
                    local = StatevectorSimulator(reference.peak_qubits).run(gates, np.eye(4))
                    expected[angle * 4 : angle * 4 + 4, angle * 4 : angle * 4 + 4] = local
                draws = sum(g[0] == "hmr" for g in circuit.gates)
                for pattern in range(1 << draws):
                    actual = StatevectorSimulator(circuit.peak_qubits).run(
                        circuit.gates,
                        columns,
                        measurement_outcomes=[pattern >> bit & 1 for bit in range(draws)],
                    )
                    np.testing.assert_allclose(actual, expected, atol=1e-11)

    def test_paper_uniform_range_and_full_inverse(self):
        circuit = DynamicCircuit(3)
        uniform_range(circuit, [0, 1], 3, 2)
        for seed in (0, 1, 7):
            output = SparseStatevectorSimulator(circuit.peak_qubits).run(
                circuit.gates, {0: np.array([1.0 + 0j])}, seed=seed
            )
            for i in range(3):
                self.assertLess(abs(output[i][0] - 1 / math.sqrt(3)), 1e-12)
            self.assertLess(sum(abs(v[0]) ** 2 for i, v in output.items() if i >= 3), 1e-24)
        uniform_range(circuit, [0, 1], 3, 2, inverse=True)
        initial = dict(enumerate(np.eye(8, dtype=complex)))
        output = SparseStatevectorSimulator(circuit.peak_qubits).run(circuit.gates, initial)
        for i, column in initial.items():
            self.assertLess(np.max(np.abs(output[i] - column)), 1e-12)

    def test_paper_deferred_qroam_preserves_phase(self):
        words = [0, 3, 1, 2, 3, 2, 1, 0]
        for block in (1, 2, 4):
            circuit = DynamicCircuit(5)
            record = deferred_lookup(circuit, [0, 1, 2], [3, 4], words, block_size=block)
            circuit.z(3)
            deferred_erase(circuit, [0, 1, 2], [3, 4], record, block_size=4)
            initial = dict(enumerate(np.eye(8, dtype=complex)))
            for seed in (0, 1, 7):
                output = SparseStatevectorSimulator(circuit.peak_qubits).run(
                    circuit.gates, initial, seed=seed
                )
                for i, column in initial.items():
                    self.assertLess(
                        np.max(np.abs(output[i] - (-1) ** (words[i] & 1) * column)), 1e-12
                    )
                self.assertLess(
                    sum(np.linalg.norm(v) ** 2 for i, v in output.items() if i >= 8), 1e-24
                )
            # Prove the clean-load/use/erase contract for every address,
            # arbitrary consumer input, and EVERY measurement record.
            circuit = DynamicCircuit(6)
            record = deferred_lookup(circuit, [0, 1, 2], [3, 4], words, block_size=block)
            circuit.cx(3, 5)
            deferred_erase(circuit, [0, 1, 2], [3, 4], record, block_size=4)
            checker = SymbolicSimulator(3, 1)
            checker.initialize([(0, [0, 1, 2]), (0, [5])])
            checker.apply_elementary(circuit.gates)
            checker.assert_clean()
            mask = sum(1 << i for i, word in enumerate(words) if word & 1)
            self.assertEqual(checker.value(5), checker.xor(checker.initial[5], {0: mask}))
            checker = SymbolicSimulator(3, 1)
            checker.initialize([(0, [0, 1, 2]), (0, [5])])
            checker.apply_elementary(
                [g for g in circuit.gates if not (g[0] in ("z", "cz") and isinstance(g[2], dict))]
            )
            with self.assertRaises(Incorrect):
                checker.assert_clean()

    def test_paper_physical_walk_and_negative_targets(self):
        model = make_model(
            np.array([[[0.6, 0.8]]]),
            np.array([[[0.7]]]),
            np.array([[0.4]]),
            np.array([-0.35, 0.6]),
            np.array([[0.8, -0.6], [0.6, 0.8]]),
            energy_offset=-1.2,
            beta=2,
            coefficient_bits=8,
        )
        task = load_walk(
            write_walk_task(
                with_paper_preparation(model, 1), self.directory / "paper", "paper-small"
            )
        )
        circuit = build_walk(
            task,
            implementation="paper",
            outer_block_size=2,
            qrom_block_size=2,
            outer_unlookup_size=2,
            unlookup_block_size=4,
        )
        report = self.check(circuit, task)
        self.assertTrue(report["physical_dynamic_crosscheck"]["checked"])
        self.assertLess(report["physical_dynamic_crosscheck"]["max_error"], 1e-10)
        corrupt = copy.deepcopy(circuit)
        corrupt.program["alias"]["outer"]["alt"][0] = (
            corrupt.program["alias"]["outer"]["alt"][0] + 1
        ) % 3
        with self.assertRaises(Incorrect):
            self.check(corrupt, task)
        corrupt = copy.deepcopy(circuit)
        corrupt.program["orbital_angles"][0][0] ^= 1
        with self.assertRaises(Incorrect):
            self.check(corrupt, task)
        spec = json.loads(task.path.read_text())
        spec["output_projector"] = "term=0; retain label"
        task.path.write_text(json.dumps(spec))
        with self.assertRaises(ValueError):
            load_walk(task.path)

    def test_dynamic_comparator_all_inputs(self):
        circuit = DynamicCircuit(5)
        compare_greater_equal(circuit, [0, 1], [2, 3], 4)
        columns = np.zeros((1 << circuit.peak_qubits, 32), complex)
        columns[np.arange(32), np.arange(32)] = 1
        result = StatevectorSimulator(circuit.peak_qubits).run(circuit.gates, columns)
        expected = np.zeros_like(columns)
        for value in range(32):
            target = value ^ (16 if (value & 3) >= (value >> 2 & 3) else 0)
            expected[target, value] = 1
        self.assertLess(np.max(np.abs(result - expected)), 1e-12)

    def test_dynamic_qroam_exact_symbolic_measurement_phase(self):
        words = [0, 3, 1, 2, 2, 3, 0, 1]
        for block in (1, 2, 4):
            circuit = DynamicCircuit(5)
            dynamic_lookup(circuit, [0, 1, 2], [3, 4], words, block_size=block, unlookup_size=2)
            checker = SymbolicSimulator(3, 2)
            checker.initialize([(0, [0, 1, 2]), (0, [3, 4])])
            checker.apply_elementary(circuit.gates)
            checker.assert_clean()
            for bit, wire in enumerate((3, 4)):
                mask = sum(1 << address for address, word in enumerate(words) if word >> bit & 1)
                expected = checker.xor(checker.initial[wire], {0: mask})
                self.assertEqual(checker.value(wire), expected)
            # A second XOR with a different measurement record restores arbitrary target input.
            dynamic_lookup(circuit, [0, 1, 2], [3, 4], words, block_size=block, unlookup_size=2)
            full = SymbolicSimulator(3, 2)
            full.initialize([(0, [0, 1, 2]), (0, [3, 4])])
            full.apply_elementary(circuit.gates)
            full.assert_clean()
            self.assertEqual(full.value(3), full.initial[3])

        circuit = DynamicCircuit(5)
        dynamic_lookup(circuit, [0, 1, 2], [3, 4], words)
        dynamic_lookup(circuit, [0, 1, 2], [3, 4], words, erase=True, unlookup_size=2)
        checker = SymbolicSimulator(3, 2)
        checker.initialize([(0, [0, 1, 2]), (0, [3, 4])])
        checker.protected = {0, 1, 2}
        checker.wires[3], checker.wires[4] = {}, {}
        checker.apply_elementary(circuit.gates)
        checker.assert_clean()
        corrupt = [g for g in circuit.gates if not (g[0] in ("z", "cz") and isinstance(g[2], dict))]
        checker = SymbolicSimulator(3, 2)
        checker.initialize([(0, [0, 1, 2]), (0, [3, 4])])
        checker.protected = {0, 1, 2}
        checker.wires[3], checker.wires[4] = {}, {}
        checker.apply_elementary(corrupt)
        with self.assertRaises(Incorrect):
            checker.assert_clean()

    def test_phase_gradient_all_angles_inputs_and_measurement_records(self):
        circuit = DynamicCircuit(5)
        gradient = circuit.qubits(3)
        initialization = prepare_phase_gradient(circuit, gradient)
        phase_gradient_givens(circuit, 0, 1, [2, 3, 4], gradient)
        circuit.gates.extend(adjoint(initialization))
        for q in reversed(gradient):
            circuit.recycle(q)
        columns = np.zeros((1 << circuit.peak_qubits, 32), complex)
        columns[np.arange(32), np.arange(32)] = 1
        expected = np.zeros_like(columns)
        for angle in range(8):
            reference = GateExpansion(5)
            gates = list(reference.givens(0, 1, angle, 3))
            local = StatevectorSimulator(reference.peak_qubits).run(
                gates, np.eye(1 << reference.peak_qubits)
            )
            for state in range(4):
                expected[(angle << 2) : (angle << 2) + 4, (angle << 2) + state] = local[:4, state]
        for seed in range(4):
            actual = StatevectorSimulator(circuit.peak_qubits).run(
                circuit.gates, columns, seed=seed
            )
            self.assertLess(np.max(np.abs(actual - expected)), 1e-11)
        draws = sum(g[0] == "hmr" for g in circuit.gates)
        self.assertLessEqual(draws, 6)
        for pattern in range(1 << draws):
            actual = StatevectorSimulator(circuit.peak_qubits).run(
                circuit.gates,
                columns,
                measurement_outcomes=[pattern >> bit & 1 for bit in range(draws)],
            )
            self.assertLess(np.max(np.abs(actual - expected)), 1e-11)
        with self.assertRaises(ValueError):
            list(adjoint(circuit.gates))

    def test_alias_state_and_inverse_coherent_garbage(self):
        table = alias_table([1, 7], 2)
        self.assertEqual(alias_counts(table, 2), [1, 7])
        circuit = DynamicCircuit(1)
        sigma, word, flag = circuit.qubits(2), circuit.qubits(4), circuit.qubit()
        alias_prepare(circuit, [0], sigma, word, flag, [table])
        initial = np.zeros(1 << circuit.peak_qubits, complex)
        initial[0] = 1
        state = StatevectorSimulator(circuit.peak_qubits).run(circuit.gates, initial)
        probabilities = np.sum(np.abs(state.reshape(-1, 2)) ** 2, axis=0)
        np.testing.assert_allclose(probabilities, [1 / 8, 7 / 8], atol=1e-12)
        alias_prepare(circuit, [0], sigma, word, flag, [table], inverse=True)
        for q in reversed(sigma + word + [flag]):
            circuit.recycle(q)
        initial = np.zeros(1 << circuit.peak_qubits, complex)
        initial[0] = 1
        for seed in range(4):
            actual = StatevectorSimulator(circuit.peak_qubits).run(
                circuit.gates, initial, seed=seed
            )
            self.assertLess(np.max(np.abs(actual - initial)), 1e-11)

    def test_frozen_alias_target_and_dynamic_walk(self):
        original = toy_model()
        model = with_alias_preparation(original, 3)
        self.assertNotIn("preparation", original)
        task = load_walk(write_walk_task(model, self.directory / "alias", "alias-test"))
        tree = self.check(build_walk(task), task)
        circuit = build_walk(
            task,
            implementation="dynamic",
            prepare_method="alias",
            qrom_block_size=2,
            unlookup_block_size=2,
        )
        report = self.check(circuit, task)
        self.assertEqual(report["correctness_status"], "exact_verified")
        self.assertGreater(report["cost"]["measurements"], 0)
        self.assertEqual(report["cost"]["single_qubit_ry"], 2 * model["angle_bits"])
        self.assertEqual(
            report["numerical_crosscheck"]["target_energies"],
            tree["numerical_crosscheck"]["target_energies"],
        )
        self.assertIn("inner_sigma=0", report["evidence"]["output_projector"])
        circuit.program["alias"]["outer"]["keep"][0] ^= 1
        with self.assertRaises((Incorrect, ValueError)):
            self.check(circuit, task)
        with self.assertRaises(ValueError):
            build_walk(self.task, implementation="dynamic", prepare_method="alias")

    def test_agent_authored_qrom_programs_are_proved_and_counted(self):
        from uncompute.gadgets.qrom import erase_register

        model = with_alias_preparation(toy_model(), 3)
        task = load_walk(write_walk_task(model, self.directory / "custom", "custom"))
        circuit = build_walk(task, implementation="dynamic", prepare_method="alias")
        spec = circuit.lookup_spec("angles")
        for erase in (False, True):
            program = Builder()
            address, target = (
                program.qubits(spec["address_bits"]),
                program.qubits(spec["word_bits"]),
            )
            program.qubit_register(address)
            program.qubit_register(target)
            if erase:
                erase_register(
                    program,
                    target,
                    address,
                    len(spec["words"]),
                    spec["words"],
                    len(spec["words"]) // 2,
                    range(len(spec["words"])),
                    "custom-erase",
                )
            else:
                lookup_xor(
                    program, address, target, spec["words"], method="selectswap", block_size=2
                )
            circuit.lookup_program("angles", program, erase=erase)
        report = self.check(circuit, task)
        proofs = report["cost"]["component_proofs"]
        self.assertEqual(len(proofs), 2)
        self.assertTrue(all(p["evidence"]["addresses"] == len(spec["words"]) for p in proofs))
        bad = Builder()
        address, target = bad.qubits(spec["address_bits"]), bad.qubits(spec["word_bits"])
        bad.qubit_register(address)
        bad.qubit_register(target)
        lookup_xor(bad, address, target, spec["words"])
        bad.z(address[0])
        circuit.lookup_program("angles", bad)
        with self.assertRaises(Incorrect):
            self.check(circuit, task)

    def test_physical_dynamic_walk_including_control_and_alias_garbage(self):
        model = toy_model()
        model["angle_bits"], model["coefficient_bits"] = 3, 3
        model["orbital_angles"] = [[1], [3]]
        model["generators"] = [
            {
                "kind": "SF",
                "terms": [
                    {"weight": 3, "orbital": 0, "spin": 0, "pauli": "Z", "phase": 0},
                    {"weight": 1, "orbital": 1, "spin": 1, "pauli": "Z", "phase": 2},
                ],
            },
            {
                "kind": "D",
                "terms": [
                    {"weight": 2, "orbital": 1, "spin": 1, "pauli": "X", "phase": 0},
                    {"weight": 2, "orbital": 1, "spin": 1, "pauli": "Y", "phase": 1},
                ],
            },
        ]
        model = with_alias_preparation(model, 1)
        task = load_walk(write_walk_task(model, self.directory / "physical", "physical"))
        circuit = build_walk(task, implementation="dynamic", prepare_method="alias")
        report = self.check(circuit, task)
        self.assertTrue(report["physical_dynamic_crosscheck"]["checked"])
        self.assertLess(report["physical_dynamic_crosscheck"]["max_error"], 1e-10)
        contract = dict(task.contract, cost_model="clifford_t", walk_error_budget=1e-4)
        compiled_report = self.check(circuit, replace(task, optimization=contract))
        compiled = list(
            expand_clifford_t(task, circuit.program, compiled_report["cost"]["synthesis"])
        )
        from collections import Counter

        from qpe_lab.circuits.sim import SparseStatevectorSimulator

        self.assertEqual(
            dict(Counter(g[0] for g in compiled)),
            {k: v for k, v in compiled_report["cost"]["compiled_elementary_gates"].items() if v},
        )
        # beta=3 uses exact Clifford+T phases. All coherent system/control inputs
        # share one trajectory normalization, so relative phases remain tested.
        start = task.label_bits + task.term_bits
        indices = [i << start for i in range(1 << (task.mode_count + 1))]
        initial = dict(zip(indices, np.eye(len(indices)), strict=True))
        simulator = SparseStatevectorSimulator(compiled_report["cost"]["qubits"])
        expected = simulator.run(circuit.elementary_gates(), initial)
        actual = simulator.run(compiled, initial)
        self.assertLess(
            max(
                float(np.max(np.abs(actual.get(i, 0) - expected.get(i, 0))))
                for i in actual.keys() | expected.keys()
            ),
            1e-10,
        )

    def test_full_rectangular_encoding_and_walk_for_both_baselines(self):
        costs = []
        for grouped in (False, True):
            report = self.check(build_walk(self.task, grouped=grouped))
            self.assertEqual(report["correctness_status"], "exact_verified")
            self.assertEqual(report["cost"]["status"], "partial")
            self.assertIsNone(report["cost"]["toffoli"])
            self.assertEqual(report["numerical_crosscheck"]["system_basis_states"], 16)
            self.assertLess(
                report["numerical_crosscheck"]["spectral_subspace_max_abs_error"], 1e-12
            )
            self.assertEqual(report["evidence"]["output_projector"], "term=0; label retained")
            costs.append(report["cost"])
        self.assertLess(costs[1]["single_qubit_ry"], costs[0]["single_qubit_ry"])
        self.assertEqual(costs[1]["discrete_toffoli"], costs[0]["discrete_toffoli"])

    def test_bitwise_rotations_and_group_permutation_preserve_target(self):
        circuit = build_walk(self.task, bitwise_rotations=True)
        circuit.program["select"].reverse()
        for group in circuit.program["select"]:
            group["branches"].reverse()
        report = self.check(circuit)
        self.assertEqual(report["correctness_status"], "exact_verified")

    def test_different_preparation_tree_and_disjoint_schedule_are_accepted(self):
        circuit = build_walk(
            self.task,
            outer_order=list(range(self.task.label_bits)),
            inner_order=list(range(self.task.term_bits)),
        )
        # Reverse independent label blocks without changing the quantum state.
        circuit.program["inner"].sort(key=lambda g: -g["label"])
        for gate in circuit.program["outer"]:
            gate["left"] *= 3
            gate["right"] *= 3
        report = self.check(circuit)
        self.assertLess(report["numerical_crosscheck"]["block_max_abs_error"], 1e-12)
        circuit.program["inner"][0]["controls"].append([self.task.term_bits, 0])
        with self.assertRaisesRegex(Incorrect, "PREPARE"):
            self.check(circuit)

    def test_qrom_walk_and_width_rejection(self):
        report = self.check(build_walk(self.task, select_method="qrom"))
        self.assertEqual(report["correctness_status"], "exact_verified")
        self.assertGreater(report["cost"]["workspace_qubits"], 10)
        with self.assertRaisesRegex(ValueError, "qubit budget"):
            self.check(
                build_walk(self.task, select_method="qrom"), task=replace(self.task, max_qubits=20)
            )

    def test_clifford_t_complete_walk_count_and_export_match(self):
        # Small angles/distribution keep this an inexpensive but full walk test.
        model = toy_model()
        model["angle_bits"] = 1
        model["orbital_angles"] = [[1] for _ in model["orbital_angles"]]
        model["generators"] = [model["generators"][2], model["generators"][5]]
        contract = {
            "cost_model": "clifford_t",
            "walk_error_budget": 1e-4,
            "objective": "t_count",
            "synthesis_backend": "pygridsynth-2.0.0",
            "ccx_decomposition": "7t-no-ancilla",
        }
        task = load_walk(write_walk_task(model, self.directory / "ft", "ft", optimization=contract))
        circuit = build_walk(task)
        report = self.check(circuit, task=task)
        self.assertEqual(report["correctness_status"], "bounded_verified")
        cost = report["cost"]
        self.assertEqual(cost["status"], "complete")
        from collections import Counter

        compiled = list(expand_clifford_t(task, circuit.program, cost["synthesis"]))
        self.assertEqual(
            dict(Counter(g[0] for g in compiled)),
            {k: v for k, v in cost["compiled_elementary_gates"].items() if v},
        )
        self.assertLess(cost["synthesis"]["walk_operator_error_bound"], 1e-4)
        # Every logical column including disabled control and complex phases.
        state = np.zeros((1 << cost["qubits"], 1 << task.logical_qubits), dtype=complex)
        state[: 1 << task.logical_qubits] = np.eye(1 << task.logical_qubits)
        sim = StatevectorSimulator(cost["qubits"])
        actual = sim.run(compiled, state)
        from qpe_lab.circuits.checks import verify_phase_fusion
        from qpe_lab.circuits.gadgets.synthesis import fuse_diagonal_phases

        for affine, complements in ((False, False), (True, False), (True, True)):
            fused = list(fuse_diagonal_phases(compiled, affine=affine, complements=complements))
            self.assertTrue(
                verify_phase_fusion(compiled, fused, affine=affine, complements=complements)[
                    "verified"
                ]
            )
            np.testing.assert_allclose(sim.run(fused, state), actual, atol=3e-13, rtol=0)
        expected = sim.run(expand_walk(task, circuit.program)[0], state)
        self.assertLess(
            np.linalg.norm(actual - expected, ord=2), cost["synthesis"]["walk_operator_error_bound"]
        )
        np.testing.assert_allclose(
            actual[:, : 1 << (task.logical_qubits - 1)],
            state[:, : 1 << (task.logical_qubits - 1)],
            atol=2e-13,
        )

    def test_compiled_phase_fusion_rejects_unallocated_wire(self):
        from unittest.mock import patch

        from qpe_lab.circuits.gadgets.synthesis import count_walk, fuse_diagonal_phases

        task = load_walk(project_root() / "benchmarks/dfthc/toy/walk-paper-strict/walk.task.json")
        doc = build_walk(task, implementation="paper_strict", fuse_phases="parity").program
        base = count_walk(task, doc)
        bad_wire = base["qubits"] + 3

        def altered(gates, **kwargs):
            yield from fuse_diagonal_phases(gates, **kwargs)
            yield "t", (bad_wire,), None
            yield "tdg", (bad_wire,), None

        with (
            patch("qpe_lab.circuits.gadgets.synthesis.fuse_diagonal_phases", altered),
            self.assertRaisesRegex(ValueError, "outside its reported allocation"),
        ):
            count_walk(task, doc)

    def test_uncontrolled_walk_retains_both_reflection_global_signs(self):
        task = replace(self.task, controlled=False)
        report = self.check(build_walk(task), task=task)
        self.assertIsNone(report["numerical_crosscheck"]["disabled_max_abs_error"])
        self.assertEqual(report["cost"]["elementary_gates"]["global_phase"], 2)
        expansion = GateExpansion(3)
        gates = list(expansion.phase([])) + list(expansion.phase([(0, 0), (1, 0)]))
        state = np.zeros((16, 8), dtype=complex)
        state[:8] = np.eye(8)
        expected = state.copy()
        for column in range(8):
            expected[:, column] *= 1 if column & 3 == 0 else -1
        np.testing.assert_allclose(StatevectorSimulator(4).run(gates, state), expected, atol=1e-14)

    def test_select_defects_fail_including_relative_i_phase(self):
        for defect in ("missing", "duplicate", "phase", "orbital", "spin", "padding"):
            circuit = build_walk(self.task)
            groups = circuit.program["select"]
            group = next(g for g in groups if g["orbital"] >= 0)
            branch = group["branches"][0]
            if defect == "missing":
                group["branches"].pop()
            elif defect == "duplicate":
                group["branches"].append(copy.deepcopy(branch))
            elif defect == "phase":
                branch["phase"] = (branch["phase"] + 1) % 4
            elif defect == "orbital":
                group["orbital"] += 1
            elif defect == "spin":
                group["spin"] ^= 1
            else:
                branch["label"] = (1 << self.task.label_bits) - 1
            with self.subTest(defect=defect), self.assertRaisesRegex(Incorrect, "SELECT"):
                self.check(circuit)

    def test_wrong_preparation_adjoint_and_rectangular_reflection_rejected(self):
        for defect in ("outer", "inner", "adjoint", "projector", "rotation", "hash"):
            circuit = build_walk(self.task)
            if defect in ("outer", "inner"):
                circuit.program[defect][0]["right"] += 1
            elif defect == "adjoint":
                circuit.program["steps"][6]["adjoint"] = False
            elif defect == "projector":
                circuit.program["steps"][4]["block"] = "rin"
            elif defect == "rotation":
                circuit.program["orbitals"][0][0][2] *= -1
            else:
                circuit.program["model_sha256"] = "0" * 64
            with self.subTest(defect=defect), self.assertRaises(Incorrect):
                self.check(circuit)

    def test_qubit_budget_includes_explicit_equality_workspace(self):
        with self.assertRaisesRegex(ValueError, "qubit budget"):
            self.check(
                build_walk(self.task), task=replace(self.task, max_qubits=self.task.logical_qubits)
            )

    def test_artifact_rejects_unknown_cost_fields_duplicate_keys_and_boolean_indices(self):
        path = self.directory / "bad.ops"
        path.write_bytes(WALK_MAGIC + b'{"schema":"qpe.walk-circuit.v1","schema":"forged"}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_walk(path)
        for defect in ("cost", "bool"):
            circuit = build_walk(self.task)
            if defect == "cost":
                circuit.program["toffoli"] = 0
            else:
                circuit.program["outer"][0]["target"] = True
            with self.assertRaises(ValueError):
                self.check(circuit)

    def test_sos_is_hermitian_positive_and_number_conserving_with_complex_generators(self):
        generators = reference_generators(self.task.model)
        h = sum(o.conj().T @ o for o in generators)
        number = np.diag([state.bit_count() for state in range(16)])
        np.testing.assert_allclose(h, h.conj().T, atol=1e-14)
        np.testing.assert_allclose(h @ number, number @ h, atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(h).min(), -1e-13)
        # A^dag A is the sum of individual squares, never the square of the sum.
        self.assertGreater(np.linalg.norm(h - sum(generators).conj().T @ sum(generators)), 0.1)

    def test_every_elementary_walk_column_matches_macro_simulator_and_clean_scratch(self):
        # Two non-Hermitian one-body generators on different spin blocks.
        model = toy_model()
        model["generators"] = [model["generators"][2], model["generators"][5]]
        task = load_walk(write_walk_task(model, self.directory / "compact", "compact"))
        circuit = build_walk(task)
        report = self.check(circuit, task=task)
        gates, expansion = expand_walk(task, circuit.program)
        gates = list(gates)
        logical, physical = task.logical_qubits, expansion.peak_qubits
        state = np.zeros((1 << physical, 1 << logical), dtype=complex)
        state[: 1 << logical] = np.eye(1 << logical)
        actual = StatevectorSimulator(physical).run(gates, state)
        macro = SOSSimulator(task, circuit.program)
        base_dimension = 1 << (logical - 1)
        full_input = np.zeros(
            (macro.labels, macro.terms, macro.dimension, base_dimension), dtype=complex
        )
        for label in range(macro.labels):
            for term in range(macro.terms):
                for system in range(macro.dimension):
                    index = (
                        label
                        | (term << task.label_bits)
                        | (system << (task.label_bits + task.term_bits))
                    )
                    full_input[label, term, system, index] = 1
        expected = np.zeros_like(actual)
        for enable in (0, 1):
            output = macro.run(full_input, enable=bool(enable))
            for label in range(macro.labels):
                for term in range(macro.terms):
                    for system in range(macro.dimension):
                        index = (
                            label
                            | (term << task.label_bits)
                            | (system << (task.label_bits + task.term_bits))
                            | (enable << (logical - 1))
                        )
                        expected[index, enable * base_dimension : (enable + 1) * base_dimension] = (
                            output[label, term, system]
                        )
        np.testing.assert_allclose(actual, expected, atol=3e-13)
        self.assertEqual(report["cost"]["qubits"], physical)
        np.testing.assert_allclose(actual[1 << logical :], 0, atol=3e-13)

    def test_task_rejects_tampering_and_bad_projectors(self):
        spec = json.loads(self.task.path.read_text())
        spec["output_projector"] = "label=0,term=0"
        self.task.path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, "projector"):
            load_walk(self.task.path)
        spec["output_projector"] = "term=0; retain label"
        model_path = self.directory / "sos-model.json"
        model_path.write_text(model_path.read_text() + " ")
        self.task.path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, "hash"):
            load_walk(self.task.path)


class ElementaryTests(unittest.TestCase):
    def test_zero_target_exact_local_columns_and_relative_phases(self):
        from qpe_lab.circuits.gadgets.synthesis import zero_target_ccx_clifford_t
        from qpe_lab.circuits.sim import Unsupported, exact_clifford_t_columns

        def truth(columns):
            return tuple(
                tuple(
                    (1, 0, 0, 0) if row == (col ^ 4 if col & 3 == 3 else col) else (0, 0, 0, 0)
                    for col in columns
                )
                for row in range(8)
            )

        reference = list(ccx_clifford_t(0, 1, 2))
        replacement = list(zero_target_ccx_clifford_t(0, 1, 2))
        self.assertEqual(exact_clifford_t_columns(reference, 3, range(8)), truth(range(8)))
        self.assertEqual(exact_clifford_t_columns(replacement, 3, range(4)), truth(range(4)))
        self.assertNotEqual(exact_clifford_t_columns(replacement, 3, range(8)), truth(range(8)))
        # Cross-check the exact arithmetic against the independent numerical
        # simulator for every supported gate on mixed random local circuits.
        rng = np.random.default_rng(93271)
        choices = [
            (name, (q,), None) for name in ("h", "x", "t", "tdg", "s", "sdg", "z") for q in range(3)
        ]
        choices += [
            (name, (a, b), None)
            for name in ("cx", "cz")
            for a in range(3)
            for b in range(3)
            if a != b
        ]
        omega = np.exp(1j * math.pi / 4)
        for _ in range(8):
            word = [choices[i] for i in rng.integers(len(choices), size=24)]
            exact = exact_clifford_t_columns(word, 3, range(8))
            numeric = np.array(
                [
                    [sum(float(c) * omega**i for i, c in enumerate(value)) for value in row]
                    for row in exact
                ]
            )
            np.testing.assert_allclose(
                numeric, StatevectorSimulator(3).run(word, np.eye(8)), atol=3e-14, rtol=0
            )
        for broken in (replacement[:-1], replacement[:-1] + [("s", (2,), None)]):
            self.assertNotEqual(exact_clifford_t_columns(broken, 3, range(4)), truth(range(4)))
            # Both errors preserve every output bit: a truth-table test misses them.
            matrix = StatevectorSimulator(3).run(broken, np.eye(8)[:, :4])
            self.assertEqual(np.argmax(np.abs(matrix), axis=0).tolist(), [0, 1, 2, 7])
        for gates, width, columns in (
            ([("h", (0,), None)] * 129, 1, [0]),
            ([], 5, [0]),
            ([("t", (0,), {"condition": 0})], 1, [0]),
            ([("ry", (0,), 0.3)], 1, [0]),
        ):
            with self.assertRaises(Unsupported):
                exact_clifford_t_columns(gates, width, columns)
        for gates, columns in (
            ([("cx", (0, 0), None)], [0]),
            ([("h", (3,), None)], [0]),
            ([("h", (True,), None)], [0]),
            ([], [0, 0]),
            ([], [True]),
            ([], []),
        ):
            with self.assertRaises(Incorrect):
                exact_clifford_t_columns(gates, 3, columns)

    def test_zero_target_reset_facts_and_forged_witnesses(self):
        from qpe_lab.circuits.checks import verify_zero_target_rewrites
        from qpe_lab.circuits.gadgets.synthesis import (
            specialize_zero_targets,
            zero_target_ccx_clifford_t,
        )

        reset = [("hmr", (2,), 0)]
        cases = [
            ([], False),
            ([("assert_zero", (2,), None)], False),
            (reset, True),
            (reset + [("h", (2,), None)], False),
            (reset + [("x", (2,), None)], False),
            (reset + [("x", (2,), {"condition": 0})], False),
            (reset + [("h", (2,), {"condition": 0})], False),
            (reset + [("cx", (0, 2), None)], False),
            (reset + [("cx", (0, 2), {"condition": 0})], False),
            (reset + [("x", (2,), None)] * 2, True),
            (reset + [("t", (2,), None), ("assert_zero", (2,), None)], True),
            (reset + [("hmr", (0,), 1), ("cx", (0, 2), None)], True),
            (reset + [("hmr", (0,), 1), ("cx", (0, 2), {"condition": 0})], True),
            (reset + [("hmr", (0,), 1), ("x", (0,), None), ("cx", (0, 2), None)], False),
            (
                reset + [("hmr", (0,), 1), ("x", (0,), None)] + [("cx", (0, 2), None)] * 2,
                True,
            ),
        ]
        reference = list(ccx_clifford_t(0, 1, 2))
        short = list(zero_target_ccx_clifford_t(0, 1, 2))
        for prefix, eligible in cases:
            with self.subTest(prefix=prefix):
                source = prefix + reference
                after, witness = specialize_zero_targets(source)
                self.assertEqual(witness, [(len(prefix), 0, 1, 2, eligible)])
                self.assertEqual(after, prefix + (short if eligible else reference))
                proof = verify_zero_target_rewrites(source, after, witness)
                self.assertEqual(proof["replaced_blocks"], int(eligible))
                if not eligible:
                    with self.assertRaisesRegex(Incorrect, "not proven zero"):
                        verify_zero_target_rewrites(
                            source, prefix + short, [(len(prefix), 0, 1, 2, True)]
                        )
        # A known-zero control makes both consecutive CCX targets remain zero.
        source = reset + [("hmr", (0,), 1)] + reference * 2
        after, witness = specialize_zero_targets(source)
        self.assertEqual(verify_zero_target_rewrites(source, after, witness)["replaced_blocks"], 2)
        # Physical line numbers need not be contiguous or start at zero.
        source = [("hmr", (9,), 0)] + list(ccx_clifford_t(3, 5, 9))
        after, witness = specialize_zero_targets(source)
        self.assertTrue(verify_zero_target_rewrites(source, after, witness)["verified"])

    def test_zero_target_certificate_rejects_phase_and_coverage_mutations(self):
        from qpe_lab.circuits.checks import verify_zero_target_rewrites
        from qpe_lab.circuits.gadgets.synthesis import specialize_zero_targets

        before = [("hmr", (2,), 0)] + list(ccx_clifford_t(0, 1, 2))
        after, witness = specialize_zero_targets(before)
        self.assertTrue(verify_zero_target_rewrites(before, after, witness)["verified"])
        corrupt_reference = list(before)
        corrupt_reference[3] = ("t", (2,), None)
        for source, result, trace in (
            (before, after[:-1], witness),
            (before, after[:-1] + [("s", (2,), None)], witness),
            (corrupt_reference, after, witness),
            (before, after, []),
            (before, after, [(0, 0, 1, 2, True)]),
            (before, after, witness * 2),
            (before, after, [(1, 0, 2, 1, True)]),
            (before, after, [(1, 0, 1, 2, False)]),
            (before, after, [(1, 0, 1, 2, 1)]),
            (before, after + [("x", (0,), None)], witness),
            (before, [("hmr", (2,), 4)] + after[1:], witness),
            (before, after[:-1] + [("sdg", (3,), None)], witness),
            (before, after[:-1] + [("sdg", (2,), {"condition": 0})], witness),
            ([("assert_zero", (2,), None)] + before[1:], after, witness),
        ):
            with self.subTest(result=result, trace=trace), self.assertRaises(Incorrect):
                verify_zero_target_rewrites(source, result, trace)

    def test_zero_target_all_kraus_branches_and_phase_fusion(self):
        from itertools import product

        from qpe_lab.circuits.checks import verify_phase_fusion, verify_zero_target_rewrites
        from qpe_lab.circuits.gadgets.synthesis import fuse_diagonal_phases, specialize_zero_targets

        source = [
            ("h", (0,), None),
            ("cx", (0, 2), None),
            ("t", (2,), None),
            ("cz", (1, 2), None),
            ("hmr", (2,), 0),
        ] + list(ccx_clifford_t(0, 1, 2))
        source += [
            ("cz", (1, 2), None),
            ("t", (2,), None),
            ("hmr", (2,), 1),
            ("cx", (0, 1), {"condition": 0}),
        ] + list(ccx_clifford_t(0, 1, 2))
        source += [
            ("cz", (0, 2), {"condition": 1}),
            ("hmr", (2,), 2),
            ("oxor", (), [3, 0, 2]),
            ("s", (1,), {"condition": 3}),
            ("assert_zero", (2,), None),
        ]
        rewritten, witness = specialize_zero_targets(source)
        self.assertEqual(
            verify_zero_target_rewrites(source, rewritten, witness)["replaced_blocks"], 2
        )
        folded = list(fuse_diagonal_phases(rewritten, affine=True, complements=True))
        self.assertTrue(
            verify_phase_fusion(rewritten, folded, affine=True, complements=True)["verified"]
        )
        completeness = np.zeros((8, 8), dtype=complex)
        for branch in product((0, 1), repeat=3):
            expected, record = measurement_kraus(source, 3, branch)
            for result in (rewritten, folded):
                actual, actual_record = measurement_kraus(result, 3, branch)
                self.assertEqual(record, actual_record)
                np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=0)
            completeness += actual.conj().T @ actual
        np.testing.assert_allclose(completeness, np.eye(8), atol=3e-14, rtol=0)

    def test_coherent_qrom_all_addresses_arbitrary_target_and_clean_workspace(self):
        words = [0, 5, 5, 7, 1, 0, 6, 3]
        for direct, block in ((False, 1), (True, 1), (False, 2), (False, 4)):
            expansion = GateExpansion(6)
            gates = list(
                coherent_lookup(
                    expansion, range(3), range(3, 6), words, direct=direct, block_size=block
                )
            )
            state = np.zeros((1 << expansion.peak_qubits, 64), dtype=complex)
            state[:64] = np.eye(64)
            expected = np.zeros_like(state)
            for column in range(64):
                expected[column ^ (words[column & 7] << 3), column] = 1
            np.testing.assert_allclose(
                StatevectorSimulator(expansion.peak_qubits).run(gates, state), expected, atol=1e-14
            )

    def test_qrom_select_all_branch_and_fock_inputs_against_independent_paulis(self):
        with tempfile.TemporaryDirectory() as folder:
            model = toy_model()
            model["angle_bits"] = 3
            model["orbital_angles"] = [[1] for _ in model["orbital_angles"]]
            model["generators"] = [model["generators"][2], model["generators"][5]]
            task = load_walk(write_walk_task(model, folder, "qrom-select"))
            artifact = build_walk(task, select_method="qrom").program
            expansion = GateExpansion(task.logical_qubits)
            gates = list(qrom_select(expansion, task, artifact, [0], [1], 2))
            reference = SOSSimulator(task, artifact)
            for label, term, operator in reference.selected:
                state = np.zeros((1 << expansion.peak_qubits, 16), dtype=complex)
                indices = label | (term << 1) | (np.arange(16) << 2)
                state[indices] = np.eye(16)
                expected = np.zeros_like(state)
                expected[indices] = operator
                np.testing.assert_allclose(
                    StatevectorSimulator(expansion.peak_qubits).run(gates, state),
                    expected,
                    atol=1e-13,
                )

    def test_actual_rotation_synthesis_phase_sensitive_and_adjoint(self):
        compiled = compile_rotations({0.3: 2, -0.3: 2}, 1e-4)
        entry = compiled["library"][(0.3).hex()]
        gates, phase = rotation_sequence(entry, 0.3, 0)
        gates += [("global_phase", (), phase * math.pi / 4)]
        actual = StatevectorSimulator(1).run(gates, np.eye(2))
        expected = StatevectorSimulator(1).run([("ry", (0,), 0.3)], np.eye(2))
        self.assertLess(np.linalg.norm(actual - expected), 1e-4 / 4)
        np.testing.assert_allclose(
            StatevectorSimulator(1).run(adjoint(gates), actual), np.eye(2), atol=1e-13
        )
        wrong = dict(entry, phase=(entry["phase"] + 1) % 8)
        self.assertGreater(rotation_residual(0.3, wrong), 0.5)
        np.testing.assert_allclose(
            StatevectorSimulator(3).run(ccx_clifford_t(0, 1, 2), np.eye(8)),
            StatevectorSimulator(3).run([("ccx", (0, 1, 2), None)], np.eye(8)),
            atol=1e-14,
        )

    def test_phase_fusion_exact_local_rules_and_barriers(self):
        from itertools import product

        from qpe_lab.circuits.checks import verify_phase_fusion
        from qpe_lab.circuits.gadgets.synthesis import fuse_diagonal_phases

        sim = StatevectorSimulator(2)
        middle = [("cx", (0, 1), None), ("cx", (1, 0), None), ("cz", (0, 1), None)]
        middle += [(name, (q,), None) for name in ("h", "x") for q in (0, 1)]
        for gate, left, right in product(middle, range(8), range(8)):
            before = [("t", (0,), None)] * left + [gate] + [("t", (0,), None)] * right
            after = list(fuse_diagonal_phases(before))
            self.assertTrue(verify_phase_fusion(before, after)["verified"])
            np.testing.assert_allclose(
                sim.run(before, np.eye(4)), sim.run(after, np.eye(4)), atol=3e-14
            )
        for gate in (("h", (0,), None), ("x", (0,), None), ("cx", (1, 0), None), ("hmr", (0,), 0)):
            before = [("t", (0,), None), gate, ("tdg", (0,), None)]
            with self.assertRaises(Incorrect):
                verify_phase_fusion(before, [gate])
        for bad in (
            [("t", (0, 1), None)],
            [("ry", (0,), 0.3)],
            [("hmr", (0,), 0), ("hmr", (0,), 0)],
            [("z", (0,), {"condition": 0})],
            [("oxor", (), [0, -1, 1])],
            [("t", (0,), 0.5)],
            [("hmr", (0,), 0), ("t", (0,), {"condition": False})],
            [("hmr", (0,), 0), ("oxor", (), [1, False, 0])],
        ):
            with self.assertRaises(Incorrect):
                verify_phase_fusion(bad, bad)

    def test_phase_fusion_all_measurement_kraus_branches(self):
        from itertools import product

        from qpe_lab.circuits.checks import verify_phase_fusion
        from qpe_lab.circuits.gadgets.synthesis import fuse_diagonal_phases

        before = [
            ("cx", (0, 2), None),
            ("t", (0,), None),
            ("cz", (1, 2), None),
            ("tdg", (2,), None),
            ("hmr", (2,), 0),
            ("oxor", (), [1, 0, 0]),
            ("z", (0,), {"condition": 0}),
            ("t", (0,), {"condition": 0}),
            ("tdg", (0,), None),
            ("assert_zero", (2,), None),
            ("x", (2,), None),
            ("s", (2,), None),
            ("t", (1,), None),
            ("hmr", (2,), 2),
            ("oxor", (), [3, 0, 2]),
            ("cz", (0, 1), {"condition": 3}),
            ("tdg", (1,), None),
            ("assert_zero", (2,), None),
        ]
        after = list(fuse_diagonal_phases(before))
        self.assertTrue(verify_phase_fusion(before, after)["verified"])

        def kraus(gates, branch):
            return measurement_kraus(gates, 3, branch)

        completeness = np.zeros((8, 8), dtype=complex)
        for branch in product((0, 1), repeat=2):
            expected, old_record = kraus(before, branch)
            actual, new_record = kraus(after, branch)
            self.assertEqual(old_record, new_record)
            np.testing.assert_allclose(actual, expected, atol=3e-14)
            completeness += actual.conj().T @ actual
        np.testing.assert_allclose(completeness, np.eye(8), atol=3e-14)
        broken = copy.deepcopy(after)
        victim = next(i for i, g in enumerate(broken) if g[0] == "cz" and isinstance(g[2], dict))
        broken[victim] = ("cz", (0, 1), {"condition": 1})
        with self.assertRaises(Incorrect):
            verify_phase_fusion(before, broken)
        with self.assertRaises(Incorrect):
            verify_phase_fusion(before, after + [("t", (0,), None)])

        # Copy a basis value, then change the original wire's basis. Equal
        # phases on different wires still cancel in every measurement branch.
        paths = [
            ("hmr", (2,), 0),
            ("cx", (0, 2), None),
            ("t", (0,), None),
            ("h", (0,), None),
            ("tdg", (2,), None),
            ("x", (1,), {"condition": 0}),
            ("t", (1,), None),
            ("h", (1,), {"condition": 0}),
            ("tdg", (1,), None),
            ("cx", (0, 1), {"condition": 0}),
            ("s", (1,), None),
            ("hmr", (2,), 2),
            ("oxor", (), [3, 0, 2]),
            ("t", (0,), {"condition": 3}),
            ("z", (0,), {"condition": 0}),
            ("t", (2,), None),  # Reset zero: exactly the identity.
            ("x", (2,), None),
            ("t", (2,), None),  # Constant one: a scalar phase that must remain.
            ("x", (2,), None),
            ("assert_zero", (2,), None),
        ]
        for source in (before, paths):
            for complements in (False, True):
                folded = list(fuse_diagonal_phases(source, affine=True, complements=complements))
                self.assertTrue(
                    verify_phase_fusion(source, folded, affine=True, complements=complements)[
                        "verified"
                    ]
                )
                completeness = np.zeros((8, 8), dtype=complex)
                for branch in product((0, 1), repeat=2):
                    expected, old_record = kraus(source, branch)
                    actual, new_record = kraus(folded, branch)
                    self.assertEqual(old_record, new_record)
                    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=0)
                    completeness += actual.conj().T @ actual
                np.testing.assert_allclose(completeness, np.eye(8), atol=3e-14, rtol=0)
        folded = list(fuse_diagonal_phases(paths, affine=True))
        scalar = next(i for i, g in enumerate(folded) if g == ("t", (2,), None))
        with self.assertRaises(Incorrect):
            verify_phase_fusion(paths, folded[:scalar] + folded[scalar + 1 :], affine=True)

    def test_affine_phase_folding_unitary_paths_and_negative_controls(self):
        from itertools import product

        from qpe_lab.circuits.checks import verify_phase_fusion
        from qpe_lab.circuits.gadgets.synthesis import fuse_diagonal_phases

        rng = np.random.default_rng(7741)
        choices = [
            (name, (q,), None) for name in ("t", "tdg", "s", "sdg", "z", "h", "x") for q in range(3)
        ]
        choices += [
            (name, (a, b), None)
            for name in ("cx", "cz")
            for a, b in product(range(3), repeat=2)
            if a != b
        ]
        samples = [
            [choices[int(i)] for i in rng.integers(len(choices), size=80)] for _ in range(64)
        ]
        # Return to the same parity after CX target changes, including X offsets.
        for offset in ([], [("x", (0,), None)]):
            for left, right in product(range(8), repeat=2):
                samples.append(
                    offset
                    + [("t", (0,), None)] * left
                    + [("cx", (1, 0), None), ("s", (0,), None), ("cx", (1, 0), None)]
                    + [("t", (0,), None)] * right
                )
        simulator = StatevectorSimulator(3)
        for source in samples:
            for complements in (False, True):
                folded = list(fuse_diagonal_phases(source, affine=True, complements=complements))
                self.assertTrue(
                    verify_phase_fusion(source, folded, affine=True, complements=complements)[
                        "verified"
                    ]
                )
                if not complements:
                    self.assertEqual(list(fuse_diagonal_phases(folded, affine=True)), folded)
                np.testing.assert_allclose(
                    simulator.run(source, np.eye(8)),
                    simulator.run(folded, np.eye(8)),
                    atol=4e-14,
                    rtol=0,
                )
        for barrier in (
            ("h", (0,), None),
            ("x", (0,), None),
            ("cx", (1, 0), None),
            ("hmr", (0,), 0),
        ):
            source = [("t", (0,), None), barrier, ("tdg", (0,), None)]
            with self.assertRaises(Incorrect):
                verify_phase_fusion(source, [barrier], affine=True)

    def test_complement_phase_folding_preserves_all_eight_scalar_phases(self):
        from itertools import product

        from qpe_lab.circuits.checks import verify_phase_fusion
        from qpe_lab.circuits.gadgets.synthesis import fuse_diagonal_phases

        observed = set()
        for left, right in product(range(8), repeat=2):
            source = [("t", (0,), None)] * left + [("x", (0,), None)]
            source += [("t", (0,), None)] * right + [("x", (0,), None)]
            folded = list(fuse_diagonal_phases(source, affine=True, complements=True))
            proof = verify_phase_fusion(source, folded, affine=True, complements=True)
            observed.add(proof["scalar_repair_pi_over_4_units"])
            np.testing.assert_allclose(
                StatevectorSimulator(1).run(source, np.eye(2)),
                StatevectorSimulator(1).run(folded, np.eye(2)),
                atol=3e-14,
                rtol=0,
            )
            if proof["scalar_repair_pi_over_4_units"]:
                start = [i for i, g in enumerate(folded) if g == ("x", (0,), None)][-2]
                with self.assertRaises(Incorrect):
                    verify_phase_fusion(source, folded[:start], affine=True, complements=True)
                wrong = list(folded)
                wrong[start] = ("x", (1,), None)
                with self.assertRaises(Incorrect):
                    verify_phase_fusion(source, wrong, affine=True, complements=True)
        self.assertEqual(observed, set(range(8)))
        for wrong in (
            [("x", (0,), None)],
            [("x", (0,), None)] * 3,
            [("x", (0,), None), ("t", (1,), None), ("x", (0,), None)],
        ):
            with self.assertRaises(Incorrect):
                verify_phase_fusion([], wrong, affine=True, complements=True)
        with self.assertRaises(Incorrect):
            verify_phase_fusion([], [], complements=True)
        with self.assertRaises(ValueError):
            list(fuse_diagonal_phases([], complements=True))

    def test_preparation_tree_with_zeros_and_unequal_weights(self):
        weights = [0, 3, 0, 5, 7, 0, 1, 0]
        expansion = GateExpansion(3)
        gates = []
        for gate in prepare_weights(weights, 3):
            target = gate["target"]
            controls = [(q, gate["prefix"] >> (q - target - 1) & 1) for q in range(target + 1, 3)]
            angle = 2 * math.atan2(math.sqrt(gate["right"]), math.sqrt(gate["left"]))
            gates.extend(expansion.ry(target, angle, controls))
        state = np.zeros(1 << expansion.peak_qubits)
        state[0] = 1
        result = StatevectorSimulator(expansion.peak_qubits).run(gates, state)
        np.testing.assert_allclose(
            result[:8], np.sqrt(np.array(weights) / sum(weights)), atol=1e-14
        )
        np.testing.assert_allclose(result[8:], 0, atol=1e-14)

    def test_givens_expansion_all_fock_columns_and_adjoint(self):
        expansion = GateExpansion(4)
        gates = list(expansion.givens(1, 2, -11, 5))
        expected = DenseFockSimulator(4)
        expected.givens(1, 2, -11, 5)
        actual = StatevectorSimulator(4).run(gates, np.eye(16))
        np.testing.assert_allclose(actual, expected.matrix, atol=1e-14)
        np.testing.assert_allclose(
            StatevectorSimulator(4).run(adjoint(gates), actual), np.eye(16), atol=1e-14
        )
        large_turns = list(expansion.givens(1, 2, -11 + (1 << 35), 5))
        np.testing.assert_allclose(
            StatevectorSimulator(4).run(large_turns, np.eye(16)), expected.matrix, atol=1e-14
        )

    def test_controlled_majoranas_all_phases_parity_and_clean_ancilla(self):
        for pauli in ("I", "X", "Y", "Z"):
            for phase in range(4):
                expansion = GateExpansion(6)
                gates = list(expansion.branch(5, pauli, phase, [(0, 0), (1, 1)], system_start=2))
                state = np.zeros((128, 64), dtype=complex)
                state[:64] = np.eye(64)
                expected = state.copy()
                selected = np.arange(16) * 4 + 2
                expected[np.ix_(selected, selected)] = 1j**phase * jordan_wigner(4, 3, pauli)
                actual = StatevectorSimulator(7).run(gates, state)
                np.testing.assert_allclose(actual, expected, atol=1e-14)


class GateRewriteTests(unittest.TestCase):
    def check(self, gates, stages, protected=(0,), *, clean=False):
        from qpe_lab.evaluation.verification.engine import verify_gate_rewrites

        return verify_gate_rewrites(
            {
                "gates": gates,
                "protected_wires": list(protected),
                **({"ancilla_initial": "zero"} if clean else {}),
            },
            {"reference_sha256": "a" * 64, "stages": stages},
            "a" * 64,
        )

    @staticmethod
    def local(start, stop, gates):
        return {"kind": "local", "edits": [{"start": start, "stop": stop, "gates": gates}]}

    def test_arbitrary_local_words_and_actual_resource_accounting(self):
        gates = [("t", (0,), None)] * 2 + [("h", (9,), None)] * 2
        after, report = self.check(
            gates, [self.local(0, 2, [("s", (0,), None)]), self.local(1, 3, [])], protected=(0, 4)
        )
        self.assertEqual(after, [("s", (0,), None)])
        self.assertEqual(report["cost"]["t_count"], 0)
        self.assertEqual(report["cost"]["qubits"], 2)  # Protected unused I/O still charged.
        self.assertEqual(report["cost"]["operations"], 1)
        word = [("cx", (0, 1), None), ("h", (2,), None), ("x", (3,), None)]
        self.assertEqual(self.check(word + word[::-1], [self.local(0, 6, [])])[0], [])
        # A whole-circuit scalar is checked exactly too; no phase alignment.
        with self.assertRaises(Incorrect):
            self.check([("x", (0,), None), ("z", (0,), None)] * 2, [self.local(0, 4, [])])

    def test_reset_facts_are_derived_and_all_branches_agree(self):
        from qpe_lab.circuits.gadgets.synthesis import zero_target_ccx_clifford_t

        old = list(ccx_clifford_t(0, 1, 2))
        new = list(zero_target_ccx_clifford_t(0, 1, 2))
        for prefix in (
            [],
            [("assert_zero", (2,), None)],
            [("hmr", (2,), 0), ("x", (2,), None)],
            [("hmr", (2,), 0), ("h", (2,), None)],
            [("hmr", (2,), 0), ("x", (2,), {"condition": 0})],
        ):
            with self.subTest(prefix=prefix), self.assertRaises(Incorrect):
                self.check(prefix + old, [self.local(len(prefix), len(prefix) + len(old), new)])
        prefix = [("hmr", (2,), 0)]
        source = prefix + old + [("z", (0,), {"condition": 0})]
        after, report = self.check(source, [self.local(1, 1 + len(old), new)])
        self.assertEqual(report["cost"]["t_count"], 4)
        for outcome in (0, 1):
            np.testing.assert_allclose(
                measurement_kraus(source, 3, [outcome])[0],
                measurement_kraus(after, 3, [outcome])[0],
                atol=2e-15,
            )

    def test_affine_correlations_and_basis_barriers(self):
        prefix = [("hmr", (1,), 0), ("cx", (0, 1), None)]
        word = [("z", (0,), None), ("z", (1,), None)]
        after, report = self.check(prefix + word, [self.local(2, 4, [])])
        self.assertEqual(report["evidence"]["stages"][0]["checks"][0]["input_columns"], [0, 3])
        for outcome in (0, 1):
            np.testing.assert_allclose(
                measurement_kraus(prefix + word, 2, [outcome])[0],
                measurement_kraus(after, 2, [outcome])[0],
                atol=2e-15,
            )
        for barrier in ([("h", (0,), None)], [("x", (0,), {"condition": 0})]):
            with self.assertRaises(Incorrect):
                self.check(prefix + barrier + word, [self.local(3, 5, [])])

    def test_random_prefix_certificates_against_all_kraus_matrices(self):
        from itertools import product

        rng = np.random.default_rng(43609)
        for _ in range(40):
            prefix, count = [], 0
            for _ in range(9):
                q = int(rng.integers(3))
                name = ("h", "x", "cx", "hmr")[int(rng.integers(4))]
                if name == "hmr" and count < 3:
                    prefix.append((name, (q,), count))
                    count += 1
                elif name == "cx":
                    prefix.append((name, (q, (q + 1) % 3), None))
                else:
                    prefix.append(("h" if name == "hmr" else name, (q,), None))
            word = [("z", (int(rng.integers(3)),), None) for _ in range(2)]
            try:
                after, _ = self.check(prefix + word, [self.local(len(prefix), len(prefix) + 2, [])])
            except Incorrect:
                continue
            for branch in product((0, 1), repeat=count):
                np.testing.assert_allclose(
                    measurement_kraus(prefix + word, 3, branch)[0],
                    measurement_kraus(after, 3, branch)[0],
                    atol=3e-15,
                )

    def test_phase_stages_and_malformed_classical_programs(self):
        source = [("t", (0,), None)] * 2 + [("h", (0,), None)]
        after = [("s", (0,), None), ("h", (0,), None)]
        self.assertEqual(self.check(source, [{"kind": "phase", "gates": after}])[0], after)
        for bad in (
            after[:-1],
            [("h", (0,), None), ("s", (0,), None)],
            [("z", (0,), {"condition": 10})],
            [("hmr", (0,), 0)] * 2,
            [("oxor", (), [0, -1, 3])],
        ):
            with self.subTest(bad=bad), self.assertRaises(Incorrect):
                self.check(source, [{"kind": "phase", "gates": bad}])

    def test_parser_rejects_claims_nonfinite_and_invalid_edits(self):
        from qpe_lab.circuits.ops import CIRCUIT_MAGIC, read_circuit

        base = {"schema": "qpe.gate-rewrites.v1", "reference_sha256": "a" * 64, "stages": []}
        bad = [
            base | {"claimed_cost": 0},
            base | {"reference_sha256": "b"},
            base | {"stages": [{"kind": "unknown"}]},
            base | {"stages": [self.local(0, 129, [])]},
            base | {"stages": [self.local(True, 2, [])]},
            base | {"stages": [self.local(0, 1, [("cx", (1, 1), None)])]},
            base | {"stages": [self.local(0, 1, [("x", (False,), None)])]},
        ]
        raw_bad = [json.dumps(doc).encode() for doc in bad] + [
            b'{"schema":"qpe.gate-rewrites.v1","schema":"other"}',
            json.dumps(base).replace('"stages": []', '"stages": NaN').encode(),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rewrites.ops"
            for raw in raw_bad:
                with self.subTest(raw=raw):
                    path.write_bytes(
                        CIRCUIT_MAGIC
                        + b'{"schema":"qpe.circuit.v1","reference_sha256":"'
                        + b"a" * 64
                        + b'","entries":{"walk":{"gates":[],"certificate":'
                        + raw
                        + b"}}}"
                    )
                    with self.assertRaises(ValueError):
                        read_circuit(path)

    def test_zx_large_blocks_preserve_phase_and_wire_order(self):
        from qpe_lab.circuits.sim import Unsupported

        # More than four wires and 128 gates: the original matrix interface
        # cannot admit this rewrite, although it is an exact unitary identity.
        word = [("cx", (i, i + 1), None) for i in range(7)]
        old = (word + word[::-1]) * 10 + [("t", (7,), None)] * 2
        new = [("s", (7,), None)]
        edit = {"start": 0, "stop": len(old), "gates": new}
        after, report = self.check(old, [{"kind": "zx", "edits": [edit]}])
        self.assertEqual(after, new)
        self.assertEqual(report["cost"]["t_count"], 0)
        for bad in ([], new + [("x", (7,), None), ("z", (7,), None)] * 2, [("s", (6,), None)]):
            with self.subTest(bad=bad), self.assertRaises((Incorrect, Unsupported)):
                self.check(old, [{"kind": "zx", "edits": [edit | {"gates": bad}]}])

    def test_zx_independent_dense_crosscheck_and_unknown_result(self):
        from qpe_lab.circuits.sim import Unsupported, verify_zx_equivalence

        rng = np.random.default_rng(10212026)
        for _ in range(12):
            word = [
                ("h" if rng.integers(2) else "t", (int(rng.integers(3)),), None) for _ in range(12)
            ]
            old = word + [("t", (0,), None)] * 2
            new = word + [("s", (0,), None)]
            self.assertTrue(verify_zx_equivalence(old, new)["verified"])
            np.testing.assert_allclose(
                StatevectorSimulator(3).run(old, np.eye(8)),
                StatevectorSimulator(3).run(new, np.eye(8)),
                atol=3e-15,
            )
            with self.assertRaises((Incorrect, Unsupported)):
                verify_zx_equivalence(old, new + [("z", (0,), None)])
        with self.assertRaises(Unsupported):
            verify_zx_equivalence([("hmr", (0,), 0)], [])

    def test_dynamic_all_kraus_records_and_external_classical_bits(self):
        from itertools import product

        prefix = [("hmr", (3,), 0)]
        old = [("hmr", (1,), 1), ("hmr", (2,), 2), ("oxor", (), [3, 0, 1])]
        old += [("t", (0,), {"condition": 3})] * 4
        new = [
            ("hmr", (2,), 2),
            ("hmr", (1,), 1),
            ("oxor", (), [3, 0, 1]),
            ("z", (0,), {"condition": 3}),
        ]
        edit = {"start": 1, "stop": 1 + len(old), "gates": new}
        after, report = self.check(prefix + old, [{"kind": "dynamic", "edits": [edit]}])
        self.assertEqual(report["cost"]["t_count"], 0)
        self.assertFalse(report["evidence"]["stages"][0]["checks"][0]["sampling"])
        for external, first, second in product((0, 1), repeat=3):
            left = measurement_kraus(prefix + old, 4, [external, first, second])
            right = measurement_kraus(after, 4, [external, second, first])
            np.testing.assert_allclose(left[0], right[0], atol=2e-15)
            self.assertEqual(left[1], right[1])
        bad = new[:-1] + [("z", (0,), {"condition": 2})]
        with self.assertRaises(Incorrect):
            self.check(prefix + old, [{"kind": "dynamic", "edits": [edit | {"gates": bad}]}])

    def test_dynamic_preserves_observable_records_and_sums_internal_phases(self):
        old = [("h", (0,), None), ("hmr", (0,), 0)]
        suffix = [("z", (1,), {"condition": 0})]
        for bad in ([old[1]], [old[1], old[0]]):
            with self.subTest(bad=bad), self.assertRaises(Incorrect):
                self.check(
                    old + suffix,
                    [{"kind": "dynamic", "edits": [{"start": 0, "stop": 2, "gates": bad}]}],
                )
        harmless = old + [("neg", (), {"condition": 0})]
        self.check(
            old + suffix,
            [{"kind": "dynamic", "edits": [{"start": 0, "stop": 2, "gates": harmless}]}],
        )
        # With the outcome unobserved both implementations reset this input;
        # retaining a fictitious record identity would wrongly reject equality.
        self.check(
            old,
            [{"kind": "dynamic", "edits": [{"start": 0, "stop": 2, "gates": [("hmr", (0,), 47)]}]}],
        )
        with self.assertRaises(Incorrect):
            self.check(
                old + suffix,
                [
                    {
                        "kind": "dynamic",
                        "edits": [{"start": 0, "stop": 2, "gates": [("hmr", (0,), 47)]}],
                    }
                ],
            )

    def test_reuse_reduces_physical_qubits_and_preserves_complete_instrument(self):
        from itertools import product

        from qpe_lab.circuits.gadgets.synthesis import reuse_clean_qubits
        from qpe_lab.circuits.sim import Unsupported

        source = []
        for q in (1, 2):
            source += [
                ("cx", (0, q), None),
                ("t", (q,), None),
                ("cx", (0, q), None),
                ("hmr", (q,), q - 1),
            ]
        source += [("assert_zero", (q,), None) for q in (1, 2)]
        moves, mapped = reuse_clean_qubits(source, [0])
        stage = {"kind": "reuse", "moves": moves, "gates": mapped}
        after, report = self.check(source, [stage], clean=True)
        self.assertEqual(report["cost"]["qubits"], 2)
        self.assertEqual(report["cost"]["t_count"], 2)
        self.assertEqual(report["cost"]["tq_cost"], 4)
        for branch in product((0, 1), repeat=2):
            # Independent dense simulation: zero ancillary inputs, arbitrary
            # coherent data inputs; both untouched auxiliary outputs are zero.
            left = measurement_kraus(source, 3, branch)[0][:, :2]
            right = measurement_kraus(after, 2, branch)[0][:, :2]
            embedded = np.zeros_like(left)
            embedded[:4] = right
            np.testing.assert_allclose(left, embedded, atol=3e-15)
        with self.assertRaises(Unsupported):
            self.check(source, [stage])  # Legacy all-input-state contract.

    def test_reuse_rejects_aliases_forged_zero_and_changed_semantics(self):
        from qpe_lab.circuits.gadgets.synthesis import reuse_clean_qubits
        from qpe_lab.evaluation.verification.exact import verify_qubit_reuse

        for first in (("h", (1,), None), ("x", (1,), None), ("cx", (0, 1), None)):
            source = [first, ("assert_zero", (1,), None)]
            with self.subTest(first=first), self.assertRaises(Incorrect):
                verify_qubit_reuse(source, source, [[0, 1, 1], [2, 1, -1]], [0])
        source = [("cx", (0, 1), None), ("hmr", (1,), 0), ("z", (0,), {"condition": 0})]
        moves, mapped = reuse_clean_qubits(source, [0])
        for bad_moves, bad_gates in (
            ([[0, 1, 0]], mapped),  # Alias a protected coherent input.
            ([[0, 0, -1]], mapped),
            (moves, mapped[:-1]),
            (moves, mapped[:-1] + [("z", (0,), None)]),
            (moves + [[len(source) + 1, 4, 2]], mapped),
        ):
            with self.subTest(moves=bad_moves), self.assertRaises(Incorrect):
                verify_qubit_reuse(source, bad_gates, bad_moves, [0])

    def test_random_reuse_against_independent_complete_kraus_matrices(self):
        from itertools import product

        from qpe_lab.circuits.gadgets.synthesis import reuse_clean_qubits

        rng = np.random.default_rng(431449)
        protected = (0, 2)  # Noncontiguous coherent I/O, including entangled inputs.
        columns = [0, 1, 4, 5]
        for _ in range(24):
            source, outcomes = [], 0
            for _ in range(12):
                q = int(rng.integers(4))
                name = ("h", "x", "t", "cx", "hmr")[int(rng.integers(5))]
                if name == "hmr" and outcomes < 2:
                    source.append((name, (q,), outcomes))
                    outcomes += 1
                elif name == "cx":
                    source.append((name, (q, (q + 1) % 4), None))
                else:
                    name = "z" if name == "hmr" else name
                    condition = {"condition": 0} if outcomes and name == "z" else None
                    source.append((name, (q,), condition))
            for q in (1, 3):
                source.append(("hmr", (q,), outcomes))
                outcomes += 1
            source += [("assert_zero", (q,), None) for q in (1, 3)]
            moves, mapped = reuse_clean_qubits(source, protected)
            after, _ = self.check(
                source,
                [{"kind": "reuse", "moves": moves, "gates": mapped}],
                protected=protected,
                clean=True,
            )
            for branch in product((0, 1), repeat=outcomes):
                left = measurement_kraus(source, 4, branch)[0][:, columns]
                right = measurement_kraus(after, 4, branch)[0][:, columns]
                np.testing.assert_allclose(left, right, atol=3e-15)

    def test_clean_input_contract_does_not_come_from_an_assertion(self):
        source = [("assert_zero", (1,), None), ("t", (1,), None)]
        stage = self.local(1, 2, [])
        self.assertEqual(self.check(source, [stage], clean=True)[0], source[:1])
        with self.assertRaises(Incorrect):
            self.check(source, [stage])

    def test_expanded_artifact_and_reference_parsing(self):
        from qpe_lab.circuits.ops import CIRCUIT_MAGIC, read_circuit, read_reference

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data"
            reference = {
                "schema": "qpe.gate-reference.v2",
                "ancilla_initial": "zero",
                "protected_wires": [0],
                "gates": [["t", [1], None]],
            }
            path.write_text(json.dumps(reference))
            self.assertEqual(read_reference(path)["ancilla_initial"], "zero")
            for bad in (
                reference | {"ancilla_initial": "unknown"},
                reference | {"claimed_clean": [0]},
                reference | {"schema": "qpe.gate-reference.v1"},
            ):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    read_reference(path)
            base = {"schema": "qpe.gate-rewrites.v1", "reference_sha256": "a" * 64}
            stage = {"kind": "reuse", "moves": [[0, 1, 1], [1, 1, -1]], "gates": [["hmr", [1], 0]]}
            envelope = {
                "schema": "qpe.circuit.v1",
                "reference_sha256": "a" * 64,
                "entries": {"walk": {"gates": [], "certificate": base | {"stages": [stage]}}},
            }
            path.write_bytes(CIRCUIT_MAGIC + json.dumps(envelope).encode())
            self.assertEqual(
                read_circuit(path)["entries"]["walk"]["certificate"]["stages"][0]["moves"],
                stage["moves"],
            )
            for moves in ([[0, True, 1]], [[-1, 1, 1]], [[0, 1, -2]], [[2, 1, 1], [1, 1, -1]]):
                envelope["entries"]["walk"]["certificate"]["stages"] = [stage | {"moves": moves}]
                path.write_bytes(CIRCUIT_MAGIC + json.dumps(envelope).encode())
                with self.assertRaises(ValueError):
                    read_circuit(path)


class CompleteCircuitTests(unittest.TestCase):
    def check(self, before, after, protected, **kwargs):
        from qpe_lab.circuits.ops import elementary_gates
        from qpe_lab.evaluation.verification.engine import verify_entry

        reference = {
            "schema": "qpe.gate-reference.v2",
            "ancilla_initial": "zero",
            "protected_wires": protected,
            "gates": elementary_gates(before),
        }
        certificate = kwargs.pop("certificate", None)
        return verify_entry(
            reference,
            {"gates": after, **({"certificate": certificate} if certificate is not None else {})},
            "a" * 64,
            exact_only=True,
            **kwargs,
        )[1]

    def test_independent_words_and_internal_measurement_records(self):
        before = [("cx", (0, 1), None), ("hmr", (1,), 0), ("z", (0,), {"condition": 0})]
        report = self.check(before, [], [0])
        self.assertEqual(report["correctness_status"], "verified")
        self.assertEqual(report["cost"]["qubits"], 1)
        other = [("hmr", (2,), 47), ("neg", (), {"condition": 47})]
        self.check(before, other, [0])
        with self.assertRaises(Incorrect):
            self.check(before, before[:-1], [0])
        with self.assertRaises(Incorrect):
            self.check([], [("assert_zero", (0,), None)], [0])

    def test_rare_phase_is_rejected_without_sampling(self):
        chain = [(0, 1, 40)] + [(i + 38, i, i + 39) for i in range(2, 40)]
        before = [g for a, b, c in chain for g in ccx_clifford_t(a, b, c)]
        after = [g for a, b, c in reversed(chain) for g in ccx_clifford_t(a, b, c)]
        identity = before + after
        report = self.check(identity, [], list(range(40)))
        self.assertEqual(report["correctness_status"], "verified")
        with self.assertRaises(Incorrect):
            self.check(identity, before + [("z", (78,), None)] + after, list(range(40)))

    def test_boolean_exact_compiled_and_and_measurement_cleanup(self):
        from qpe_lab.circuits.gadgets.synthesis import zero_target_ccx_clifford_t
        from qpe_lab.circuits.sim import verify_monomial_equivalence

        old = list(ccx_clifford_t(0, 1, 2)) * 2
        new = list(zero_target_ccx_clifford_t(0, 1, 2)) + [
            ("hmr", (2,), 99),
            ("cz", (0, 1), {"condition": 99}),
        ]
        result = verify_monomial_equivalence(old, new, [0, 1])
        self.assertFalse(result["sampling"])
        self.check(old, new, [0, 1])
        with self.assertRaises(Incorrect):
            self.check(old, new[:-1], [0, 1])

    def test_boolean_phase_arithmetic_against_independent_dense_columns(self):
        from qpe_lab.circuits.sim import BooleanDiagram, monomial_signature

        rng = np.random.default_rng(210021)
        for trial in range(24):
            gates = []
            for _ in range(24):
                name = rng.choice(["x", "cx", "ccx", "t", "tdg", "s", "sdg", "z", "cz"])
                arity = 3 if name == "ccx" else 2 if name in ("cx", "cz") else 1
                wires = tuple(int(q) for q in rng.choice(3, size=arity, replace=False))
                gates.extend(ccx_clifford_t(*wires) if name == "ccx" else [(name, wires, None)])
            gates += [("cx", (0, 3), None), ("hmr", (3,), trial), ("z", (0,), {"condition": trial})]
            manager = BooleanDiagram()
            outputs, phase, _ = monomial_signature(gates, manager, [0, 1, 2])

            def value(node, x, manager=manager):
                while node >= 2:
                    variable, low, high = manager.nodes[node]
                    node = high if x >> variable & 1 else low
                return node

            expected = np.zeros((16, 8), dtype=complex)
            for x in range(8):
                row = sum(value(node, x) << q for q, node in outputs.items())
                power = sum(value(node, x) << i for i, node in enumerate(phase))
                expected[row, x] = np.exp(1j * math.pi * power / 4)
            for branch in (0, 1):
                actual = measurement_kraus(gates, 4, [branch])[0][:, :8]
                np.testing.assert_allclose(actual, expected / np.sqrt(2), atol=1e-14)

    def test_multiple_independent_quantum_words(self):
        old = [("h", (i,), None) for i in range(8)] * 2 + [("t", (7,), None)] * 2
        new = [("s", (7,), None)]
        self.check(old, new, list(range(8)))

    def test_certificate_cannot_certify_a_different_payload(self):
        cert = {"schema": "qpe.gate-rewrites.v1", "reference_sha256": "a" * 64, "stages": []}
        with self.assertRaisesRegex(Incorrect, "different circuit"):
            self.check([], [("t", (0,), None)], [0], certificate=cert)

    def test_independent_wide_qrom_constructions_without_rewrite_history(self):
        from qpe_lab.circuits.gadgets.synthesis import compile_gate_stream

        # Actual Fe2S2 angle-word width: 19 angles x 16 bits, with arbitrary
        # coherent target contents. Both algorithms clean all their own work.
        words = [sum(((i * 7 + bit * 11) % 13 < 6) << bit for bit in range(304)) for i in range(8)]
        programs = []
        for method in ("unary", "selectswap"):
            circuit = DynamicCircuit(307)
            lookup_xor(circuit, [0, 1, 2], list(range(3, 307)), words, method=method, block_size=2)
            programs.append(list(compile_gate_stream(circuit.gates, {"library": {}})))
        report = self.check(programs[0], programs[1], list(range(307)))
        self.assertEqual(report["correctness_status"], "verified")
        self.assertIn("Boolean", report["evidence"]["proof"]["method"])
        certificate = {
            "schema": "qpe.gate-rewrites.v1",
            "reference_sha256": "a" * 64,
            "stages": [
                {
                    "kind": "boolean",
                    "edits": [{"start": 0, "stop": len(programs[0]), "gates": programs[1]}],
                }
            ],
        }
        self.check(programs[0], programs[1], list(range(307)), certificate=certificate)
        with self.assertRaises(Incorrect):
            self.check(programs[0], programs[1] + [("z", (0,), None)], list(range(307)))

    def test_pair_cost_uses_shared_physical_allocation(self):
        from qpe_lab.evaluation.cost import circuit_pair_cost

        entries = {
            "forward": {"protected_wires": [0], "gates": [("t", (1,), None)]},
            "adjoint": {"protected_wires": [0], "gates": [("t", (7,), None)]},
        }
        cost = circuit_pair_cost(entries)
        self.assertEqual((cost["t_count"], cost["qubits"], cost["tq_cost"]), (2, 3, 6))

    def test_orbital_basis_all_fe2s2_vectors_and_multimode_fock_lift(self):
        from unittest.mock import patch

        from qpe_lab.circuits.gadgets.givens import orbital_basis
        from qpe_lab.circuits.sim import GaussianSimulator, fock_lift
        from qpe_lab.tasks.data.angles import reconstruct

        task = load_walk(project_root() / "benchmarks/dfthc/fe2s2/walk-paper-strict/walk.task.json")
        n, beta = task.model["spatial_modes"], task.model["angle_bits"]
        for row in task.model["orbital_angles"]:
            word = sum(value << (i * beta) for i, value in enumerate(row))
            matrix = GaussianSimulator(n)

            def record(
                _circuit,
                p,
                q,
                bits,
                _gradient,
                *,
                inverse=False,
                adder="general",
                word=word,
                matrix=matrix,
            ):
                turns = sum(((word >> bit) & 1) << i for i, bit in enumerate(bits))
                matrix.givens(p, q, -turns if inverse else turns, beta)

            with patch("qpe_lab.circuits.gadgets.givens.phase_gradient_givens", side_effect=record):
                orbital_basis(None, list(range(n)), list(range((n - 1) * beta)), list(range(beta)))
            np.testing.assert_allclose(
                matrix.matrix[:, 0],
                reconstruct(np.asarray(row) * math.tau / (1 << beta)),
                atol=1e-13,
            )
        # Two unequal, non-quarter-turn angles expose both the ordering and sign.
        row, beta = [1, 3], 3
        for inverse in (False, True):
            circuit = DynamicCircuit(3)
            angles, gradient = circuit.qubits(6), circuit.qubits(3)
            initialization = prepare_phase_gradient(circuit, gradient)
            orbital_basis(circuit, list(range(3)), angles, gradient, inverse=inverse)
            circuit.gates.extend(adjoint(initialization))
            matrix = GaussianSimulator(3)
            for q, value in enumerate(reversed(row), 1):
                matrix.givens(q - 1, q, -value, beta)
            expected = fock_lift(matrix.matrix)
            if inverse:
                expected = expected.conj().T
            word = sum(v << (i * beta) for i, v in enumerate(row))
            base = sum(((word >> i) & 1) << q for i, q in enumerate(angles))
            indices = [base | i for i in range(8)]
            for seed in (0, 7):
                result = SparseStatevectorSimulator(circuit.peak_qubits).run(
                    circuit.gates, dict(zip(indices, np.eye(8), strict=True)), seed=seed
                )
                np.testing.assert_allclose(
                    [result.get(i, np.zeros(8)) for i in indices], expected, atol=1e-12
                )
                self.assertLess(
                    sum(np.linalg.norm(v) ** 2 for i, v in result.items() if i not in indices),
                    1e-22,
                )

    def test_core_branches_against_independent_majoranas_including_failed_success(self):
        from itertools import product

        from qpe_lab.circuits.gadgets.prepare import paper_layout
        from qpe_lab.circuits.gadgets.synthesis import expand_strict_paper_walk

        model = make_model(
            np.array([[[0.6, 0.8]]]),
            np.array([[[0.7]]]),
            np.array([[0.4]]),
            np.array([-0.35, 0.6]),
            np.array([[0.8, -0.6], [0.6, 0.8]]),
            energy_offset=-1.2,
            beta=3,
            coefficient_bits=8,
        )
        with tempfile.TemporaryDirectory() as folder:
            task = load_walk(
                write_walk_task(
                    with_paper_preparation(model, 1, uniform_bits=3), folder, "core-branches"
                )
            )
            doc = build_walk(
                task,
                implementation="paper_strict",
                specialize_constants=True,
                outer_block_size=2,
                qrom_block_size=2,
                outer_unlookup_size=2,
                unlookup_block_size=4,
            ).program
            layout = paper_layout(task.model)
            identity = np.eye(16, dtype=complex)
            annihilation = [
                (jordan_wigner(4, p, "X") + 1j * jordan_wigner(4, p, "Y")) / 2 for p in range(4)
            ]
            rotated = {}
            for orbital, angles in enumerate(task.model["orbital_angles"]):
                theta = math.tau * angles[0] / 8
                for spin in range(2):
                    a = (
                        math.cos(theta) * annihilation[2 * spin]
                        + math.sin(theta) * annihilation[2 * spin + 1]
                    )
                    rotated[orbital, spin, "X"] = a + a.conj().T
                    rotated[orbital, spin, "Y"] = -1j * (a - a.conj().T)
                    rotated[orbital, spin, "Z"] = identity - 2 * a.conj().T @ a
            parity = jordan_wigner(4, 0, "Z") @ jordan_wigner(4, 1, "Z")
            for inverse in (False, True):
                raw, expansion = expand_strict_paper_walk(
                    task, doc, core=True, inverse_core=inverse
                )
                resource = DynamicCircuit(expansion.peak_qubits)
                initialization = prepare_phase_gradient(resource, expansion.registers["gradient"])
                gates = initialization + list(raw) + list(adjoint(initialization))
                for x, si, so, good in product(range(4), range(2), range(2), range(2)):
                    b, os, ins = (x + si) % 2, (so + good) % 2, (x + good) % 2
                    table = doc["alias"]["outer"]
                    keep = table["keep"][x] if x < len(table["keep"]) else 0
                    alt = table["alt"][x] if x < len(table["alt"]) else 0
                    chosen = alt if os >= keep else x
                    if chosen < len(layout["sf"]):
                        table = doc["alias"]["inner"][chosen]
                        term = table["alt"][b] if ins >= table["keep"][b] else b
                        source = task.model["generators"][layout["sf"][chosen]]["terms"]
                        t = source[0 if term == layout["bases"] else 1 + 2 * term + si]
                        expected = (1j ** t["phase"]) * (
                            rotated[t["orbital"], si, "Z"]
                            if good and t["pauli"] != "I"
                            else identity
                        )
                    elif good:
                        source = task.model["generators"][
                            layout["single"][chosen - len(layout["sf"])][so]
                        ]["terms"]
                        t = source[si]
                        expected = (1j ** t["phase"]) * rotated[t["orbital"], so, t["pauli"]]
                    else:
                        expected = parity if so else identity
                    if inverse:
                        expected = expected.conj().T
                    values = {
                        "label": x,
                        "term": b,
                        "spin_inner": si,
                        "spin_outer": so,
                        "outer_success": good,
                        "outer_sigma": os,
                        "inner_sigma": ins,
                    }
                    base = sum(
                        ((value >> i) & 1) << q
                        for name, value in values.items()
                        for i, q in enumerate(expansion.registers[name])
                    )
                    indices = [
                        base
                        | sum(
                            ((v >> i) & 1) << q for i, q in enumerate(expansion.registers["system"])
                        )
                        for v in range(16)
                    ]
                    actual = SparseStatevectorSimulator(expansion.peak_qubits).run(
                        gates, dict(zip(indices, identity, strict=True)), seed=7
                    )
                    with self.subTest(inverse=inverse, x=x, si=si, so=so, good=good):
                        np.testing.assert_allclose(
                            [actual.get(i, np.zeros(16)) for i in indices], expected, atol=1e-11
                        )
                        self.assertLess(
                            sum(
                                np.linalg.norm(v) ** 2
                                for i, v in actual.items()
                                if i not in indices
                            ),
                            1e-20,
                        )

    def test_modular_core_walk_against_independent_hamiltonian(self):
        from unittest.mock import patch

        from qpe_lab.circuits.gadgets.synthesis import core_reference, expand_core_walk
        from qpe_lab.evaluation.walk import crosscheck_paper_walk

        model = make_model(
            np.array([[[0.6, 0.8]]]),
            np.array([[[0.7]]]),
            np.array([[0.4]]),
            np.array([-0.35, 0.6]),
            np.array([[0.8, -0.6], [0.6, 0.8]]),
            energy_offset=-1.2,
            beta=3,
            coefficient_bits=8,
        )
        with tempfile.TemporaryDirectory() as folder:
            task = load_walk(
                write_walk_task(
                    with_paper_preparation(model, 1, uniform_bits=3), folder, "core-small"
                )
            )
            doc = build_walk(
                task,
                implementation="paper_strict",
                outer_block_size=2,
                qrom_block_size=2,
                outer_unlookup_size=2,
                unlookup_block_size=4,
            ).program
            reference, costs = core_reference(task, doc)
            self.assertEqual(set(reference["entries"]), {"forward", "adjoint"})
            self.assertGreater(costs["forward"]["structural_toffoli"], 0)
            for entries in (None, reference["entries"]):
                gates, expansion = expand_core_walk(task, doc, entries)
                gates = list(gates)
                with patch(
                    "qpe_lab.circuits.gadgets.synthesis.expand_walk",
                    return_value=(iter(gates), expansion),
                ):
                    report = crosscheck_paper_walk(task, doc)
                self.assertTrue(report["checked"])
                self.assertLess(report["max_error"], 1e-10)


class WalkDataAndRunnerTests(unittest.TestCase):
    def test_frozen_judge_rejects_task_or_grader_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            task = load_walk(write_walk_task(toy_model(), folder, "pin"))
            path = Path(folder) / "judge.json"
            pin = freeze_judge(task.path, path)
            prov = provenance(project_root())
            check_judge(pin, task, prov)
            prov["source_hashes"]["src/qpe_lab/evaluation/walk.py"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "changed"):
                check_judge(pin, task, prov)
            with self.assertRaisesRegex(ValueError, "already exists"):
                freeze_judge(task.path, path)

    def test_centered_hamiltonian_recovery_and_quantization_bound_in_full_fock_space(self):
        u = np.array([[[0.6, 0.8], [0.8, -0.6]]])
        w = np.array([[[0.7, -0.2], [-0.3, 0.5]]])
        shifts = np.array([[0.25, -0.4]])
        sym = np.array([[0.1, 0.03], [0.03, -0.2]])
        hcore = np.array([[-0.7, 0.13], [0.13, 0.2]])
        tensor = np.einsum(
            "pq,rs->pqrs", np.array([[0.4, 0.1], [0.1, 0.3]]), np.array([[0.4, 0.1], [0.1, 0.3]])
        )
        factors = {
            "params": {
                "U": u,
                "W": w,
                "H2_I": shifts,
                "H1_I": 0.15,
                "Hsym_lower": sym[np.tril_indices(2)],
            }
        }
        _, _, _, eig, vectors, diag = centered_sos(factors, hcore, tensor, 0.12, 2)
        model = make_model(
            u,
            w,
            shifts,
            eig,
            vectors,
            energy_offset=diag["energy_offset"],
            beta=12,
            coefficient_bits=18,
        )
        a = [(jordan_wigner(4, p, "X") + 1j * jordan_wigner(4, p, "Y")) / 2 for p in range(4)]
        e = np.array(
            [
                [
                    sum(a[2 * spin + p].conj().T @ a[2 * spin + q] for spin in range(2))
                    for q in range(2)
                ]
                for p in range(2)
            ]
        )
        identity = np.eye(16)
        k = e - np.eye(2)[:, :, None, None] * identity
        hthc = hcore - 0.5 * np.einsum("pqqs->ps", tensor)
        original = 0.12 * identity + np.einsum("pq,pqij->ij", hthc, e)
        original += 0.5 * np.einsum("pqrs,pqik,rskj->ij", tensor, e, e)
        f = np.einsum("rbp,rbq,rbc->rcpq", u, u, w)
        shift_tensor = 0.5 * (
            np.einsum("pq,rs->pqrs", np.eye(2), sym) + np.einsum("pq,rs->pqrs", sym, np.eye(2))
        )
        delta = np.einsum("abpq,abrs->pqrs", f, f) - tensor - shift_tensor
        number = e[0, 0] + e[1, 1]
        shift_operator = (number - 2 * identity) @ (
            0.15 * identity + 0.5 * np.einsum("pq,pqij->ij", sym, k)
        )
        target = original + shift_operator + 0.5 * np.einsum("pqrs,pqik,rskj->ij", delta, k, k)
        generators = reference_generators(model)
        implemented = sum(o.conj().T @ o for o in generators) + model["energy_offset"] * identity
        error = float(np.linalg.norm(implemented - target, ord=2))
        self.assertLessEqual(
            error, model["precision"]["hamiltonian_quantization_norm_bound"] + 1e-12
        )
        self.assertLess(error, 0.01)
        sector = [state for state in range(16) if state.bit_count() == 2]
        np.testing.assert_allclose(shift_operator[np.ix_(sector, sector)], 0, atol=1e-14)

    def test_fe2s2_recovers_publication_normalization_and_offset(self):
        root = project_root()
        source = root / "benchmarks/dfthc/fe2s2/factors.json"
        factors = json.loads(source.read_text())
        h, g, core, electrons = read_fcidump(
            root / "third_party/uncompute/tasks/chemistry-fe2s2/data/fcidump.xz"
        )
        *_, diagnostics = centered_sos(factors, h, g, core, electrons)
        for key in ("lambda_H1", "lambda_H2", "lambda_DFTHC", "H0", "error_h2"):
            self.assertAlmostEqual(diagnostics[key], factors["published_values"][key], delta=1e-3)
        task = load_walk(root / "benchmarks/dfthc/fe2s2/walk/walk.task.json")
        self.assertEqual(task.model["source"]["factor_sha256"], digest(source))
        self.assertEqual(len(task.model["generators"]), 110)
        self.assertEqual(len(task.model["orbital_angles"]), 230)
        self.assertEqual(task.mode_count, 40)

    def test_python_generator_isolation_and_frozen_report(self):
        with tempfile.TemporaryDirectory(
            dir=project_root() / "runs", prefix="walk-test-"
        ) as folder:
            root = Path(folder)
            path = write_walk_task(toy_model(), root / "task", "isolated-walk")
            candidate = root / "generator.py"
            candidate.write_text("from qpe_lab import build_walk\nbuild_walk().emit()\n")
            run, report = evaluate(path, candidate)
            self.assertEqual(report["correctness_status"], "exact_verified", report)
            self.assertTrue(report["numerical_crosscheck"]["checked"])
            self.assertFalse(report["kernel_crosscheck"]["applicable"])
            self.assertEqual(report["h_ops"], digest(run / "circuit.ops"))
            self.assertTrue((run / "input/sos-model.json").is_file())


class SamplingVerificationTests(unittest.TestCase):
    def test_exact_sparse_kraus_matches_independent_dense_columns(self):
        from fractions import Fraction
        from itertools import product

        from qpe_lab.circuits.sim import exact_instrument_columns, exact_sparse_branch

        rng = np.random.default_rng(7621)
        for case in range(12):
            gates = []
            for _ in range(20):
                name = str(rng.choice(["h", "t", "tdg", "x", "s", "cz", "cx"]))
                qs = tuple(map(int, rng.choice(3, 2 if name in {"cz", "cx"} else 1, replace=False)))
                gates.append((name, qs, None))
            gates += [
                ("hmr", (0,), 17),
                ("z", (2,), {"condition": 17}),
                ("hmr", (1,), 19),
                ("oxor", (), (21, 17, 19)),
                ("x", (0,), {"condition": 21}),
            ]
            for bits in product((0, 1), repeat=2):
                outcomes = dict(zip((17, 19), bits, strict=True))
                dense, _ = exact_instrument_columns(gates, 3, list(range(8)), outcomes, {})
                for column in range(8):
                    sparse, denominator, _ = exact_sparse_branch(
                        gates, {column: (1, 0, 0, 0)}, outcomes, {}
                    )
                    for row in range(8):
                        self.assertEqual(
                            tuple(
                                Fraction(x, 2**denominator) for x in sparse.get(row, (0, 0, 0, 0))
                            ),
                            dense[row][column],
                            (case, bits, column, row),
                        )

    def test_fingerprints_sum_kraus_weights_and_detect_dephasing(self):
        from qpe_lab.circuits.sim import Incorrect, sample_channel_equivalence

        measured = [("cx", (0, 1), None), ("hmr", (1,), 7), ("z", (0,), {"condition": 7})]
        kwargs = {"seed": b"fixed regression seed", "rounds": 2}
        evidence = sample_channel_equivalence([], measured, [0], **kwargs)
        self.assertLess(evidence["false_accept_bound"], 1e-16)
        for bad in (
            measured[:-1],
            [("h", (0,), None)],
            [("t", (0,), None)],
            [("assert_zero", (0,), None)],
        ):
            with self.subTest(bad=bad), self.assertRaises(Incorrect):
                sample_channel_equivalence([], bad, [0], **kwargs)
        # Internal classical record names and global Kraus phases are immaterial.
        alternate = [
            ("cx", (0, 1), None),
            ("hmr", (1,), 31),
            ("z", (0,), {"condition": 31}),
            ("neg", (), {"condition": 31}),
        ]
        sample_channel_equivalence(measured, alternate, [0], **kwargs)
        # A branch whose probability is zero must not be renormalized or skipped
        # by a test that compares a single, normalized trajectory.
        sample_channel_equivalence([], [("h", (1,), None), ("hmr", (1,), 0)], [0], **kwargs)

    def test_classical_interface_and_scratch_contract_are_checked(self):
        from qpe_lab.circuits.sim import Incorrect, sample_channel_equivalence

        kwargs = {"seed": b"interface", "rounds": 2}
        before = [("hmr", (1,), 0)]
        after = [("h", (1,), None), ("hmr", (1,), 0)]
        sample_channel_equivalence(before, after, [0], **kwargs)
        with self.assertRaises(Incorrect):
            sample_channel_equivalence(before, after, [0], outputs=[0], **kwargs)
        with self.assertRaises(Incorrect):
            sample_channel_equivalence(
                [], [("x", (0,), {"condition": 5})], [0], external=[5], **kwargs
            )
        with self.assertRaises(Incorrect):
            sample_channel_equivalence([], [("x", (3,), None)], [], **kwargs)
        # A nontrivial relative phase on an input superposition is observable.
        with self.assertRaises(Incorrect):
            sample_channel_equivalence([], [("cz", (0, 1), None)], [0, 1], **kwargs)

    def test_wide_basis_sampling_preserves_phase_and_all_record_debts(self):
        from qpe_lab.circuits.sim import Incorrect, sample_monomial_equivalence

        reference = [("cx", (q, q + 1), None) for q in range(39)]
        candidate = reference + [
            ("cx", (0, 40), None),
            ("hmr", (40,), 8),
            ("z", (0,), {"condition": 8}),
        ]
        kwargs = {"seed": b"basis", "shots": 2048, "batch": 127}
        evidence = sample_monomial_equivalence(reference, candidate, list(range(40)), **kwargs)
        self.assertEqual(evidence["random_shots"], 2048)
        for bad in (
            candidate[:-1],
            reference + [("t", (17,), None)],
            reference + [("x", (0,), None)],
        ):
            with self.subTest(bad=bad[-1]), self.assertRaises(Incorrect):
                sample_monomial_equivalence(reference, bad, list(range(40)), **kwargs)

    def test_automatic_sampling_covers_insertions_and_all_changed_regions(self):
        from qpe_lab.circuits.sim import Incorrect, Unsupported
        from qpe_lab.evaluation.verification.engine import verify_entry
        from qpe_lab.evaluation.verification.sampled import SamplingContext

        def verify_sampled_circuit(reference, proposed, pin, context):
            return verify_entry(
                reference, {"gates": proposed}, pin, context=context, exact_seconds=0
            )

        reference = {
            "gates": [("h", (q,), None) for q in range(13)] * 2,
            "protected_wires": list(range(21)),
            "ancilla_initial": "zero",
        }
        identity = [("h", (0,), None), ("h", (0,), None)]
        candidate = identity + reference["gates"] + [("t", (1,), None), ("tdg", (1,), None)]
        _, report = verify_sampled_circuit(
            reference, candidate, "pin", SamplingContext(b"blocks", 1e-12, shots=512)
        )
        self.assertTrue(report["evidence"]["sampling"])
        with self.assertRaises((Incorrect, Unsupported)):
            verify_sampled_circuit(
                reference,
                candidate + [("x", (2,), None)],
                "pin",
                SamplingContext(b"blocks", 1e-12, shots=512),
            )

    def test_fingerprint_budget_does_not_drop_branches_or_amplitudes(self):
        from qpe_lab.circuits.sim import Unsupported, sample_channel_equivalence

        for gates, kwargs in [
            ([("h", (q,), None) for q in range(13)], {}),
            ([("hmr", (0,), q) for q in range(9)], {}),
            ([("h", (0,), None)] * 10, {"max_work": 1}),
        ]:
            with self.subTest(gates=gates), self.assertRaises(Unsupported):
                sample_channel_equivalence([], gates, [], seed=b"limits", rounds=1, **kwargs)


if __name__ == "__main__":
    unittest.main()
