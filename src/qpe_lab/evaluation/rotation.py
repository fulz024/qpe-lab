"""Exact fixed-consumer composition plus independent numerical cross-checks.

Acceptance uses full-domain XOR verification and integer identities within each
commuting Givens group. Floating-point Gaussian/Fock checks are supplementary.
The accepted model has ideal dyadic rotations; synthesis is explicitly pending.
"""

import base64
import math
from pathlib import Path

import numpy as np

from qpe_lab.circuits.checks import LookupChecker
from qpe_lab.circuits.compact_reader import interpret
from qpe_lab.circuits.ops import read_hybrid
from qpe_lab.circuits.sim import DenseFockSimulator, GaussianSimulator, Incorrect, fock_lift
from qpe_lab.contracts import digest
from qpe_lab.evaluation.kernel import sample_lookup


def _consumer_groups(gates, beta):
    """Combine commuting controls within a consecutive mode pair, modulo 2*pi."""
    groups = []
    modulus = 1 << beta
    for gate in gates:
        p, q, turns = gate["p"], gate["q"], gate["turns"]
        if p > q:
            p, q, turns = q, p, -turns
        if turns % modulus == 0:
            continue
        pair = (p, q)
        if not groups or groups[-1][0] != pair:
            groups.append((pair, {}))
        coefficients = groups[-1][1]
        key = (gate["angle_bit"], gate["controlled"])
        coefficients[key] = (coefficients.get(key, 0) + turns) % modulus
    return [
        (pair, {key: value for key, value in coeff.items() if value})
        for pair, coeff in groups
        if any(coeff.values())
    ]


def _expected_groups(task):
    # Independently derive from the task's packed elimination-angle convention.
    result = []
    beta, n = task.angle_bits, task.spatial_modes
    for block in range(task.spin_copies):
        for right in range(1, n):
            angle = n - right - 1
            terms = {
                (angle * beta + bit, task.controlled): (-(1 << bit)) % (1 << beta)
                for bit in range(beta)
            }
            result.append(((block * n + right - 1, block * n + right), terms))
    return result


def reference_modes(task, word, *, enable=True):
    """Reference directly decodes complete angles; it never calls the gadget."""
    matrix = np.eye(task.mode_count)
    if task.controlled and not enable:
        return matrix
    modulus = 1 << task.angle_bits
    for spin in range(task.spin_copies):
        for right in range(1, task.spatial_modes):
            angle_index = task.spatial_modes - 1 - right
            integer = (word >> (angle_index * task.angle_bits)) & (modulus - 1)
            theta = -math.tau * integer / modulus
            left = spin * task.spatial_modes + right - 1
            c, s = math.cos(theta), math.sin(theta)
            rotation = np.eye(task.mode_count)
            rotation[left : left + 2, left : left + 2] = [[c, s], [-s, c]]
            matrix = rotation @ matrix
    return matrix


def verify_rotation(task, stream_path, output_directory, *, kernel=True):
    doc = read_hybrid(stream_path)
    for key in ("address_bits", "word_bits", "angle_bits", "mode_count", "controlled"):
        if doc[key] != getattr(task, key):
            raise Incorrect(f"Hybrid register contract mismatch: {key}")
    steps = doc["steps"]
    if steps[0]["kind"] != "lookup" or steps[-1]["kind"] != "lookup":
        raise Incorrect("Rotation requires load and unload lookup blocks")
    gates = steps[1:-1]
    if any(gate["kind"] != "cgivens" for gate in gates):
        raise Incorrect("Only fixed-consumer rotations may occur between load and unload")
    if _consumer_groups(gates, task.angle_bits) != _expected_groups(task):
        raise Incorrect("Rotation consumer differs from the fixed dyadic Givens contract")

    output = Path(output_directory)
    output.mkdir(exist_ok=True)
    lookup_task = task.lookup_contract()
    blocks = []
    for name, step in (("load", steps[0]), ("unload", steps[-1])):
        raw = base64.b64decode(step["ops_b64"], validate=True)
        if len(raw) > 32 * 1024**2:
            raise ValueError("Embedded lookup exceeds size limit")
        path = output / (name + ".ops")
        path.write_bytes(raw)
        path.chmod(0o444)
        checker = LookupChecker(lookup_task)
        cost = interpret(path, checker, lookup_task.max_qubits)
        block = {"name": name, "h_ops": digest(path), "cost": cost, "evidence": checker.evidence()}
        if kernel:
            destination = output / (name + "-kernel")
            destination.mkdir()
            rc, report = sample_lookup(lookup_task, path, destination)
            if rc != 0 or report is None:
                raise Incorrect(f"C kernel cross-check failed for {name}")
            for key in ("toffoli", "clifford", "qubits", "ops"):
                if report["metrics"][key] != cost[key]:
                    raise Incorrect(f"Kernel/symbolic count discrepancy in {name}: {key}")
            block["kernel_crosscheck"] = report
        blocks.append(block)

    maximum = dense_maximum = 0.0
    dense = task.mode_count <= 6 and len(task.words) <= 32
    for word in task.words:
        for enable in (False, True) if task.controlled else (True,):
            actual = GaussianSimulator(task.mode_count).run(
                gates, word, task.angle_bits, enable=enable
            )
            expected = reference_modes(task, word, enable=enable)
            error = float(np.max(np.abs(actual - expected)))
            if not math.isfinite(error) or error > 1e-10:
                raise Incorrect("Gaussian cross-check disagrees with the fixed consumer")
            maximum = max(maximum, error)
            if dense:
                full = DenseFockSimulator(task.mode_count).run(
                    gates, word, task.angle_bits, enable=enable
                )
                error = float(np.max(np.abs(full - fock_lift(expected))))
                if not math.isfinite(error) or error > 1e-10:
                    raise Incorrect("Full Fock-space cross-check disagrees (phase retained)")
                dense_maximum = max(dense_maximum, error)
    discrete = {
        key: sum(block["cost"][key] for block in blocks)
        for key in ("toffoli", "clifford", "ops", "measurements")
    }
    qubits = max(block["cost"]["qubits"] for block in blocks) + task.reserved_qubits
    return {
        "correctness_status": "exact_verified",
        "cost": {
            "status": "partial",
            "toffoli": None,
            "clifford": None,
            "tq_cost": None,
            "qubits": qubits,
            "workspace_qubits": qubits - task.address_bits - task.reserved_qubits,
            "discrete": discrete,
            "controlled_givens": len(gates),
            "rotation_controls": "angle bit plus enable" if task.controlled else "angle bit",
            "synthesis_status": "not_synthesized",
            "synthesis_t_count": None,
            "scope": "full load/use/unload at ideal dyadic-rotation gate level; fault-tolerant cost incomplete",
        },
        "evidence": {
            "method": "exact XOR blocks + integer fixed-consumer identities + clean-register composition",
            "addresses": len(task.words),
            "system_modes": task.mode_count,
            "controlled": task.controlled,
            "vacuum_phase": "+1 (fixed by each allowed Givens primitive)",
            "logical_scope": "arbitrary coherent address, enable, and Fock input; angle/workspace start and end zero",
            "rotation_model": "ideal Givens with dyadic angles; no synthesis approximation",
            "consumer_groups": len(_expected_groups(task)),
        },
        "lookup_blocks": blocks,
        "numerical_crosscheck": {
            "gaussian_max_abs_error": maximum,
            "dense_fock_checked": dense,
            "dense_fock_max_abs_error": dense_maximum if dense else None,
            "phase_alignment_used": False,
            "role": "supplementary; exact verdict uses integer composition",
        },
    }
