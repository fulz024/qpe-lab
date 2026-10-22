"""Explicit real adjacent-mode Givens convention and frozen integer angle table."""

import json
import math
from pathlib import Path

import numpy as np

from qpe_lab.contracts import digest
from qpe_lab.tasks.task import load_lookup, load_rotation


def elimination_angles(vector):
    work = np.asarray(vector, dtype=np.float64).copy()
    norm = np.linalg.norm(work)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Orbital vector must have finite nonzero norm")
    work /= norm
    angles = []
    for j in range(len(work) - 1, 0, -1):
        angles.append(math.atan2(work[j], work[j - 1]))
        work[j - 1] = math.hypot(work[j - 1], work[j])
        work[j] = 0
    return np.asarray(angles)


def reconstruct(angles):
    work = np.zeros(len(angles) + 1)
    work[0] = 1
    for j, theta in enumerate(reversed(angles), 1):
        c, s = math.cos(theta), math.sin(theta)
        a, b = work[j - 1], work[j]
        work[j - 1], work[j] = c * a - s * b, s * a + c * b
    return work


def write_task(factors_path, directory, beta=16):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    factors = json.loads(Path(factors_path).read_text())
    vectors = np.asarray(factors["params"]["U"], dtype=np.float64).reshape(-1, factors["N"])
    words = []
    max_ideal = max_quantized = 0.0
    for vector in vectors:
        normalized = vector / np.linalg.norm(vector)
        angles = elimination_angles(vector)
        max_ideal = max(max_ideal, float(np.linalg.norm(reconstruct(angles) - normalized)))
        integer = np.rint(angles / (2 * math.pi) * (1 << beta)).astype(np.int64) % (1 << beta)
        max_quantized = max(
            max_quantized,
            float(np.linalg.norm(reconstruct(integer * (2 * math.pi / (1 << beta))) - normalized)),
        )
        words.append(sum(int(word) << (j * beta) for j, word in enumerate(integer)))
    if max_ideal > 1e-12:
        raise ValueError("Givens convention does not reconstruct the source vectors")
    address_bits = (len(words) - 1).bit_length()
    words += [0] * ((1 << address_bits) - len(words))
    table = {
        "schema": "qpe.angle-table.v1",
        "words_hex": [hex(word) for word in words],
        "valid_vectors": len(vectors),
        "address": "r * B + b; zero on padded addresses",
        "angle_bits": beta,
        "angles_per_word": factors["N"] - 1,
        "packing": "angle j in bits [beta*j, beta*(j+1)); little endian",
        "decode": "theta = 2*pi*k/2**beta; modulo 2*pi",
        "rounding": "nearest integer, ties to even (binary64 preprocessing)",
        "convention": "G(theta)=[[cos,sin],[-sin,cos]] on (j-1,j), j=N-1 down to 1, maps u to e0",
        "factor_sha256": digest(factors_path),
        "max_ideal_vector_l2_residual": max_ideal,
        "max_quantized_vector_l2_residual": max_quantized,
        "vector_l2_error_bound_from_angles": (factors["N"] - 1) * math.pi / (1 << beta),
        "precision_scope": "16-bit microkernel choice, not a reproduced chemical-energy error guarantee",
    }
    table_path = directory / "angle-table.json"
    table_path.write_text(json.dumps(table, indent=2) + "\n")
    task = {
        "schema": "qpe.lookup.v1",
        "name": "fe2s2-dfthc-shared-vector-angles-b16",
        "table": table_path.name,
        "table_sha256": digest(table_path),
        "address_bits": address_bits,
        "word_bits": beta * (factors["N"] - 1),
        "max_qubits": 1024,
        "mapping": "|a>|z> -> |a>|z XOR table[a]>",
        "workspace": "starts and ends zero",
        "phase": "zero on every input and measurement branch",
        "delivery": "Python generator emitting OPSTRM01 via UNCOMPUTE_OPS_PATH",
    }
    (directory / "lookup.task.json").write_text(json.dumps(task, indent=2) + "\n")
    return task, table


def write_rotation_task(lookup_path, *, spin_copies=2, controlled=True):
    """Create a separate consumer task without changing the frozen lookup input."""

    lookup_path = Path(lookup_path)
    lookup = load_lookup(lookup_path)
    spec = json.loads(lookup_path.read_text())
    table = json.loads((lookup_path.parent / spec["table"]).read_text())
    spec.update(
        schema="qpe.rotation.v1",
        name=lookup.name + "-load-rotate-unload",
        angle_bits=table["angle_bits"],
        spatial_modes=table["angles_per_word"] + 1,
        spin_copies=spin_copies,
        controlled=controlled,
        consumer="inverse-elimination-givens-v1",
        mode_order="spin-blocked",
        angle_initial="zero",
        mapping="|a>|enable>|psi>|0> -> |a>|enable> U(table[a])**enable |psi>|0>"
        if controlled
        else "|a>|psi>|0> -> |a> U(table[a]) |psi>|0>",
        phase="fixed vacuum phase +1; no branch-wise phase alignment",
        delivery="Python generator emitting QPECIR01 with OPSTRM01 lookup blocks and dyadic CGivens",
        cost_scope="ideal rotation primitives; Clifford+T synthesis not included",
    )
    path = lookup_path.parent / "rotation.task.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")
    load_rotation(path)
    return path
