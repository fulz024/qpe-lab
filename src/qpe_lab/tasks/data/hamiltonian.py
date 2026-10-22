"""Recover the centered spin-free SOS model from pinned numerical factors.

Independent NumPy implementation of the definitions in arXiv:2502.15882 and
Zenodo 17066718's df_thc_sga.py. No archive code is imported or executed.
K_pq = sum_spin a_p^dagger a_q - delta_pq; mode order is spin-blocked.
"""

import hashlib
import json
import lzma
import math
import re
from pathlib import Path

import numpy as np

from qpe_lab.circuits.gadgets.prepare import paper_layout
from qpe_lab.tasks.data.angles import elimination_angles, reconstruct
from qpe_lab.tasks.task import load_walk

FCIDUMP_SHA256 = "95d8786af06eeea2107e19ffd98c66a6ca97fc8c9864175a4f6d64512b6f2df9"


def read_fcidump(path):
    """Read real spatial Mulliken integrals, including all eight symmetries."""
    raw = lzma.decompress(Path(path).read_bytes())
    if hashlib.sha256(raw).hexdigest() != FCIDUMP_SHA256:
        raise ValueError("FCIDUMP does not match the pinned Fe2S2 source")
    text = raw.decode()
    end = re.search(r"&END|/", text)
    if end is None:
        raise ValueError("Missing FCIDUMP header terminator")
    header = text[: end.start()]
    n = int(re.search(r"NORB\s*=\s*(\d+)", header).group(1))
    electrons = int(re.search(r"NELEC\s*=\s*(\d+)", header).group(1))
    h, g, core = np.zeros((n, n)), np.zeros((n, n, n, n)), 0.0
    for line in text[end.end() :].splitlines():
        if not line.strip():
            continue
        value, *indices = line.split()
        value = float(value.replace("D", "E").replace("d", "e"))
        i, j, k, l = (int(x) - 1 for x in indices)
        if i == j == k == l == -1:
            core = value
        elif k == l == -1:
            h[i, j] = h[j, i] = value
        else:
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    g[a, b, c, d] = g[c, d, a, b] = value
    return h, g, core, electrons


def centered_sos(factors, hcore, integrals, core, electrons):
    """Return M, F, and constants for H = M:K + sum(d+F:K)^2/2 + c.

    The symmetry term is (Nhat-Ne)*(H1_I + Hsym:K/2), which vanishes in
    the specified particle sector. Approximation error is in the centered
    two-electron tensor, not an asserted chemical-energy error bound.
    """
    p = {key: np.asarray(value, dtype=float) for key, value in factors["params"].items()}
    u = p["U"] / np.linalg.norm(p["U"], axis=-1, keepdims=True)
    f = np.einsum("rbp,rbq,rbc->rcpq", u, u, p["W"])
    n = hcore.shape[0]
    sym = np.zeros((n, n))
    sym[np.tril_indices(n)] = p["Hsym_lower"]
    sym = sym + sym.T - np.diag(np.diag(sym))
    hdf = hcore + np.einsum("pprs->rs", integrals) - 0.5 * np.einsum("pqqs->ps", integrals)
    cdf = core + np.trace(hdf) - 0.5 * np.einsum("ppqq", integrals)
    m = (
        hdf
        + 0.5 * (n - electrons) * sym
        + float(p["H1_I"]) * np.eye(n)
        - np.einsum("rc,rcpq->pq", p["H2_I"], f)
    )
    c = cdf + (n - electrons) * float(p["H1_I"]) - 0.5 * np.sum(p["H2_I"] ** 2)
    shifted_g = np.einsum("rcpq,rckl->pqkl", f, f)
    identity = np.eye(n)
    delta = (
        shifted_g
        - integrals
        - 0.5 * (np.einsum("pq,rs->pqrs", identity, sym) + np.einsum("pq,rs->pqrs", sym, identity))
    )
    eig, vectors = np.linalg.eigh(m)
    lambda1 = float(np.sum(np.abs(eig)))
    lambda2 = float(np.sum((np.abs(p["H2_I"]) + np.abs(p["W"]).sum(axis=1)) ** 2) / 4)
    diagnostics = {
        "lambda_H1": lambda1,
        "lambda_H2": lambda2,
        "lambda_DFTHC": lambda1 + lambda2,
        "H0": float(-c),
        "energy_offset": float(c - lambda1),
        "error_h2": float(np.linalg.norm(delta)),
        "error_h2_ord=1": float(np.abs(delta).sum()),
        "core_energy": float(core),
        "electrons": electrons,
        "eigendecomposition_residual": float(np.max(np.abs((vectors * eig) @ vectors.T - m))),
    }
    return u, p["W"], p["H2_I"], eig, vectors, diagnostics


def make_model(u, weights, shifts, eig, vectors, *, energy_offset, beta=16, coefficient_bits=24):
    """Freeze a finite-angle, dyadic-coefficient SOS target.

    Each term is integer_weight / 2**coefficient_bits times i**phase times
    a rotated I/Z/Jordan-Wigner X/Y operator. Weights are nonnegative.
    """
    n = u.shape[-1]
    all_vectors = np.concatenate((u.reshape(-1, n), vectors.T))
    angles, vector_errors, max_vector_error = [], [], 0.0
    for vector in all_vectors:
        theta = elimination_angles(vector)
        integers = np.rint(theta / math.tau * (1 << beta)).astype(np.int64) % (1 << beta)
        angles.append([int(x) for x in integers])
        error = float(np.linalg.norm(reconstruct(integers * math.tau / (1 << beta)) - vector))
        vector_errors.append(error)
        max_vector_error = max(max_vector_error, error)
    generators, coefficient_error = [], 0.0
    term_bounds = {}
    scale = 1 << coefficient_bits

    def term(coefficient, orbital, spin, pauli, phase=0):
        nonlocal coefficient_error
        integer = round(abs(float(coefficient)) * scale)
        coefficient_error += abs(integer / scale - abs(coefficient))
        result = {
            "weight": integer,
            "orbital": orbital,
            "spin": spin,
            "pauli": pauli,
            "phase": (phase + (2 if coefficient < 0 else 0)) % 4,
        }
        rotation_error = 0 if pauli == "I" else vector_errors[orbital] * (4 if pauli == "Z" else 1)
        term_bounds[id(result)] = (
            abs(coefficient),
            abs(integer / scale - abs(coefficient)) + abs(coefficient) * rotation_error,
        )
        return result

    ranks, bases, copies = weights.shape
    for r in range(ranks):
        for c in range(copies):
            terms = [term(shifts[r, c] / math.sqrt(2), -1, 0, "I")]
            for b in range(bases):
                for spin in range(2):
                    terms.append(
                        term(-weights[r, b, c] / (2 * math.sqrt(2)), r * bases + b, spin, "Z")
                    )
            generators.append({"name": f"SF-{r}-{c}", "terms": terms})
    for j, value in enumerate(eig):
        for spin in range(2):
            amplitude = math.sqrt(abs(float(value))) / 2
            generators.append(
                {
                    "name": f"{'D' if value >= 0 else 'Q'}-{j}-{spin}",
                    "terms": [
                        term(amplitude, ranks * bases + j, spin, "X"),
                        term(amplitude, ranks * bases + j, spin, "Y", 1 if value >= 0 else 3),
                    ],
                }
            )
    lambdas = [sum(t["weight"] for t in g["terms"]) / scale for g in generators]
    quantization_bound = 0.0
    for generator, quantized_norm in zip(generators, lambdas, strict=True):
        ideal_norm = sum(term_bounds[id(t)][0] for t in generator["terms"])
        error = sum(term_bounds[id(t)][1] for t in generator["terms"])
        quantization_bound += (ideal_norm + quantized_norm) * error
    # Include dropped generators in the error bound before removing them.
    generators = [g for g in generators if sum(t["weight"] for t in g["terms"])]
    return {
        "schema": "qpe.sos-model.v1",
        "spatial_modes": n,
        "angle_bits": beta,
        "coefficient_bits": coefficient_bits,
        "mode_order": "spin-blocked",
        "orbital_angles": angles,
        "generators": generators,
        "energy_offset": energy_offset,
        "lambda_sqrt_squared": math.fsum(x * x for x in lambdas),
        "precision": {
            "max_vector_l2_error": max_vector_error,
            "coefficient_l1_rounding_error": coefficient_error,
            "hamiltonian_quantization_norm_bound": float(quantization_bound),
            "bound_method": "sum_alpha (LCU_norm_ideal+LCU_norm_quantized)*generator_error; Majorana error <= vector L2, reflection error <= 4*vector L2; binary64 evaluation, not interval arithmetic",
            "scope": "frozen quantized SOS target; no chemical-energy guarantee",
        },
    }


def write_walk_task(model, directory, name, *, optimization=None):
    from qpe_lab.contracts import digest

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sos-model.json"
    path.write_text(json.dumps(model, indent=2) + "\n")
    spec = {
        "schema": "qpe.walk.v1",
        "name": name,
        "table": path.name,
        "table_sha256": digest(path),
        "max_qubits": 1024,
        "controlled": True,
        "walk_convention": "Rin Udag Rout U; R=2P-I",
        "input_projector": "label=0,term=0",
        "output_projector": "term=0; retain label",
        "gate_model": "ideal algebraic RY and dyadic Givens; explicit reversible controls",
    }
    if "paper" in model:
        spec["input_projector"] = "x,b,spin_outer,spin_inner,sigma_outer,sigma_inner,uniform=0"
        spec["output_projector"] = "b,spin_inner,sigma_inner,uniform_inner=0; retain outer"
        if "uniform_bits" in model["paper"]:
            spec["output_projector"] += "; success_outer=1"
    task_path = directory / "walk.task.json"
    if optimization is not None:
        spec["optimization"] = optimization
    task_path.write_text(json.dumps(spec, indent=2) + "\n")
    load_walk(task_path)
    return task_path


def with_alias_preparation(model, bits):
    """Freeze an explicit finite-precision preparation target, preserving the input.

    Original vectors/phases and normalization remain fixed. Coefficients of the
    new SOS generators follow the actual outer/inner alias cell distributions.
    """
    import copy

    from qpe_lab.circuits.gadgets.prepare import quantize_distribution

    if "preparation" in model:
        raise ValueError("Model already has a frozen preparation target")
    result = copy.deepcopy(model)
    weights = [[t["weight"] for t in g["terms"]] for g in model["generators"]]
    width = 1 << max(1, (max(map(len, weights)) - 1).bit_length())
    inner = [quantize_distribution(row + [0] * (width - len(row)), bits) for row in weights]
    outer = quantize_distribution([sum(row) ** 2 for row in weights], bits)
    result["preparation"] = {
        "schema": "qpe.alias-target.v1",
        "alias_bits": bits,
        "outer": outer,
        "inner": inner,
    }
    norm2 = sum(sum(row) ** 2 for row in weights) / (1 << (2 * model["coefficient_bits"]))
    bound = 0.0
    for a, row in enumerate(weights):
        old = [w / (1 << model["coefficient_bits"]) for w in row]
        new_norm = math.sqrt(norm2 * outer[a] / sum(outer))
        new = [new_norm * w / sum(inner[a]) for w in inner[a][: len(row)]]
        delta = math.fsum(abs(x - y) for x, y in zip(old, new, strict=True))
        bound += (math.fsum(old) + new_norm) * delta
    result["precision"]["alias_hamiltonian_norm_bound"] = bound
    result["precision"]["alias_bound_method"] = (
        "sum (old_LCU_norm+new_LCU_norm)*coefficient_L1_difference; "
        "binary64 numerical bound relative to the pre-alias quantized SOS model"
    )
    result["precision"]["scope"] = (
        "fixed alias-cell SOS target; separate angle/coefficient and alias bounds; "
        "not the publication's empirical chemical accuracy certificate"
    )
    return result


def with_paper_preparation(model, bits, *, uniform_bits=None):
    """Freeze compact spin-free aliases; retain expanded factors as evidence.

    Uniform domains are exactly N+RC and B+1, not padded domains. Rounding is
    deterministic largest remainder, explicitly distinct from Appendix D's
    stochastic chemical-error experiment.
    """
    import copy

    from qpe_lab.circuits.gadgets.prepare import quantize_distribution

    if "preparation" in model or "paper" in model:
        raise ValueError("Model already has a preparation profile")
    layout = paper_layout(model)
    result = copy.deepcopy(model)
    generators = model["generators"]
    norms = [sum(t["weight"] for t in g["terms"]) for g in generators]
    inner_weights = [
        [2 * t["weight"] for t in generators[i]["terms"][1::2]]
        + [generators[i]["terms"][0]["weight"]]
        for i in layout["sf"]
    ]
    outer_weights = [norms[i] ** 2 for i in layout["sf"]] + [
        sum(norms[i] ** 2 for i in pair) for pair in layout["single"]
    ]
    inner = [quantize_distribution(row, bits, pad=False) for row in inner_weights]
    outer = quantize_distribution(outer_weights, bits, pad=False)
    result["paper"] = {
        "schema": "qpe.paper-target.v1",
        "alias_bits": bits,
        "rank": layout["rank"],
        "bases": layout["bases"],
        "copies": layout["copies"],
        "outer": outer,
        "inner": inner,
    }
    if uniform_bits is not None:
        if type(uniform_bits) is not int or not 2 <= uniform_bits <= 16:
            raise ValueError("Invalid uniform rotation precision")
        if (layout["bases"] + 1) & layout["bases"]:
            raise ValueError("Appendix B executable profile currently requires binary B+1")
        result["paper"].update(schema="qpe.paper-target.v2", uniform_bits=uniform_bits)
    norm2 = sum(x * x for x in norms) / (1 << (2 * model["coefficient_bits"]))
    bound = 0.0
    for x, i in enumerate(layout["sf"]):
        old = [w / (1 << model["coefficient_bits"]) for w in inner_weights[x]]
        new_norm = math.sqrt(norm2 * outer[x] / sum(outer))
        new = [new_norm * w / sum(inner[x]) for w in inner[x]]
        delta = sum(abs(a - b) for a, b in zip(old, new, strict=True))
        bound += (sum(old) + new_norm) * delta
    for j, pair in enumerate(layout["single"]):
        new_norm = math.sqrt(norm2 * outer[len(layout["sf"]) + j] / sum(outer) / 2)
        for i in pair:
            old_norm = norms[i] / (1 << model["coefficient_bits"])
            bound += (old_norm + new_norm) * abs(old_norm - new_norm)
    result["precision"].update(
        alias_hamiltonian_norm_bound=bound,
        alias_bound_method="sum (old_LCU_norm+new_LCU_norm)*coefficient_L1_difference; binary64",
        scope="compact paper-layout alias target; deterministic rounding, not Appendix D stochastic tensors",
    )
    return result
