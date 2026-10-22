import math
from fractions import Fraction

import numpy as np

from qpe_lab.circuits.gadgets.synthesis import count_walk
from qpe_lab.circuits.ops import read_walk
from qpe_lab.circuits.sim import Incorrect, SOSSimulator, jordan_wigner


def crosscheck_dynamic_walk(task, doc):
    """Physical gate trajectories, including controls, garbage and relative phase."""
    from qpe_lab.circuits.gadgets.synthesis import expand_dynamic_walk
    from qpe_lab.circuits.sim import SparseStatevectorSimulator

    if (
        task.mode_count > 4
        or task.label_bits + task.term_bits > 2
        or task.model["angle_bits"] > 3
        or task.model.get("preparation", {}).get("alias_bits", 0) > 1
    ):
        return {"checked": False, "reason": "physical trajectory size limit; component composition"}
    gates, circuit = expand_dynamic_walk(task, doc)
    gates = list(gates)
    dimension = 1 << task.mode_count
    identity = np.eye(dimension)
    operators = reference_generators(task.model)
    h = sum(o.conj().T @ o for o in operators)
    norm2 = sum(sum(t["weight"] for t in g["terms"]) ** 2 for g in task.model["generators"]) / (
        1 << (2 * task.model["coefficient_bits"])
    )
    error = 0.0
    for enabled in (False, True) if task.controlled else (True,):
        base = (1 << (task.logical_qubits - 1)) if enabled and task.controlled else 0
        indices = [base | (i << (task.label_bits + task.term_bits)) for i in range(dimension)]
        initial = dict(zip(indices, identity, strict=True))
        expected = 2 * h / norm2 - identity if enabled else identity
        for seed in (0, 1):
            actual = SparseStatevectorSimulator(circuit.peak_qubits).run(gates, initial, seed=seed)
            compressed = np.array([actual.get(i, np.zeros(dimension)) for i in indices])
            matrix = np.array(list(actual.values()))
            error = max(
                error,
                float(np.max(np.abs(compressed - expected))),
                float(np.max(np.abs(matrix.conj().T @ matrix - identity))),
            )
            if not enabled:
                error = max(
                    error,
                    max(
                        (float(np.max(np.abs(v))) for k, v in actual.items() if k not in initial),
                        default=0.0,
                    ),
                )
    if error > 1e-9:
        raise Incorrect(f"Physical dynamic walk differs from independent Hamiltonian: {error}")
    return {
        "checked": True,
        "system_columns": dimension,
        "max_error": error,
        "seeds": [0, 1],
        "scope": "physical measurement trajectories, phase-sensitive compression and isometry; not exhaustive outcome enumeration",
    }


def _check_prepare(gates, distributions, bits):
    """Prove the prepared nonnegative state with exact rational probabilities.

    Fresh-zero splits cannot interfere, so probabilities suffice without floating
    square roots. Different bit orders and disjoint-branch schedules are legal.
    This checks the required isometry, not an arbitrary extension off |0>.
    """
    from qpe_lab.circuits.gadgets.prepare import tree_controls

    allowed = {label for label, _ in distributions}
    if any(g["label"] not in allowed for g in gates):
        raise Incorrect("PREPARE label is outside the fixed distribution")
    for label, weights in distributions:
        probabilities = {0: Fraction(1)}
        for gate in (g for g in gates if g["label"] == label):
            target = gate["target"]
            controls = tree_controls(gate, bits)
            if (
                not 0 <= target < bits
                or len({q for q, _ in controls}) != len(controls)
                or any(not 0 <= q < bits or q == target for q, _ in controls)
                or ("controls" not in gate and gate["prefix"] >= 1 << (bits - target - 1))
            ):
                raise Incorrect("PREPARE control or target is invalid")
            mask = sum(1 << q for q, _ in controls)
            value = sum(v << q for q, v in controls)
            selected = [(s, p) for s, p in probabilities.items() if s & mask == value]
            if any(s >> target & 1 for s, _ in selected):
                raise Incorrect("PREPARE split target is not fresh zero")
            denominator = gate["left"] + gate["right"]
            for state, probability in selected:
                del probabilities[state]
                if gate["left"]:
                    probabilities[state] = probability * Fraction(gate["left"], denominator)
                probabilities[state | (1 << target)] = probability * Fraction(
                    gate["right"], denominator
                )
        expected = {i: Fraction(w, sum(weights)) for i, w in enumerate(weights) if w}
        if probabilities != expected:
            raise Incorrect("PREPARE does not prepare the fixed exact weighted state")


def _check_orbitals(task, networks):
    rows, n, beta = (
        task.model["orbital_angles"],
        task.model["spatial_modes"],
        task.model["angle_bits"],
    )
    modulus = 1 << beta
    if len(networks) != len(rows):
        raise Incorrect("Orbital network count mismatch")
    for row, network in zip(rows, networks, strict=True):
        actual = []
        for p, q, turn in network:
            if not 0 <= p < q < n or q != p + 1:
                raise Incorrect("Unsupported orbital mode pair")
            if not turn % modulus:
                continue
            if actual and actual[-1][:2] == [p, q]:
                actual[-1][2] = (actual[-1][2] + turn) % modulus
            else:
                actual.append([p, q, turn % modulus])
        actual = [entry for entry in actual if entry[2]]
        expected = [
            [q - 1, q, (-row[n - 1 - q]) % modulus] for q in range(1, n) if row[n - 1 - q] % modulus
        ]
        if actual != expected:
            raise Incorrect("Orbital rotations differ from the fixed integer-angle target")


def reference_generators(model):
    """Independent reference from ladder operators, without generated networks."""
    n, beta = model["spatial_modes"], model["angle_bits"]
    modes, dimension = 2 * n, 1 << (2 * n)
    if modes > 6:
        raise ValueError("Dense generator reference limited to six modes")
    annihilation = [
        (jordan_wigner(modes, p, "X") + 1j * jordan_wigner(modes, p, "Y")) / 2 for p in range(modes)
    ]
    rotated = {}
    for orbital, angles in enumerate(model["orbital_angles"]):
        # R e0 directly as a spherical-coordinate vector; no gate builder calls.
        vector = np.zeros(n)
        vector[0] = 1
        for right, integer in enumerate(reversed(angles), 1):
            theta = math.tau * integer / (1 << beta)
            left = vector[right - 1]
            vector[right - 1], vector[right] = math.cos(theta) * left, math.sin(theta) * left
        for spin in range(2):
            a = sum(vector[p] * annihilation[spin * n + p] for p in range(n))
            rotated[orbital, spin, "X"] = a + a.conj().T
            rotated[orbital, spin, "Y"] = -1j * (a - a.conj().T)
            rotated[orbital, spin, "Z"] = np.eye(dimension) - 2 * a.conj().T @ a
    operators = []
    norm2 = sum(sum(t["weight"] for t in g["terms"]) ** 2 for g in model["generators"]) / (
        1 << (2 * model["coefficient_bits"])
    )
    paper_coefficients = {}
    if "paper" in model:
        from qpe_lab.circuits.gadgets.prepare import paper_layout

        prep, layout = model["paper"], paper_layout(model)
        for x, a in enumerate(layout["sf"]):
            amplitude = math.sqrt(norm2 * prep["outer"][x] / sum(prep["outer"]))
            probabilities = [w / sum(prep["inner"][x]) for w in prep["inner"][x]]
            paper_coefficients[a] = [amplitude * probabilities[-1]] + [
                amplitude * probabilities[b] / 2 for b in range(layout["bases"]) for _ in range(2)
            ]
        for j, pair in enumerate(layout["single"]):
            amplitude = math.sqrt(
                norm2 * prep["outer"][len(layout["sf"]) + j] / sum(prep["outer"]) / 2
            )
            for a in pair:
                paper_coefficients[a] = [amplitude / 2, amplitude / 2]
    for a, generator in enumerate(model["generators"]):
        operator = np.zeros((dimension, dimension), dtype=complex)
        for b, term in enumerate(generator["terms"]):
            pauli = (
                np.eye(dimension)
                if term["pauli"] == "I"
                else rotated[term["orbital"], term["spin"], term["pauli"]]
            )
            coefficient = term["weight"] / (1 << model["coefficient_bits"])
            if "paper" in model:
                coefficient = paper_coefficients[a][b]
            if "preparation" in model:
                prep = model["preparation"]
                coefficient = (
                    math.sqrt(norm2 * prep["outer"][a] / sum(prep["outer"]))
                    * prep["inner"][a][b]
                    / sum(prep["inner"][a])
                )
            operator += coefficient * (1j ** term["phase"]) * pauli
        operators.append(operator)
    return operators


def crosscheck_walk(task, doc):
    simulation = SOSSimulator(task, doc)
    j = simulation.embedding()
    encoded = simulation.run(j, encoding_only=True)
    operators = reference_generators(task.model)
    lambdas = [
        sum(t["weight"] for t in g["terms"]) / (1 << task.model["coefficient_bits"])
        for g in task.model["generators"]
    ]
    normalization = math.sqrt(math.fsum(x * x for x in lambdas))
    expected = np.zeros_like(encoded[:, 0])
    expected[: len(operators)] = np.asarray(operators) / normalization
    block_error = float(np.max(np.abs(encoded[:, 0] - expected)))
    h = sum(o.conj().T @ o for o in operators)
    walked = simulation.run(j)
    cosine = 2 * h / normalization**2 - np.eye(simulation.dimension)
    compression_error = float(np.max(np.abs(walked[0, 0] - cosine)))
    # Verify the two-dimensional invariant spaces, including eigenphase sign.
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    cosine_values = np.clip(2 * eigenvalues / normalization**2 - 1, -1, 1)
    v = j @ eigenvectors
    wv = walked @ eigenvectors
    # Keep the companion unnormalized: dividing by sin(theta) is ill-conditioned
    # at zero and pi, where degenerate singular values are legitimate.
    companion = wv - v * cosine_values
    w_companion = simulation.run(companion)
    spectral_error = float(
        np.max(np.abs(w_companion + v * (1 - cosine_values**2) - companion * cosine_values))
    )
    disabled_error = (
        float(np.max(np.abs(simulation.run(j, enable=False) - j))) if task.controlled else None
    )
    norm_error = float(
        np.max(
            np.abs(
                encoded.reshape(-1, simulation.dimension).conj().T
                @ encoded.reshape(-1, simulation.dimension)
                - np.eye(simulation.dimension)
            )
        )
    )
    values = [
        block_error,
        compression_error,
        spectral_error,
        norm_error,
        disabled_error or 0,
    ]
    if any(not math.isfinite(x) or x > 1e-9 for x in values):
        raise Incorrect(f"Rectangular SOS/walk numerical cross-check failed: {values}")
    return {
        "checked": True,
        "system_basis_states": simulation.dimension,
        "block_max_abs_error": block_error,
        "walk_compression_max_abs_error": compression_error,
        "spectral_subspace_max_abs_error": spectral_error,
        "spectral_check": "unnormalized invariant-subspace identity, including endpoint singular values",
        "encoding_isometry_max_abs_error": norm_error,
        "disabled_max_abs_error": disabled_error,
        "phase_alignment_used": False,
        "target_energies": (eigenvalues + task.model["energy_offset"]).tolist(),
        "walk_eigenphases": np.arccos(cosine_values).tolist(),
    }


def verify_walk(task, stream_path):
    from qpe_lab.circuits.gadgets.prepare import alias_counts, walk_distributions

    doc = read_walk(stream_path)
    for key in ("label_bits", "term_bits", "mode_count", "controlled"):
        if doc[key] != getattr(task, key):
            raise Incorrect(f"Walk register mismatch: {key}")
    if doc["model_sha256"] != task.table_hash:
        raise Incorrect("Walk model hash mismatch")
    if doc.get("implementation", {}).get("kind") in ("paper", "paper_strict"):
        return verify_paper_walk(task, doc)
    if "paper" in task.model:
        raise Incorrect("Compact paper task requires the compact projector convention")
    weights = [[t["weight"] for t in g["terms"]] for g in task.model["generators"]]
    norms = [sum(row) for row in weights]
    outer_weights, inner_weights = walk_distributions(task.model)
    dynamic = doc.get("implementation", {}).get("kind") == "dynamic"
    alias = dynamic and doc["implementation"]["prepare"] == "alias"
    if alias:
        if "preparation" not in task.model:
            raise Incorrect("Alias cannot change the frozen exact-tree target")
        bits = task.model["preparation"]["alias_bits"]
        if (
            alias_counts(doc["alias"]["outer"], bits) != outer_weights
            or len(doc["alias"]["inner"]) != len(inner_weights)
            or any(
                alias_counts(table, bits) != expected
                for table, expected in zip(doc["alias"]["inner"], inner_weights, strict=True)
            )
        ):
            raise Incorrect("Alias table changes the fixed distribution")
    else:
        _check_prepare(doc["outer"], [(None, outer_weights)], task.label_bits)
        _check_prepare(doc["inner"], list(enumerate(inner_weights)), task.term_bits)
    _check_orbitals(task, doc["orbitals"])
    expected = {
        (a, b): (term["orbital"], term["spin"], term["pauli"], term["phase"])
        for a, generator in enumerate(task.model["generators"])
        for b, term in enumerate(generator["terms"])
        if term["weight"]
    }
    actual = {}
    for group in doc["select"]:
        if (
            not group["branches"]
            or not -1 <= group["orbital"] < len(doc["orbitals"])
            or group["spin"] not in (0, 1)
        ):
            raise Incorrect("Invalid SELECT basis group")
        for branch in group["branches"]:
            key = (branch["label"], branch["term"])
            if key in actual:
                raise Incorrect("Duplicate SELECT branch")
            actual[key] = (group["orbital"], group["spin"], branch["pauli"], branch["phase"])
    if actual != expected:
        raise Incorrect("SELECT has wrong/missing branches, relative phase, or orbital")
    sequence = [(step["block"], step["adjoint"]) for step in doc["steps"]]
    if sequence != [
        ("outer", False),
        ("inner", False),
        ("select", False),
        ("inner", True),
        ("rout", False),
        ("inner", False),
        ("select", True),
        ("inner", True),
        ("outer", True),
        ("rin", False),
    ]:
        raise Incorrect("Wrong rectangular walk order, adjoint, or reflection projector")
    normalization_squared = sum(x * x for x in norms) / (1 << (2 * task.model["coefficient_bits"]))
    numerical = (
        crosscheck_walk(task, doc)
        if task.mode_count <= 6 and task.label_bits + task.term_bits <= 8
        else {
            "checked": False,
            "reason": "large Fock space; accepted by fixed-template composition, not dense simulation",
        }
    )
    cost = count_walk(task, doc)
    result = {
        "correctness_status": "bounded_verified"
        if cost["status"] == "complete"
        else "exact_verified",
        "cost": cost,
        "optimization_contract": {
            **task.contract,
            "task_sha256": task.task_hash,
            "model_sha256": task.table_hash,
            "target": "fixed quantized SOS Hamiltonian; fixed normalization and energy offset",
            "quantization_error_separate_from_compilation": task.model.get("precision", {}),
            "query_cost_scope": "one controlled walk; normalization fixed across candidates",
        },
        "evidence": {
            "method": "exact rational PREPARE state semantics + integer orbital identities + complete SELECT branch map + rectangular walk composition",
            "select_implementation": doc.get("select_implementation", {"kind": "grouped"}),
            "scope": "ideal algebraic/dyadic rotation model; clean elementary control workspace; all coherent selector and Fock inputs",
            "generators": len(norms),
            "select_branches": len(actual),
            "basis_groups": len(doc["select"]),
            "label_bits": task.label_bits,
            "term_bits": task.term_bits,
            "padding": "identity SELECT and identity inner preparation for unused labels/terms",
            "input_projector": "label=0,term=0",
            "output_projector": "term=0; label retained",
            "controlled": task.controlled,
            "control_strategy": "control both reflections; U and Udag cancel when disabled",
        },
        "normalization": {
            "lambda_sqrt_squared": normalization_squared,
            "Lambda": normalization_squared / 2,
            "energy_offset": task.model["energy_offset"],
            "decode": "E = energy_offset + Lambda * (1 + cos(theta))",
            "target": "frozen quantized SOS model",
        },
        "call_graph": {
            "outer_prepare": 1,
            "outer_prepare_adjoint": 1,
            "inner_prepare": 2,
            "inner_prepare_adjoint": 2,
            "select": 1,
            "select_adjoint": 1,
            "input_reflection": 1,
            "output_reflection": 1,
        },
        "numerical_crosscheck": numerical,
    }
    if dynamic:
        result["evidence"].update(
            method=("exact alias cell distributions" if alias else "exact rational PREPARE state")
            + " + complete SELECT branches + certified component composition",
            select_implementation=doc["implementation"],
            output_projector=(
                "term=0,inner_sigma=0; retain outer alias garbage"
                if alias
                else "term=0; label retained"
            ),
            input_projector=(
                "label=0,term=0,outer_sigma=0,inner_sigma=0; workspace starts zero"
                if alias
                else "label=0,term=0"
            ),
            measurement_semantics="X measurement/reset and explicit conditional phase correction",
            scope=(
                "reachable clean-workspace encoding subspace; arbitrary coherent system input; "
                "not arbitrary dirty-ancilla inputs"
            ),
            inverse_strategy="fresh inverse construction, never reverse a measured gate stream",
            catalyst="phase-gradient prepared and retired within each walk",
        )
        result["numerical_crosscheck"]["representation"] = (
            "compressed preparation isometry with the same encoded operators and walk spectrum; "
            "physical dynamic gates are tested separately at component level"
        )
        result["optimization_contract"]["target"] = (
            "fixed alias-cell SOS model"
            if "preparation" in task.model
            else "fixed quantized SOS model"
        )
        result["physical_dynamic_crosscheck"] = crosscheck_dynamic_walk(task, doc)
    return result


def crosscheck_paper_walk(task, doc, *, expander=None):
    """All system columns of a small physical circuit, including measurements."""
    from qpe_lab.circuits.gadgets.synthesis import expand_walk
    from qpe_lab.circuits.sim import SparseStatevectorSimulator

    prep = task.model["paper"]
    if (
        task.mode_count > 4
        or len(prep["outer"]) > 4
        or prep["bases"] > 1
        or prep["alias_bits"] > 1
        or task.model["angle_bits"] > 3
    ):
        return {
            "checked": False,
            "reason": "physical state-space limit; full-size acceptance uses fixed-template composition",
        }
    operators = reference_generators(task.model)
    norm2 = sum(sum(t["weight"] for t in g["terms"]) ** 2 for g in task.model["generators"]) / (
        1 << (2 * task.model["coefficient_bits"])
    )
    if "uniform_bits" in prep:
        from qpe_lab.circuits.gadgets.prepare import flagged_uniform_parameters

        norm2 /= flagged_uniform_parameters(
            len(prep["outer"]), task.label_bits, prep["uniform_bits"]
        )["success_probability"]
    dimension = 1 << task.mode_count
    identity = np.eye(dimension, dtype=complex)
    h = sum(o.conj().T @ o for o in operators)
    gates, circuit = (expander or expand_walk)(task, doc)
    gates = list(gates)
    error = 0.0
    for enabled in (False, True) if task.controlled else (True,):
        base = 1 << (task.logical_qubits - 1) if enabled and task.controlled else 0
        indices = [base | (i << (task.label_bits + task.term_bits)) for i in range(dimension)]
        initial = dict(zip(indices, identity, strict=True))
        expected = 2 * h / norm2 - identity if enabled else identity
        for seed in (0, 1, 7):
            output = SparseStatevectorSimulator(circuit.peak_qubits).run(gates, initial, seed=seed)
            compressed = np.array([output.get(i, np.zeros(dimension)) for i in indices])
            columns = np.array(list(output.values()))
            error = max(
                error,
                float(np.max(np.abs(compressed - expected))),
                float(np.max(np.abs(columns.conj().T @ columns - identity))),
            )
            if not enabled:
                error = max(
                    error,
                    max(
                        (float(np.max(np.abs(v))) for i, v in output.items() if i not in initial),
                        default=0.0,
                    ),
                )
    if error > 1e-9:
        raise Incorrect(
            f"Physical compact walk disagrees with ladder-operator Hamiltonian: {error}"
        )
    return {
        "checked": True,
        "max_error": error,
        "system_columns": dimension,
        "seeds": [0, 1, 7],
        "scope": "phase-sensitive compressed walk, disabled-control identity and full-column isometry; sampled measurement records",
    }


def walk_normalization(task):
    """Energy conversion for the fixed target, including flagged preparation."""
    norm2 = sum(sum(t["weight"] for t in g["terms"]) ** 2 for g in task.model["generators"]) / (
        1 << (2 * task.model["coefficient_bits"])
    )
    result = {
        "lambda_sqrt_squared": norm2,
        "Lambda": norm2 / 2,
        "energy_offset": task.model["energy_offset"],
        "decode": "E = energy_offset + Lambda * (1 + cos(theta))",
    }
    prep = task.model.get("paper", {})
    if "uniform_bits" in prep:
        from qpe_lab.circuits.gadgets.prepare import flagged_uniform_parameters

        p = flagged_uniform_parameters(len(prep["outer"]), task.label_bits, prep["uniform_bits"])[
            "success_probability"
        ]
        result.update(
            uniform_success_probability=p,
            effective_lambda_sqrt_squared=norm2 / p,
            Lambda=norm2 / (2 * p),
        )
    return result


def verify_strict_outer_alias(task, doc):
    """Prove the actual outer alias load/use/erase on its full address domain.

    Uniform-state rotations stay outside this classical reversible region.
    Copying both the selected label and alias garbage into symbolic observers
    checks coherent consumers while requiring every measurement phase to cancel.
    """
    from qpe_lab.circuits.gadgets.synthesis import expand_walk
    from qpe_lab.circuits.sim import SymbolicSimulator

    la, bits = task.label_bits, task.model["paper"]["alias_bits"]
    if la > 12:
        raise Incorrect("Strict outer proof exceeds exact address-domain limit")
    low_bits = min(bits, 12 - la)
    if bits - low_bits > 8:
        raise Incorrect("Strict outer proof exceeds exact symbolic-sigma limit")
    _, circuit = expand_walk(task, doc)
    regs = circuit.registers
    stages = {}
    for name in ("outer", "outer_adjoint"):
        regions = [(start, end) for block, start, end in circuit.blocks if block == name]
        if len(regions) != 1:
            raise Incorrect("Strict outer alias requires one forward and one inverse region")
        start, end = regions[0]
        stages[name] = circuit.gates[start:end]
    forward, inverse = stages["outer"], stages["outer_adjoint"]
    sigma_h = [("h", (q,), None) for q in regs["outer_sigma"]]
    spin_h = ("h", tuple(regs["spin_outer"]), None)
    starts = [i for i, gate in enumerate(forward) if gate in sigma_h]
    ends = [i for i, gate in enumerate(inverse) if gate in sigma_h]
    if (
        len(starts) != bits
        or len(ends) != bits
        or forward[starts[0] : starts[0] + bits] != sigma_h
        or inverse[ends[0] : ends[0] + bits] != sigma_h
        or forward[-1] != spin_h
        or inverse[0] != spin_h
    ):
        raise Incorrect("Cannot isolate outer alias from its uniform preparation")
    selected = regs["label"] + regs["outer_word"] + regs["outer_flag"]
    observer = list(range(circuit.peak_qubits, circuit.peak_qubits + len(selected)))
    program = forward[starts[-1] + 1 : -1]
    program += [("cx", (q, out), None) for q, out in zip(selected, observer, strict=True)]
    program += inverse[1 : ends[0]]
    table = doc["alias"]["outer"]
    checker = SymbolicSimulator(la + low_bits, bits - low_bits + len(observer))
    checker.initialize(
        [
            (0, regs["label"] + regs["outer_sigma"][:low_bits]),
            (0, regs["outer_sigma"][low_bits:] + observer),
        ]
    )
    checker.apply_elementary(program)
    checker.assert_clean()
    for q in regs["label"] + regs["outer_sigma"]:
        if checker.value(q) != checker.initial[q]:
            raise Incorrect("Outer alias changes an address or sigma register")
    alt = table["alt"] + [0] * ((1 << la) - len(table["alt"]))
    keep = table["keep"] + [0] * ((1 << la) - len(table["keep"]))
    label_mask = (1 << la) - 1
    # Independent integer comparison truth table, transformed in high sigma bits.
    coefficients = [
        sum(
            1 << lane
            for lane in range(checker.addresses)
            if ((high << low_bits) | (lane >> la)) >= keep[lane & label_mask]
        )
        for high in range(1 << (bits - low_bits))
    ]
    for bit in range(bits - low_bits):
        for mask in range(len(coefficients)):
            if mask >> bit & 1:
                coefficients[mask] ^= coefficients[mask ^ (1 << bit)]
    flag = {m: word for m, word in enumerate(coefficients) if word}

    def constant(values, bit):
        mask = sum(
            1 << lane for lane in range(checker.addresses) if values[lane & label_mask] >> bit & 1
        )
        return {0: mask} if mask else {}

    expected = []
    for bit in range(la):
        original = checker.initial[regs["label"][bit]]
        delta = checker.xor(original, constant(alt, bit))
        expected.append(checker.xor(original, checker.mul(flag, delta)))
    for bit in range(la):
        delta = checker.xor(checker.initial[regs["label"][bit]], constant(alt, bit))
        expected.append(checker.xor(constant(alt, bit), checker.mul(flag, delta)))
    expected += [constant(keep, bit) for bit in range(bits)] + [flag]
    for q, value in zip(observer, expected, strict=True):
        if checker.value(q) != checker.xor(checker.initial[q], value):
            raise Incorrect("Outer alias outputs incorrect selected label or garbage")
    return {
        "verified": True,
        "addresses": 1 << la,
        "sigma_values": 1 << bits,
        "label_sigma_combinations": 1 << (la + bits),
        "consumer_bits": len(observer),
        "finite_sigma_bits": low_bits,
        "symbolic_sigma_bits": bits - low_bits,
        "measurement_coverage": "all outcomes, exact symbolic phase polynomial",
        "scope": "complete outer alias load/use/erase; independent integer threshold oracle; all addresses and all sigma words",
    }


def verify_strict_paper_tables(task, doc):
    """Exact joint load/use/erase on all addresses, sigma words and outcomes.

    Batch high address bits, enumerate low address/sigma bits, and retain high
    sigma and consumer bits symbolically. Integer threshold truth tables give
    an independent oracle for both alias paths, without boundary sampling.
    """
    from qpe_lab.circuits.gadgets.prepare import paper_layout
    from qpe_lab.circuits.gadgets.synthesis import expand_walk
    from qpe_lab.circuits.sim import SymbolicSimulator

    prep, layout = task.model["paper"], paper_layout(task.model)
    bits = prep["alias_bits"]
    tb, la = task.term_bits, task.label_bits
    address_low = min(6, la + tb)
    sigma_low = min(bits, 12 - address_low)
    if la + tb > 12 or bits - sigma_low > 8:
        raise Incorrect("Strict joint proof exceeds exact address/sigma limits")
    _, circuit = expand_walk(task, doc)
    regs = circuit.registers
    width = (task.model["spatial_modes"] - 1) * task.model["angle_bits"]
    metadata = regs["factor_metadata"]
    consumer_bits = width + 1 + len(metadata)
    observer = list(range(circuit.peak_qubits, circuit.peak_qubits + consumer_bits))
    address = regs["term"] + regs["label"]
    low_size = 1 << address_low
    group_count = 1 << (la + tb - address_low)
    names = ("inner", "angle_load", "angle_erase", "inner_adjoint")
    stages = {name: [] for name in names}
    for name, start, end in circuit.blocks:
        if name in stages:
            stages[name].append(circuit.gates[start:end])
    if any(len(v) != 2 for v in stages.values()):
        raise Incorrect("Strict walk must verify both load/use/erase invocations")
    programs = []
    for invocation in range(2):
        erase = stages["angle_erase"][invocation]
        prefix = erase[: width + 1]
        if len(prefix) != width + 1 or any(g[0] != "hmr" for g in prefix):
            raise Incorrect("Angle erase must measure every angle bit and the active flag")
        angle = [g[1][0] for g in prefix]
        if len(set(angle)) != width + 1:
            raise Incorrect("Angle erase measures a duplicate wire")
        program = [g for g in stages["inner"][invocation] if g[0] != "h"]
        program += stages["angle_load"][invocation]
        program += [
            ("cx", (q, out), None) for q, out in zip(angle + metadata, observer, strict=True)
        ]
        program += [("cz", (regs["inner_flag"][0], regs["inner_word"][-1]), None)]
        program += erase
        program += [g for g in stages["inner_adjoint"][invocation] if g[0] != "h"]
        programs.append(program)
    orbital_words = [
        sum(v << (i * task.model["angle_bits"]) for i, v in enumerate(row))
        for row in doc["orbital_angles"]
    ]
    for group in range(group_count):
        keeps, original, alternate, signs, alt_signs = [], [], [], [], []
        for low in range(low_size):
            value = (group << address_low) | low
            x, b = value >> tb, value & ((1 << tb) - 1)
            meta, orig_word, alt_word, sign, alt_sign, keep = 0, 0, 0, 0, 0, 0
            if x < len(layout["sf"]):
                table = doc["alias"]["inner"][x]
                a, keep = table["alt"][b], table["keep"][b]
                r = x // prep["copies"]
                orig_word = (
                    (orbital_words[r * prep["bases"] + b] | (1 << width))
                    if b < prep["bases"]
                    else 0
                )
                alt_word = (
                    (orbital_words[r * prep["bases"] + a] | (1 << width))
                    if a < prep["bases"]
                    else 0
                )
                source = task.model["generators"][layout["sf"][x]]["terms"]
                row_signs = [t["phase"] // 2 for t in source[1::2]] + [source[0]["phase"] // 2]
                sign, alt_sign = row_signs[b], row_signs[a]
                meta = 1 | (r << 2)
            elif x < len(prep["outer"]):
                orig_word = alt_word = orbital_words[
                    prep["rank"] * prep["bases"] + x - len(layout["sf"])
                ]
                gi = layout["single"][x - len(layout["sf"])][0]
                meta = 2 if task.model["generators"][gi]["name"].startswith("Q-") else 0
            keeps.append(keep)
            original.append(orig_word | (meta << (width + 1)))
            alternate.append(alt_word | (meta << (width + 1)))
            signs.append(sign)
            alt_signs.append(alt_sign)
        lanes = 1 << (address_low + sigma_low)
        repeated = ((1 << lanes) - 1) // ((1 << low_size) - 1)
        coefficients = [
            sum(
                1 << lane
                for lane in range(lanes)
                if ((high << sigma_low) | (lane >> address_low)) >= keeps[lane & (low_size - 1)]
            )
            for high in range(1 << (bits - sigma_low))
        ]
        for bit in range(bits - sigma_low):
            for mask in range(len(coefficients)):
                if mask >> bit & 1:
                    coefficients[mask] ^= coefficients[mask ^ (1 << bit)]
        flag = {m: word for m, word in enumerate(coefficients) if word}

        def constant(values, bit, repeated=repeated):
            mask = repeated * sum(1 << i for i, v in enumerate(values) if v >> bit & 1)
            return {0: mask} if mask else {}

        for invocation, program in enumerate(programs):
            checker = SymbolicSimulator(address_low + sigma_low, bits - sigma_low + consumer_bits)
            checker.initialize(
                [
                    (0, address[:address_low] + regs["inner_sigma"][:sigma_low]),
                    (0, regs["inner_sigma"][sigma_low:] + observer),
                ]
            )
            for bit, q in enumerate(address[address_low:]):
                checker.wires[q] = checker.one.copy() if group >> bit & 1 else {}
                checker.protected.add(q)
                checker.initial[q] = checker.wires[q].copy()
            checker.apply_elementary(program)
            expected_sign = checker.xor(
                constant(signs, 0),
                checker.mul(flag, checker.xor(constant(signs, 0), constant(alt_signs, 0))),
            )
            if checker.phase != expected_sign:
                raise Incorrect(f"Wrong sign in group {group}, invocation {invocation}")
            checker.phase = {}
            checker.assert_clean()
            for q in address + regs["inner_sigma"]:
                if checker.value(q) != checker.initial[q]:
                    raise Incorrect("Address/sigma changed")
            for bit, q in enumerate(observer):
                first = constant(original, bit)
                result = checker.xor(
                    first, checker.mul(flag, checker.xor(first, constant(alternate, bit)))
                )
                if checker.value(q) != checker.xor(checker.initial[q], result):
                    raise Incorrect(f"Wrong data in group {group}, bit {bit}")
    return {
        "verified": True,
        "addresses": 1 << (la + tb),
        "sigma_values": 1 << bits,
        "address_sigma_combinations": 1 << (la + tb + bits),
        "consumer_bits": consumer_bits,
        "angle_and_active_bits": width + 1,
        "metadata_bits": len(metadata),
        "invocations_verified": 2,
        "proof_groups": group_count,
        "finite_sigma_bits": sigma_low,
        "symbolic_sigma_bits": bits - sigma_low,
        "measurement_coverage": "all outcomes, exact symbolic phase polynomial",
        "scope": "joint alias, SF/D1 angles, metadata, sign and measured erase; all addresses and all sigma words; independent integer threshold oracle",
    }


def verify_paper_walk(task, doc):
    """Certify finite aliases and the fixed compact compiler, then count gates."""
    from qpe_lab.circuits.gadgets.prepare import alias_counts, paper_layout

    if "paper" not in task.model:
        raise Incorrect("Paper circuit cannot replace a different frozen Hamiltonian")
    prep, layout = task.model["paper"], paper_layout(task.model)
    strict = doc["implementation"]["kind"] == "paper_strict"
    if strict != ("uniform_bits" in prep):
        raise Incorrect("Circuit uniform preparation changes the frozen projector")
    tables, bits = doc["alias"], prep["alias_bits"]
    if (
        alias_counts(tables["outer"], bits, compact=True) != prep["outer"]
        or len(tables["inner"]) != len(prep["inner"])
        or any(
            alias_counts(table, bits, compact=True) != row
            for table, row in zip(tables["inner"], prep["inner"], strict=True)
        )
        or any(k >= 1 << bits for t in [tables["outer"], *tables["inner"]] for k in t["keep"])
    ):
        raise Incorrect("Compact alias table changes the target or uses an endpoint bit")
    if doc["orbital_angles"] != task.model["orbital_angles"]:
        raise Incorrect("Compact angle table changes the frozen Hamiltonian")
    cost = count_walk(task, doc)
    totals = {}
    for block in cost["block_costs"]:
        name = block["block"]
        totals[name] = totals.get(name, 0) + block["gates"].get("ccx", 0)
    breakdown = {
        "outer": totals["outer"] + totals["outer_adjoint"],
        "inner_and_metadata": totals["inner"]
        + totals["inner_adjoint"]
        + totals.get("metadata_load", 0)
        + totals.get("metadata_erase", 0),
        "angles": totals["angle_load"] + totals["angle_erase"],
        "select": totals["majorana_and_rotations"],
        "rout": totals["rout"],
        "rin": totals["rin"],
    }
    paper_parameters = (
        task.model["spatial_modes"],
        prep["rank"],
        prep["bases"],
        prep["copies"],
        bits,
        task.model["angle_bits"],
    ) == (20, 14, 15, 5, 11, 15)
    published = dict(zip(breakdown, (175, 806, 518, 2374, 16, 35), strict=True))
    result = {
        "correctness_status": "bounded_verified"
        if cost["status"] == "complete"
        else "exact_verified",
        "cost": cost,
        "optimization_contract": {
            **task.contract,
            "task_sha256": task.task_hash,
            "model_sha256": task.table_hash,
            "target": "fixed compact aliases, angles, normalization and offset",
            "quantization_error_separate_from_compilation": task.model["precision"],
            "query_cost_scope": "one complete controlled walk, including gradient preparation",
        },
        "evidence": {
            "method": "exact alias cell counts + validated spin-paired SOS factors + fixed compact gate-template composition",
            "scope": "reachable clean-workspace subspace; arbitrary coherent system input; large-instance physical simulation not performed",
            "compact_outer_labels": len(prep["outer"]),
            "inner_indices": prep["bases"] + 1,
            "spin_free_factors": len(layout["sf"]),
            "one_body_spin_pairs": len(layout["single"]),
            "input_projector": "x,b,sigma_outer,sigma_inner,spin_outer,spin_inner,uniform auxiliaries = 0",
            "output_projector": "b,sigma_inner,spin_inner,inner uniform auxiliary = 0; retain outer registers",
            "measurement_semantics": "explicit X measurements, classical XORs and conditional phase corrections",
            "inverse_strategy": "fresh inverse construction; QROAM phase debt retired at original address",
            "uniform_preparation": "exact one-round amplification; not paper's approximate seven-bit rotation",
            "precision_convention": "deterministic nearest angles/largest-remainder aliases, not Appendix D random rounding",
        },
        "normalization": walk_normalization(task),
        "physical_dynamic_crosscheck": crosscheck_paper_walk(task, doc),
        "paper_comparison": {
            "scope": "Appendix B circuit structure with explicit total-domain padding and complete gate accounting; not an assertion of exact Table V totals",
            "matches_fe2s2_resource_parameters": paper_parameters,
            "published_fe2s2_walk_toffoli": 3924 if paper_parameters else None,
            "published_fe2s2_qubits": 463 if paper_parameters else None,
            "actual_ccx_by_component": breakdown,
            "published_toffoli_by_component": published if paper_parameters else None,
            "ccx_difference": {name: count - published[name] for name, count in breakdown.items()}
            if paper_parameters
            else None,
            "differences": [
                "exact uniform-range preparation instead of approximate rotation and success flag",
                "total-domain QROM controls; two explicit sign bits",
                "one shared outer metadata load and explicit phase-corrected cleanup",
                "power-of-two uniform ranges use only Hadamards",
                "gradient preparation/retirement included; no amortization over phase estimation",
            ],
        },
    }
    if strict:
        from qpe_lab.circuits.gadgets.prepare import flagged_uniform_parameters

        phase_adder = doc["implementation"].get("gradient_adder", "general") == "phase"
        uniform = flagged_uniform_parameters(
            len(prep["outer"]), task.label_bits, prep["uniform_bits"]
        )
        result["evidence"].update(
            output_projector="success_outer=1; b,sigma_inner,spin_inner,inner uniform/success auxiliaries=0",
            uniform_preparation="Sanders et al. III E 2; two constant-gradient rotations; explicit success flag",
            uniform_parameters=uniform,
            sign_handling="one delta-sign bit; original signs folded into measured inner unlookup",
            storage_schedule="SF QROAM banks measured before one-body angles; shared x unary traversal",
            angle_erase="two-dimensional SF unlookup; one-body phases merged with inner alias and metadata erase",
            joint_table_proof=verify_strict_paper_tables(task, doc),
            outer_table_proof=verify_strict_outer_alias(task, doc),
            sf_lookup_control={
                "kind": doc["implementation"].get("sf_lookup_control", "explicit"),
                "invariant": "non-SF branches have b=r=0 after inner alias preparation",
                "verification": "both actual joint load/use/erase invocations",
            },
            gradient_adder={
                "kind": "phase" if phase_adder else "general",
                "ccx_per_givens": max(0, 2 * task.model["angle_bits"] - 3)
                if phase_adder
                else 2 * task.model["angle_bits"] - 1,
                "contract": "controlled modular addition on a phase gradient; highest target bit is |->"
                if phase_adder
                else "controlled modular addition on arbitrary target bits",
                "phase_tail": "high addend and final carry replaced by CZ/Z; lower sum and all inputs restored as specified"
                if phase_adder
                else None,
            },
            inner_phase_repair={
                "method": "joint_boolean_planes"
                if "phase_unlookup_bits" in doc["implementation"]
                else "mixed_sf_one_body",
                "plane_bits": doc["implementation"].get("phase_unlookup_bits"),
                "padding": "exact zero phase outside the frozen domain",
            },
            outer_phase_repair={
                "method": "joint_boolean_planes"
                if "outer_phase_unlookup_bits" in doc["implementation"]
                else "deferred_window",
                "plane_bits": doc["implementation"].get("outer_phase_unlookup_bits"),
                "padding": "exact zero phase outside the frozen domain",
            },
        )
        result["paper_comparison"].update(
            scope="Appendix B executable reconstruction; published estimate and executed gates reported separately",
            exact_published_gate_counts_reproduced=False,
            differences=[
                "author estimator ignores O(1) constants and charges binary B+1 uniform ranges",
                "total-domain padding controls retained in executable unary traversals",
                (
                    "phase-gradient-specific controlled adder executes max(0,2*beta-3) CCX"
                    if phase_adder
                    else "measured controlled adder executes 2*beta-1 CCX, versus paper's 2*beta estimate"
                ),
                "one-body angle load shares inner x traversal; its CCX assigned to inner in actual block breakdown",
                "finite aliases/angles use frozen deterministic rounding; not an identified author stochastic sample",
                "gradient preparation/retirement included per exported walk, not amortized",
            ],
        )
        aligned = dict(breakdown)
        # The N one-body angle leaves share the forward x traversal and
        # therefore live in the executable "inner" block, not "angle_load".
        aligned["inner_and_metadata"] -= 2 * task.model["spatial_modes"]
        aligned["angles"] += 2 * task.model["spatial_modes"]
        result["paper_comparison"]["actual_ccx_paper_component_attribution"] = aligned
        baseline_layout = {
            "kind": "paper_strict",
            "outer_block": 2,
            "inner_block": 8,
            "outer_unlookup": 8,
            "inner_unlookup": 32,
        }
        implementation = doc["implementation"]
        specialized = implementation.get("specialize_constants", False)
        result["evidence"]["constant_specialization"] = {
            "enabled": specialized,
            "invariants": [
                "binary inner uniform auxiliary remains zero and success is one",
                "active implies spin-free; qtype is zero on spin-free rows",
            ]
            if specialized
            else [],
        }
        # Historical explanations describe one baseline layout, not arbitrary
        # optimized candidates with different register lifetimes.
        if (
            paper_parameters
            and not specialized
            and {k: v for k, v in implementation.items() if k != "specialize_constants"}
            == baseline_layout
        ):
            explanations = {
                "outer": {
                    "uniform_constant_terms": 4,
                    "forward_total_domain_controls": 1,
                    "inverse_total_domain_controls": 1,
                },
                "inner_and_metadata": {
                    "four_binary_uniform_calls_have_no_toffoli": -64,
                    "inverse_endpoint_constants": -2,
                },
                "angles": {"two_unary_erase_endpoint_constants": -2},
                "select": {
                    "76_controlled_adders_cost_2beta_minus_1": -76,
                    "two_explicit_spin_selector_computations": 4,
                    "two_spin_blocked_Jordan_Wigner_parity_controls": 2,
                },
                "rout": {"explicit_uniform_and_success_projector_controls": 3},
                "rin": {"four_explicit_uniform_and_success_projector_controls": 4},
            }
            result["paper_comparison"].update(
                baseline_delta_explanations=explanations,
                baseline_unexplained_ccx={
                    key: aligned[key] - published[key] - sum(explanations[key].values())
                    for key in aligned
                },
                baseline_qubit_accounting={
                    "published_persistent": 416,
                    "published_max_temporary": 47,
                    "explicit_persistent_during_select": 423,
                    "extra_persistent_wires": {
                        "two_alias_flags": 2,
                        "inner_sign_bit": 1,
                        "two_spin_bits": 2,
                        "walk_control": 1,
                        "identity_flag": 1,
                    },
                    "actual_select_temporary": {"spin_selector": 1, "controlled_adder": 29},
                    "note": "423+30=453 actual lifetime peak; 416+47=463 is the author's coarser resource accounting",
                },
            )
    return result
