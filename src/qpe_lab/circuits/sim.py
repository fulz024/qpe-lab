"""Discrete symbolic simulation and numerical Gaussian/Fock cross-checks.

In SymbolicSimulator each wire is an algebraic normal form (ANF) in target input bits.
Coefficients are bitsets covering EVERY address. Measurement draws are independent
formal variables, and their phase debts are cancelled algebraically. No sampling
or floating point is used in that backend. Unsupported expressions fail closed.
GaussianSimulator and DenseFockSimulator instead use floating-point matrices and
never supply an exact equivalence verdict by themselves.
exact_clifford_t_columns supplies bounded local proofs over rational cyclotomic
coefficients, without numerical tolerances or global-phase alignment.
"""

import math

import numpy as np
from uncompute.sim import Simulator

__all__ = [
    "DenseFockSimulator",
    "GaussianSimulator",
    "Incorrect",
    "Simulator",
    "SymbolicSimulator",
    "Unsupported",
    "exact_clifford_t_columns",
    "fock_lift",
]


class Unsupported(ValueError):
    pass


class Incorrect(ValueError):
    pass


class BooleanDiagram:
    """Canonical reduced ordered BDDs. Limits cause unsupported, never acceptance."""

    def __init__(self, limit=250_000):
        self.one = 1
        self.nodes = [None, None]
        self.unique, self.cache = {}, {}
        self.limit = limit

    def node(self, variable, low, high):
        if low == high:
            return low
        key = variable, low, high
        if key not in self.unique:
            if len(self.nodes) >= self.limit:
                raise Unsupported("Exact Boolean diagram node budget exceeded")
            self.unique[key] = len(self.nodes)
            self.nodes.append(key)
        return self.unique[key]

    def apply(self, op, a, b):
        if a > b:
            a, b = b, a
        if op == "xor":
            if a == b:
                return 0
            if a == 0:
                return b
        elif op == "and":
            if a == 0:
                return 0
            if a == 1 or a == b:
                return b
        else:
            raise ValueError("Unknown Boolean operation")
        if a < 2 and b < 2:
            return (a ^ b) if op == "xor" else (a & b)
        key = op, a, b
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= 4 * self.limit:
            raise Unsupported("Exact Boolean diagram work budget exceeded")
        va = self.nodes[a][0] if a >= 2 else math.inf
        vb = self.nodes[b][0] if b >= 2 else math.inf
        variable = min(va, vb)
        al, ah = self.nodes[a][1:] if va == variable else (a, a)
        bl, bh = self.nodes[b][1:] if vb == variable else (b, b)
        result = self.node(variable, self.apply(op, al, bl), self.apply(op, ah, bh))
        self.cache[key] = result
        return result


def monomial_signature(gates, manager, inputs, *, clean=True, forms=None):
    """Exact basis map and phase for a monomial circuit with measured uncompute.

    Internal X-measurement records may differ. Each record has a uniform
    magnitude; only input-independent record phases may remain. No sampling.
    Classical feedback may change phase, but not the basis map in this backend.
    """
    one = manager.one
    if forms is None:
        wires = {q: manager.node(i, 0, one) for i, q in enumerate(sorted(inputs))}
    else:
        # Preserve the trusted prefix's affine correlations and constant bits.
        # Uniform probes of these variables induce uniform probes of the image;
        # sampling each output wire separately would dilute correlated errors.
        variables = sorted(set().union(*(terms - {(2,)} for terms in forms.values())))
        values = {v: manager.node(i, 0, one) for i, v in enumerate(variables)}
        wires = {}
        for q, terms in forms.items():
            value = one if (2,) in terms else 0
            for v in sorted(terms - {(2,)}):
                value = manager.apply("xor", value, values[v])
            wires[q] = value
    initial = dict(wires)
    records, debts = {}, {}
    phase = [0, 0, 0]
    xor = lambda a, b: manager.apply("xor", a, b)
    product = lambda a, b: manager.apply("and", a, b)

    def bit(q):
        if q not in wires:
            if not clean:
                raise Unsupported("Unknown scratch input in Boolean proof")
            wires[q] = 0
        return wires[q]

    def add_phase(expression, power):
        for i in range(3):
            if power % 8 >> i & 1:
                carry = expression
                for j in range(i, 3):
                    phase[j], carry = xor(phase[j], carry), product(phase[j], carry)

    cursor = 0
    while cursor < len(gates):
        name, qs, value = gates[cursor]
        if name in {"h", "cx"} and value is None:
            from qpe_lab.circuits.gadgets.synthesis import (
                ccx_clifford_t,
                zero_target_ccx_clifford_t,
            )

            # These are checked concrete words, not candidate-supplied macros.
            # The four-T word is only lifted after this simulator proves its
            # zero-target precondition on every input.
            matched = False
            for length, a_slot, b_slot, factory, adjoint in (
                (15, 3, 1, ccx_clifford_t, False),
                (15, 3, 1, ccx_clifford_t, True),
                (10, 4, 2, zero_target_ccx_clifford_t, False),
            ):
                if (name == "cx") != adjoint:
                    continue
                word = gates[cursor : cursor + length]
                if adjoint:
                    inverse = {"t": "tdg", "tdg": "t", "s": "sdg", "sdg": "s"}
                    word = [(inverse.get(n, n), ws, v) for n, ws, v in reversed(word)]
                if (
                    len(word) != length
                    or word[0][0] != "h"
                    or word[a_slot][0] != "cx"
                    or word[b_slot][0] != "cx"
                ):
                    continue
                a, b, t = word[a_slot][1][0], word[b_slot][1][0], word[0][1][0]
                if len({a, b, t}) != 3 or word != list(factory(a, b, t)):
                    continue
                if length == 10 and bit(t) != 0:
                    raise Unsupported("Four-T AND target is not proved zero")
                name, qs, value = "ccx", (a, b, t), None
                cursor += length
                matched = True
                break
            if not matched:
                if name == "h":
                    raise Unsupported("Hadamard outside a recognized exact reversible word")
                cursor += 1
        else:
            cursor += 1
        condition = value.get("condition") if isinstance(value, dict) else None
        if condition is not None and condition not in records:
            raise Incorrect("Undefined feedback record")
        if name == "oxor":
            target, left, right = value
            if target in records or right not in records or (left != -1 and left not in records):
                raise Incorrect("Invalid classical dependency")
            records[target] = (frozenset() if left == -1 else records[left]) ^ records[right]
            continue
        if name == "hmr":
            if value in records:
                raise Incorrect("Repeated measurement identifier")
            records[value] = frozenset({value})
            debts[value] = bit(qs[0])
            wires[qs[0]] = 0
            continue
        if name == "assert_zero":
            if bit(qs[0]) != 0:
                raise Incorrect("Unproved zero assertion")
            continue
        if name in {"x", "cx", "ccx", "swap"}:
            if condition is not None:
                raise Unsupported("Basis-changing measurement feedback requires channel backend")
            if name == "swap":
                wires[qs[0]], wires[qs[1]] = bit(qs[1]), bit(qs[0])
            else:
                control = one
                for q in qs[:-1]:
                    control = product(control, bit(q))
                wires[qs[-1]] = xor(bit(qs[-1]), control)
            continue
        powers = {"z": 4, "cz": 4, "ccz": 4, "neg": 4, "s": 2, "sdg": 6, "t": 1, "tdg": 7}
        if name not in powers:
            raise Unsupported("Circuit is not in the exact monomial sublanguage")
        expression = one
        for q in qs:
            expression = product(expression, bit(q))
        if condition is None:
            add_phase(expression, powers[name])
        elif powers[name] == 4:
            for record in records[condition]:
                debts[record] = xor(debts.get(record, 0), expression)
        else:
            raise Unsupported("Non-Pauli feedback requires channel backend")
    if any(debt not in (0, one) for debt in debts.values()):
        raise Incorrect("Measurement leaves input-dependent phase or dephasing")
    return {q: bit(q) for q in sorted(wires)}, tuple(phase), initial


def verify_monomial_equivalence(before, after, inputs, *, clean=True, forms=None):
    manager = BooleanDiagram()
    try:
        left, lp, _ = monomial_signature(before, manager, inputs, clean=clean, forms=forms)
    except Incorrect as exc:
        raise Unsupported("Reference is outside the monomial unitary proof domain") from exc
    right, rp, _ = monomial_signature(after, manager, inputs, clean=clean, forms=forms)
    if any(left.get(q, 0) != right.get(q, 0) for q in left.keys() | right.keys()) or lp != rp:
        raise Incorrect("Exact Boolean input map or coherent phase differs")
    return {
        "method": "exact reduced Boolean diagrams and all-record phase debt",
        "nodes": len(manager.nodes),
        "input_bits": len(inputs),
        "sampling": False,
    }


def verify_gradient_monomial(before, after, forms, resources):
    """Prove catalytic monomial actions using p + y_out - y_in modulo 2**g.

    Logical outputs and the corrected phase must be independent of EVERY
    gradient input bit. This establishes restoration and disentanglement,
    not just agreement after tracing out the borrowed register.
    """
    # Prefix-derived logical support may retain historical gradient symbols.
    # A proved product resource has its own independent current coordinate.
    forms = forms | {q: frozenset({(3, q)}) for q in resources}
    manager = BooleanDiagram()
    bits = max([3] + [b for b, _ in resources.values()])
    variables = sorted(set().union(*(v - {(2,)} for v in forms.values())))
    resource_variables = set()
    for q in resources:
        resource_variables.add(variables.index((3, q)))
    xor = lambda a, b: manager.apply("xor", a, b)
    product = lambda a, b: manager.apply("and", a, b)

    def action(word):
        output, phase, initial = monomial_signature(word, manager, [], forms=forms)
        corrected = [0] * (bits - 3) + list(phase)

        def add(value, power):
            for i in range(bits):
                if power % (1 << bits) >> i & 1:
                    carry = value
                    for j in range(i, bits):
                        corrected[j], carry = xor(corrected[j], carry), product(corrected[j], carry)

        for q, (precision, power) in resources.items():
            coefficient = power * (1 << (bits - precision))
            add(initial[q], coefficient)
            add(output[q], -coefficient)
        logical = {q: v for q, v in output.items() if q not in resources}
        pending, visited = list(logical.values()) + corrected, set()
        while pending:
            node = pending.pop()
            if node < 2 or node in visited:
                continue
            visited.add(node)
            variable, low, high = manager.nodes[node]
            if variable in resource_variables:
                raise Unsupported("Monomial action does not establish catalyst restoration")
            pending.extend((low, high))
        return logical, corrected

    try:
        left, right = action(before), action(after)
    except Incorrect as exc:
        raise Unsupported("Resource action needs a coherent verification method") from exc
    if left != right:
        raise Incorrect("Catalytic logical action or kickback phase differs")
    return {
        "method": "exact catalytic Boolean action and dyadic kickback phase",
        "nodes": len(manager.nodes),
        "resource_wires": sorted(resources),
        "resource_precision_bits": bits,
        "catalyst_restored": True,
    }


def verify_channel_equivalence(before, after, wires, columns, *, external=(), outputs=()):
    """Exact Choi equality, summing internal records instead of matching their IDs.

    Output classical bits remain observable. The bound limits arithmetic work;
    it is not a tolerance. Assertions are checked before any branch is summed.
    """
    from fractions import Fraction
    from itertools import product

    if len(wires) > 4 or max(len(before), len(after)) > 128:
        raise Unsupported("Exact channel block exceeds four wires / 128 operations")
    mapping = {q: i for i, q in enumerate(wires)}
    zero = (Fraction(0),) * 4
    work = 0

    def multiply(a, b):
        out = [Fraction(0)] * 4
        # b is conjugated: omega**j -> omega**(-j).
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                d, r = divmod(i - j, 4)
                out[r] += (-1 if d % 2 else 1) * x * y
        return tuple(out)

    def choi(gates, values):
        nonlocal work
        measured = [v for n, _, v in gates if n == "hmr"]
        if len(measured) + len(external) > 6:
            raise Unsupported("Exact channel record budget exceeded")
        result = {}
        normalized = [(n, tuple(mapping[q] for q in qs), v) for n, qs, v in gates]
        for assignment in product((0, 1), repeat=len(measured)):
            matrix, record = exact_instrument_columns(
                normalized,
                len(wires),
                columns,
                dict(zip(measured, assignment, strict=True)),
                values,
            )
            record = dict(record)
            if any(q not in record for q in outputs):
                raise Incorrect("Classical output not defined")
            label = tuple(record[q] for q in outputs)
            vector = [
                (r * len(columns) + c, x)
                for r, row in enumerate(matrix)
                for c, x in enumerate(row)
                if x != zero
            ]
            work += len(vector) ** 2
            if work > 300_000:
                raise Unsupported("Exact channel arithmetic budget exceeded")
            for i, a in vector:
                for j, b in vector:
                    key = label, i, j
                    value = tuple(
                        x + y for x, y in zip(result.get(key, zero), multiply(a, b), strict=True)
                    )
                    if value == zero:
                        result.pop(key, None)
                    else:
                        result[key] = value
        return result

    for assignment in product((0, 1), repeat=len(external)):
        values = dict(zip(external, assignment, strict=True))
        if choi(before, values) != choi(after, values):
            raise Incorrect("Exact logical channel or classical output differs")
    return {
        "method": "exact Choi equality with internal measurement records summed",
        "products": work,
        "sampling": False,
    }


class SampledBits:
    """Bit-sliced basis probes; measurement phase debts remain symbolic."""

    def __init__(self, seed, count):
        self.seed, self.count = seed, count + 1
        self.one = (1 << self.count) - 1
        self.variables = {}

    def node(self, variable, low, high):
        import hashlib

        if (low, high) != (0, self.one):
            raise ValueError("Sampled input must be a fresh Boolean variable")
        if variable not in self.variables:
            raw = hashlib.shake_256(self.seed + variable.to_bytes(8, "little")).digest(
                (self.count + 7) // 8
            )
            self.variables[variable] = (int.from_bytes(raw, "little") & self.one) & ~1
        return self.variables[variable]

    @staticmethod
    def apply(op, a, b):
        return a ^ b if op == "xor" else a & b


def sample_monomial_equivalence(before, after, inputs, *, seed, shots, batch=65536, forms=None):
    """Uniform basis-input tests with exact phases and all-record uncomputation."""
    import hashlib

    if shots < 1 or batch < 1:
        raise ValueError("A positive sample count is required")
    # Check language support before committing to a large sample campaign.
    for word in (before, after):
        try:
            monomial_signature(word, SampledBits(seed, 8), inputs, forms=forms)
        except Incorrect:
            pass  # The actual paired tests below determine the verdict.
    for start in range(0, shots, batch):
        count = min(batch, shots - start)
        manager = SampledBits(hashlib.sha256(seed + start.to_bytes(8, "little")).digest(), count)
        try:
            left, lp, _ = monomial_signature(before, manager, inputs, forms=forms)
        except Incorrect as exc:
            raise Unsupported("Reference is outside the sampled monomial domain") from exc
        right, rp, _ = monomial_signature(after, manager, inputs, forms=forms)
        if any(left.get(q, 0) != right.get(q, 0) for q in left.keys() | right.keys()) or lp != rp:
            raise Incorrect("Sampled basis action or coherent phase differs")
    return {
        "method": "bit-sliced basis probes with exact mod-8 phases",
        "sampling": True,
        "random_shots": shots,
        "input_bits": len(inputs),
        "distribution": "uniform affine input image"
        if forms is not None
        else "independent uniform input bits; zero scratch inputs",
        "measurement_check": "symbolic phase debts for every internal measurement record",
        "anchor": "zero variable assignment (including affine offset) in every batch",
    }


def _cyclotomic_product(a, b, *, conjugate=False):
    """Integer coefficients in Z[omega], omega**4=-1."""
    out = [0] * 4
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            d, r = divmod(i - j if conjugate else i + j, 4)
            out[r] += (-1 if d % 2 else 1) * x * y
    return tuple(out)


def exact_sparse_branch(gates, state, outcomes, external, *, max_states=8192, max_work=2000000):
    """Unnormalized Kraus action; integer numerators and a common power-of-two denominator.

    Extra high bits in basis indices may tag reference-system columns. Gates
    never address them. No amplitude is rounded, discarded, or normalized.
    """
    state, record, denominator, work = dict(state), dict(external), 0, 0
    zero = (0, 0, 0, 0)

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b, strict=True))

    def omega(a, power):
        out = [0] * 4
        for i, x in enumerate(a):
            d, r = divmod(i + power, 4)
            out[r] += -x if d % 2 else x
        return tuple(out)

    def hfactor(a):
        return add(omega(a, 1), omega(a, -1))  # sqrt(2); divide by 2 below.

    for name, wires, value in gates:
        work += len(state)
        if work > max_work:
            raise Unsupported("Coherent trajectory arithmetic budget exceeded")
        if name == "oxor":
            target, left, right = value
            if target in record or right not in record or (left != -1 and left not in record):
                raise Incorrect("Invalid classical dependency")
            record[target] = (0 if left == -1 else record[left]) ^ record[right]
            continue
        if isinstance(value, dict):
            if value["condition"] not in record:
                raise Incorrect("Undefined classical condition")
            if not record[value["condition"]]:
                continue
            value = None
        if name == "assert_zero":
            if any(k >> wires[0] & 1 for k in state):
                raise Incorrect("Nonzero amplitude on a released wire")
        elif name == "neg":
            state = {k: tuple(-x for x in a) for k, a in state.items()}
        elif name in {"x", "cx"}:
            target = 1 << wires[-1]
            controls = sum(1 << q for q in wires[:-1])
            state = {k ^ (target if k & controls == controls else 0): a for k, a in state.items()}
        elif name in {"z", "cz", "s", "sdg", "t", "tdg"}:
            mask = sum(1 << q for q in wires)
            power = {"z": 4, "cz": 4, "s": 2, "sdg": 6, "t": 1, "tdg": 7}[name]
            state = {k: omega(a, power) if k & mask == mask else a for k, a in state.items()}
        elif name in {"h", "hmr"}:
            target = 1 << wires[0]
            if name == "hmr":
                if value in record or value not in outcomes:
                    raise Incorrect("Invalid measurement identifier")
                record[value] = outcomes[value]
            result = {}
            for low in {k & ~target for k in state}:
                a, b = state.get(low, zero), state.get(low | target, zero)
                plus = hfactor(add(a, b))
                minus = hfactor(add(a, tuple(-x for x in b)))
                if name == "h":
                    if plus != zero:
                        result[low] = plus
                    if minus != zero:
                        result[low | target] = minus
                else:
                    amplitude = minus if record[value] else plus
                    if amplitude != zero:
                        result[low] = amplitude
            state = result
            denominator += 1
            # Cancel common powers of two without changing the Kraus scalar.
            common = 0
            for amplitude in state.values():
                for x in amplitude:
                    common |= abs(x)
            if common and denominator:
                shift = min(denominator, (common & -common).bit_length() - 1)
                if shift:
                    state = {k: tuple(x >> shift for x in a) for k, a in state.items()}
                    denominator -= shift
            if common.bit_length() > 8192:
                raise Unsupported("Coherent trajectory coefficient budget exceeded")
        else:
            raise Unsupported("Unsupported exact trajectory operation")
        if len(state) > max_states:
            raise Unsupported("Coherent trajectory state budget exceeded")
    return state, denominator, record


def sample_channel_equivalence(
    before,
    after,
    inputs,
    *,
    seed,
    rounds,
    coefficient_bits=32,
    external=(),
    outputs=(),
    columns=None,
    max_wires=12,
    max_measurements=8,
    max_work=2000000,
):
    """Random polynomial fingerprints of the COMPLETE logical channel.

    Evaluate u^dagger E(|v><w|) z with independent random complex integer
    vectors. A nonzero channel difference is a degree-four polynomial in
    their real/imaginary parts. Schwartz--Zippel gives at most 4/2**bits per
    round. Every internal measurement branch is summed with its Kraus weight.
    """
    import hashlib
    from fractions import Fraction
    from itertools import product

    if rounds < 1 or coefficient_bits < 8:
        raise ValueError("Positive rounds and at least eight random coefficient bits required")
    wires = sorted({q for word in (before, after) for _, qs, _ in word for q in qs})
    if len(wires) > max_wires:
        raise Unsupported("Coherent fingerprint exceeds active-wire budget")
    mapping = {q: i for i, q in enumerate(wires)}
    protected = [mapping[q] for q in inputs if q in mapping]
    words = [
        [(n, tuple(mapping[q] for q in qs), v) for n, qs, v in word] for word in (before, after)
    ]
    measured = [[v for n, _, v in word if n == "hmr"] for word in words]
    if max(map(len, measured)) + len(external) > max_measurements:
        raise Unsupported("Coherent fingerprint exceeds record enumeration budget")
    size, tag = 1 << len(wires), 1 << len(wires)
    if columns is None:
        columns = [
            sum((x >> j & 1) << q for j, q in enumerate(protected))
            for x in range(1 << len(protected))
        ]
    if size * (1 << (max(map(len, measured)) + len(external))) > 32768:
        raise Unsupported("Coherent fingerprint branch-state budget exceeded")
    zero = (0, 0, 0, 0)
    scalar_zero = (Fraction(0),) * 4
    assertions = sum(n == "assert_zero" for word in words for n, _, _ in word)
    degree = max(4, assertions + 4)

    for trial in range(rounds):
        rng = hashlib.shake_256(seed + trial.to_bytes(8, "little")).digest(
            (4 * size + 4 * len(columns)) * ((coefficient_bits + 7) // 8)
        )
        cursor = 0

        def vector(length, rng=rng):
            nonlocal cursor
            result = []
            width = (coefficient_bits + 7) // 8
            for _ in range(length):
                a = int.from_bytes(rng[cursor : cursor + width], "little") & (
                    (1 << coefficient_bits) - 1
                )
                cursor += width
                b = int.from_bytes(rng[cursor : cursor + width], "little") & (
                    (1 << coefficient_bits) - 1
                )
                cursor += width
                result.append((a, 0, b, 0))
            return result

        u, z, v, w = vector(size), vector(size), vector(len(columns)), vector(len(columns))
        initial = {q: a for q, a in zip(columns, v, strict=True) if a != zero}
        initial.update({q | tag: a for q, a in zip(columns, w, strict=True) if a != zero})

        def fingerprint(word, ids, values, initial=initial, u=u, z=z):
            result = {}
            for assignment in product((0, 1), repeat=len(ids)):
                state, denominator, records = exact_sparse_branch(
                    word,
                    initial,
                    dict(zip(ids, assignment, strict=True)),
                    values,
                    max_work=max_work,
                )
                if any(q not in records for q in outputs):
                    raise Incorrect("Observable classical output is undefined")
                label = tuple(records[q] for q in outputs)
                a, b = zero, zero
                for key, amplitude in state.items():
                    term = _cyclotomic_product(
                        amplitude, z[key ^ tag] if key & tag else u[key], conjugate=True
                    )
                    if key & tag:
                        b = tuple(x + y for x, y in zip(b, term, strict=True))
                    else:
                        a = tuple(x + y for x, y in zip(a, term, strict=True))
                value = _cyclotomic_product(a, b, conjugate=True)
                total = tuple(
                    x + Fraction(y, 1 << (2 * denominator))
                    for x, y in zip(result.get(label, scalar_zero), value, strict=True)
                )
                if total == scalar_zero:
                    result.pop(label, None)
                else:
                    result[label] = total
            return result

        for assignment in product((0, 1), repeat=len(external)):
            values = dict(zip(external, assignment, strict=True))
            if fingerprint(words[0], measured[0], values) != fingerprint(
                words[1], measured[1], values
            ):
                raise Incorrect("Coherent channel fingerprint differs")
    return {
        "method": "exact cyclotomic channel polynomial fingerprints",
        "sampling": True,
        "rounds": rounds,
        "coefficient_bits": coefficient_bits,
        "false_accept_bound": (degree / (1 << coefficient_bits)) ** rounds,
        "polynomial_degree_bound": degree,
        "active_wires": len(wires),
        "input_wires": len(protected),
        "measurement_branches": [1 << len(ids) for ids in measured],
        "arithmetic": "integer cyclotomic numerators and exact powers of two",
    }


class CyclotomicNetwork:
    """Exact binary tensor contraction over Z[z]/(z**D+1), D a power of two.

    Sparse coefficients are never rounded or truncated. A min-frontier
    elimination order bounds intermediate tensors rather than circuit width.
    H numerators are sqrt(2); a separate common denominator accounts for /2.
    """

    def __init__(self, bits=3, *, max_frontier=16, max_products=2000000):
        if not 3 <= bits <= 24:
            raise Unsupported("Cyclotomic resource precision exceeds 24 bits")
        self.degree = 1 << (bits - 1)
        self.max_frontier, self.max_products = max_frontier, max_products
        self.products, self.peak, self.next_axis, self.denominator = 0, 0, 0, 0
        self.factors = []

    def root(self, exponent):
        quotient, power = divmod(exponent, self.degree)
        return ((power, -1 if quotient % 2 else 1),)

    def add(self, a, b):
        result = dict(a)
        for p, c in b:
            result[p] = result.get(p, 0) + c
        return tuple(sorted((p, c) for p, c in result.items() if c))

    def mul(self, a, b):
        self.products += len(a) * len(b)
        if self.products > self.max_products:
            raise Unsupported("Exact tensor coefficient-product budget exceeded")
        result = {}
        for p, c in a:
            for q, d in b:
                quotient, power = divmod(p + q, self.degree)
                result[power] = result.get(power, 0) + (-c * d if quotient % 2 else c * d)
        if len(result) > 32768 or any(abs(c).bit_length() > 65536 for c in result.values()):
            raise Unsupported("Exact tensor coefficient-size budget exceeded")
        return tuple(sorted((p, c) for p, c in result.items() if c))

    def axis(self):
        result = self.next_axis
        self.next_axis += 1
        if self.next_axis > 200000:
            raise Unsupported("Tensor graph size budget exceeded")
        return result

    def factor(self, axes, table):
        if len(set(axes)) != len(axes):
            raise ValueError("Repeated tensor axis")
        self.factors.append((tuple(axes), {k: v for k, v in table.items() if v}))

    def boolean(self, axes, predicate):
        self.factor(axes, {x: ((0, 1),) for x in range(1 << len(axes)) if predicate(x)})

    def join(self, left, right):
        la, lt = left
        ra, rt = right
        axes = la + tuple(q for q in ra if q not in la)
        if len(axes) > self.max_frontier:
            raise Unsupported("Exact tensor contraction frontier budget exceeded")
        self.peak = max(self.peak, len(axes))
        shared = [(i, la.index(q)) for i, q in enumerate(ra) if q in la]
        extra = [(i, axes.index(q)) for i, q in enumerate(ra) if q not in la]
        buckets = {}
        for key, value in rt.items():
            index = sum(((key >> i) & 1) << j for i, j in shared)
            new = sum(((key >> i) & 1) << j for i, j in extra)
            buckets.setdefault(index, []).append((new, value))
        mask = sum(1 << j for _, j in shared)
        table = {}
        for key, value in lt.items():
            for new, other in buckets.get(key & mask, ()):
                product = self.mul(value, other)
                if product:
                    table[key | new] = product
        return axes, table

    def contract(self):
        import heapq
        from fractions import Fraction

        factors = dict(enumerate(self.factors))
        incidence = {}
        for key, (axes, _) in factors.items():
            for q in axes:
                incidence.setdefault(q, set()).add(key)
        heap, versions = [], {}

        def schedule(q):
            if q in incidence:
                neighbors = set().union(*(set(factors[k][0]) for k in incidence[q]))
                version = versions.get(q, 0) + 1
                versions[q] = version
                heapq.heappush(heap, (len(neighbors), len(incidence[q]), q, version))

        for q in incidence:
            schedule(q)
        serial = len(factors)
        while heap:
            width, _, q, version = heapq.heappop(heap)
            if q not in incidence or versions[q] != version:
                continue
            if width > self.max_frontier:
                raise Unsupported("Exact tensor contraction frontier budget exceeded")
            keys = incidence[q].copy()
            bucket = sorted((factors[k] for k in keys), key=lambda f: len(f[1]))
            joined = bucket[0]
            for item in bucket[1:]:
                joined = self.join(joined, item)
            axes, table = joined
            position = axes.index(q)
            summed = {}
            for key, value in table.items():
                reduced = (key & ((1 << position) - 1)) | ((key >> (position + 1)) << position)
                summed[reduced] = self.add(summed.get(reduced, ()), value)
            neighbors = set(axes) - {q}
            for key in keys:
                for v in factors[key][0]:
                    incidence[v].discard(key)
                del factors[key]
            del incidence[q]
            axes = tuple(v for v in axes if v != q)
            factors[serial] = axes, {k: v for k, v in summed.items() if v}
            for v in axes:
                incidence[v].add(serial)
            serial += 1
            for v in neighbors:
                schedule(v)
        result = ((0, 1),)
        for axes, table in factors.values():
            if axes:
                raise ValueError("Uncontracted tensor boundary")
            result = self.mul(result, table.get(0, ()))
        return {p: Fraction(c, 1 << self.denominator) for p, c in result}

    def program(
        self, gates, forms, resources, variables, *, conjugate=False, records=None, external=()
    ):
        """Build an isometry with all quantum outputs retained, including catalysts."""
        wires = {}
        one = ((0, 1),)
        sign = -1 if conjugate else 1
        for q, terms in forms.items():
            wire = self.axis()
            wires[q] = wire
            if q in resources:
                bits, exponent = resources[q]
                power = exponent * (2 * self.degree // (1 << bits)) * sign
                self.factor([wire], {0: one, 1: self.root(power)})
                continue
            parts = sorted(terms - {(2,)})
            if not parts:
                self.factor([wire], {int((2,) in terms): one})
                continue
            current = variables[parts[0]]
            for part in parts[1:]:
                target = self.axis()
                self.boolean(
                    [current, variables[part], target],
                    lambda x: (x & 1) ^ (x >> 1 & 1) == (x >> 2 & 1),
                )
                current = target
            offset = int((2,) in terms)
            self.boolean([current, wire], lambda x, offset=offset: (x & 1) ^ offset == x >> 1)
        h = self.add(self.root(self.degree // 4), self.root(-self.degree // 4))
        available = set(external)
        for name, qs, value in gates:
            if records is None and (value is not None or name in {"hmr", "oxor"}):
                raise Unsupported("Tensor isometry backend requires a coherent gate region")
            condition = value.get("condition") if isinstance(value, dict) else None
            if condition is not None and condition not in available:
                raise Incorrect("Undefined tensor feedback record")
            if name == "oxor":
                target, left, right = value
                if (
                    target in available
                    or right not in available
                    or (left != -1 and left not in available)
                ):
                    raise Incorrect("Invalid tensor classical XOR")
                records.setdefault(target, self.axis())
                axes = list(
                    dict.fromkeys(
                        [records[target], records[right]] + ([] if left == -1 else [records[left]])
                    )
                )
                a, b = axes.index(records[target]), axes.index(records[right])
                c = None if left == -1 else axes.index(records[left])
                self.boolean(
                    axes,
                    lambda x, a=a, b=b, c=c: (
                        (x >> a & 1) == ((x >> b & 1) ^ (0 if c is None else x >> c & 1))
                    ),
                )
                available.add(target)
                continue
            if name == "hmr":
                if value in available:
                    raise Incorrect("Repeated tensor measurement record")
                records.setdefault(value, self.axis())
                old, new = wires[qs[0]], self.axis()
                wires[qs[0]] = new
                self.factor(
                    [old, new, records[value]],
                    {
                        x: self.mul(h, ((0, -1 if x & 1 and x >> 2 & 1 else 1),))
                        for x in range(8)
                        if not x >> 1 & 1
                    },
                )
                self.denominator += 1
                available.add(value)
                continue
            if name in {"z", "cz", "s", "sdg", "t", "tdg", "neg"}:
                power = {"z": 4, "cz": 4, "s": 2, "sdg": -2, "t": 1, "tdg": -1, "neg": 4}[name]
                axes = [wires[q] for q in qs]
                if condition is not None:
                    axes.append(records[condition])
                self.factor(
                    axes,
                    {
                        x: self.root(sign * power * self.degree // 4)
                        if x == (1 << len(axes)) - 1
                        else one
                        for x in range(1 << len(axes))
                    },
                )
            elif name == "assert_zero":
                self.factor([wires[qs[0]]], {0: one})
            elif name in {"h", "x", "cx"}:
                old, new = wires[qs[-1]], self.axis()
                wires[qs[-1]] = new
                if name == "h":
                    if condition is not None:
                        self.factor(
                            [old, new, records[condition]],
                            {
                                x: self.mul(h, ((0, -1 if x & 3 == 3 else 1),))
                                if x >> 2
                                else ((0, 2),)
                                for x in range(8)
                                if x >> 2 or (x & 1) == (x >> 1 & 1)
                            },
                        )
                        self.denominator += 1
                        continue
                    self.factor(
                        [old, new], {x: self.mul(h, ((0, -1 if x == 3 else 1),)) for x in range(4)}
                    )
                    self.denominator += 1
                elif name == "x":
                    if condition is None:
                        self.boolean([old, new], lambda x: (x & 1) != x >> 1)
                    else:
                        self.boolean(
                            [old, new, records[condition]],
                            lambda x: (x & 1) ^ (x >> 2) == (x >> 1 & 1),
                        )
                else:
                    axes = [wires[qs[0]], old, new]
                    if condition is not None:
                        axes.append(records[condition])
                    self.boolean(
                        axes,
                        lambda x, controlled=condition is not None: (
                            ((x & 1) & ((x >> 3) if controlled else 1)) ^ (x >> 1 & 1)
                            == (x >> 2 & 1)
                        ),
                    )
            else:
                raise Unsupported("Unsupported tensor isometry gate")
        return wires


def tensor_isometry_check(
    before, after, forms, resources=None, *, seed=None, rounds=1, coefficient_bits=32
):
    """Complete isometry equality, exact or product-polynomial fingerprints.

    Exact mode contracts the squared Hilbert--Schmidt norm of U-V. Sampled
    mode evaluates <product| (U-V) |product> over independent complex integer
    coefficients; product vectors span the FULL operator space. This is not
    computational-basis sampling. Resource amplitudes remain exact algebraic
    constants. Trace loss from assertions is separately tested, never discarded.
    """
    import hashlib

    resources = {q: v for q, v in (resources or {}).items() if q in forms}
    words = (before, after)
    if any(v is not None or n in {"hmr", "oxor"} for word in words for n, _, v in word):
        raise Unsupported("Tensor isometry backend requires a coherent gate region")
    bits = max([3] + [b for b, _ in resources.values()])
    terms = sorted(set().union(*(v - {(2,)} for q, v in forms.items() if q not in resources)))
    stats = {"peak_frontier": 0, "coefficient_products": 0}

    def network():
        return CyclotomicNetwork(bits)

    def finish(net):
        value = net.contract()
        stats["peak_frontier"] = max(stats["peak_frontier"], net.peak)
        stats["coefficient_products"] += net.products
        return value

    def variables(net):
        return {v: net.axis() for v in terms}

    def inner(left, right, weights=None):
        net = network()
        if weights is None:
            lv = rv = variables(net)
            for axis in lv.values():
                net.factor([axis], {0: ((0, 1),), 1: ((0, 1),)})
        else:
            lv, rv = variables(net), variables(net)
            for v in terms:
                a, b = weights[v]
                net.factor([lv[v]], {0: ((0, 1),), 1: net.add(((0, a),), ((net.degree // 2, b),))})
                net.factor([rv[v]], {0: ((0, 1),), 1: net.add(((0, a),), ((net.degree // 2, -b),))})
        lo = net.program(left, forms, resources, lv)
        ro = net.program(right, forms, resources, rv, conjugate=True)
        for q in forms:
            net.boolean([lo[q], ro[q]], lambda x: (x & 1) == x >> 1)
        return finish(net)

    if seed is None:
        norms = [inner(word, word) for word in words]
        expected = {0: 1 << (len(terms) + len(resources))}
        if any(norm != expected for norm in norms):
            raise Incorrect("A zero assertion discards nonzero amplitude")
        cross = inner(before, after)
        # Re(<V,U>) = its constant coefficient for this power-of-two field.
        # Comparing the entire conjugate sum avoids relying on that shortcut.
        ring = network()
        real = ring.add(
            tuple(cross.items()), tuple((p, c) for p, c in inner(after, before).items())
        )
        if dict(real) != {0: 2 * expected[0]}:
            raise Incorrect("Exact tensor isometry differs (including phase or borrowed state)")
    else:
        if rounds < 1 or coefficient_bits < 8:
            raise ValueError("Invalid tensor fingerprint randomization")
        for trial in range(rounds):
            stream = hashlib.shake_256(seed + trial.to_bytes(8, "little")).digest(
                (len(terms) + len(forms)) * 2 * ((coefficient_bits + 7) // 8)
            )
            cursor = 0

            def draw(stream=stream):
                nonlocal cursor
                width = (coefficient_bits + 7) // 8
                value = int.from_bytes(stream[cursor : cursor + width], "little") & (
                    (1 << coefficient_bits) - 1
                )
                cursor += width
                return value

            ins = {v: (draw(), draw()) for v in terms}
            outs = {q: (draw(), draw()) for q in forms}

            def amplitude(word, ins=ins, outs=outs):
                net = network()
                variables_ = variables(net)
                for v, axis in variables_.items():
                    a, b = ins[v]
                    net.factor(
                        [axis], {0: ((0, 1),), 1: net.add(((0, a),), ((net.degree // 2, b),))}
                    )
                final = net.program(word, forms, resources, variables_)
                for q, axis in final.items():
                    a, b = outs[q]
                    net.factor(
                        [axis], {0: ((0, 1),), 1: net.add(((0, a),), ((net.degree // 2, b),))}
                    )
                return finish(net)

            for word in words:
                if any(n == "assert_zero" for n, _, _ in word):
                    plain = [g for g in word if g[0] != "assert_zero"]
                    if inner(word, word, ins) != inner(plain, plain, ins):
                        raise Incorrect("Tensor fingerprint detects a false zero assertion")
            if amplitude(before) != amplitude(after):
                raise Incorrect(
                    "Tensor isometry fingerprint differs (including phase or borrowed state)"
                )
    degree = max(1, len(terms) + len(forms), 2 * len(terms))
    return {
        "method": "exact cyclotomic tensor isometry"
        if seed is None
        else "exact cyclotomic tensor product fingerprints",
        "sampling": seed is not None,
        "active_wires": len(forms),
        "resource_wires": sorted(resources),
        "resource_precision_bits": bits,
        "arithmetic": "exact sparse cyclotomic integers and powers of two; no truncation",
        **stats,
        **(
            {
                "rounds": rounds,
                "coefficient_bits": coefficient_bits,
                "polynomial_degree_bound": degree,
                "false_accept_bound": 3 * (degree / (1 << coefficient_bits)) ** rounds,
            }
            if seed is not None
            else {}
        ),
    }


def tensor_channel_fingerprint(
    before, after, forms, resources, *, seed, rounds, coefficient_bits=32, external=(), outputs=()
):
    """Full CP instrument fingerprint, summing records inside the tensor graph.

    Ket and bra share classical records, never amplitudes. Distinct product
    variables at four quantum boundaries test the entire Choi tensor; output
    records and external classical inputs get additional independent weights.
    No measurement branch is sampled, normalized, or omitted.
    """
    import hashlib

    if rounds < 1 or coefficient_bits < 8:
        raise ValueError("Invalid tensor channel randomization")
    resources = {q: v for q, v in (resources or {}).items() if q in forms}
    terms = sorted(set().union(*(v - {(2,)} for q, v in forms.items() if q not in resources)))
    bits = max([3] + [b for b, _ in resources.values()])
    stats = {"peak_frontier": 0, "coefficient_products": 0}
    degree = max(1, 2 * (len(terms) + len(forms)) + len(external) + len(outputs))
    width = (coefficient_bits + 7) // 8
    for trial in range(rounds):
        stream = hashlib.shake_256(seed + trial.to_bytes(8, "little")).digest(
            (4 * (len(terms) + len(forms)) + len(external) + len(outputs)) * width
        )
        cursor = 0

        def draw(stream=stream):
            nonlocal cursor
            value = int.from_bytes(stream[cursor : cursor + width], "little") & (
                (1 << coefficient_bits) - 1
            )
            cursor += width
            return value

        inputs = [{v: (draw(), draw()) for v in terms} for _ in range(2)]
        boundaries = [{q: (draw(), draw()) for q in forms} for _ in range(2)]
        classical = [{v: draw() for v in keys} for keys in (external, outputs)]

        def evaluate(
            word, *, trace=False, inputs=inputs, boundaries=boundaries, classical=classical
        ):
            net = CyclotomicNetwork(bits)
            records = {v: net.axis() for v in external}
            for v in external:
                net.factor([records[v]], {0: ((0, 1),), 1: ((0, classical[0][v]),)})
            ends = []
            for side in range(2):
                variables = {v: net.axis() for v in terms}
                for v, axis in variables.items():
                    a, b = inputs[0 if trace else side][v]
                    if trace and side:
                        b = -b
                    net.factor(
                        [axis], {0: ((0, 1),), 1: net.add(((0, a),), ((net.degree // 2, b),))}
                    )
                final = net.program(
                    word,
                    forms,
                    resources,
                    variables,
                    conjugate=bool(side),
                    records=records,
                    external=external,
                )
                ends.append(final)
                if not trace:
                    for q, axis in final.items():
                        a, b = boundaries[side][q]
                        net.factor(
                            [axis], {0: ((0, 1),), 1: net.add(((0, a),), ((net.degree // 2, b),))}
                        )
            if trace:
                for q in forms:
                    net.boolean([ends[0][q], ends[1][q]], lambda x: (x & 1) == x >> 1)
            else:
                for v in outputs:
                    if v not in records:
                        raise Incorrect("Observable tensor classical output is undefined")
                    net.factor([records[v]], {0: ((0, 1),), 1: ((0, classical[1][v]),)})
            value = net.contract()
            stats["peak_frontier"] = max(stats["peak_frontier"], net.peak)
            stats["coefficient_products"] += net.products
            return value

        for word in (before, after):
            if any(n == "assert_zero" for n, _, _ in word) and evaluate(
                word, trace=True
            ) != evaluate([g for g in word if g[0] != "assert_zero"], trace=True):
                raise Incorrect("Tensor channel fingerprint detects a false zero assertion")
        if evaluate(before) != evaluate(after):
            raise Incorrect(
                "Tensor channel fingerprint differs (including resource or classical output)"
            )
    return {
        "method": "exact cyclotomic tensor channel fingerprints",
        "sampling": True,
        "active_wires": len(forms),
        "resource_wires": sorted(resources),
        "resource_precision_bits": bits,
        "rounds": rounds,
        "coefficient_bits": coefficient_bits,
        "polynomial_degree_bound": degree,
        "false_accept_bound": 3 * (degree / (1 << coefficient_bits)) ** rounds,
        "measurement_check": "all branches contracted with exact Kraus weights",
        "arithmetic": "exact sparse cyclotomic integers and powers of two; no truncation",
        **stats,
    }


class SymbolicSimulator:
    """Exact gate propagation for a finite address domain and symbolic target word.

    This engine has no expected lookup table or task-file dependency. It supports
    the discrete upstream gate set, not arbitrary complex-amplitude rotations.
    """

    max_terms = 4096
    max_products = 100_000

    def __init__(self, address_bits, target_bits):
        if type(address_bits) is not int or not 1 <= address_bits <= 12:
            raise ValueError("Unsupported address width")
        if type(target_bits) is not int or not 1 <= target_bits <= 4096:
            raise ValueError("Unsupported target width")
        self.address_bits = address_bits
        self.target_bits = target_bits
        self.addresses = 1 << address_bits
        self.mask = (1 << self.addresses) - 1
        self.one = {0: self.mask}
        self.wires = {}
        self.phase = {}
        self.debts = {}
        self.outcomes = {}
        self.draws = 0
        self.registers = None

    @staticmethod
    def unsupported(message):
        raise Unsupported(message)

    def xor(self, a, b):
        out = a.copy()
        for monomial, coeff in b.items():
            value = out.get(monomial, 0) ^ coeff
            if value:
                out[monomial] = value
            else:
                out.pop(monomial, None)
        if len(out) > self.max_terms:
            raise Unsupported("ANF term limit exceeded; no exact verdict")
        return out

    def mul(self, a, b):
        if len(a) * len(b) > self.max_products:
            raise Unsupported("ANF product limit exceeded; no exact verdict")
        out = {}
        for ma, ca in a.items():
            for mb, cb in b.items():
                m = ma | mb  # Boolean variables satisfy x*x=x.
                value = out.get(m, 0) ^ (ca & cb)
                if value:
                    out[m] = value
                else:
                    out.pop(m, None)
        if len(out) > self.max_terms:
            raise Unsupported("ANF term limit exceeded; no exact verdict")
        return out

    def initialize(self, registers):
        if self.registers is not None:
            return
        widths = [self.address_bits, self.target_bits]
        if len(registers) != 2 or any(
            kind != 0 or len(ids) != width for (kind, ids), width in zip(registers, widths)
        ):
            raise Incorrect("Expected exactly two quantum registers: address, target")
        ids = [q for _, reg in registers for q in reg]
        if len(ids) != len(set(ids)):
            raise Incorrect("Register wires overlap")
        self.registers = registers
        self.protected = set(ids)
        for bit, q in enumerate(registers[0][1]):
            coeff = sum(1 << address for address in range(self.addresses) if address >> bit & 1)
            self.wires[q] = {0: coeff} if coeff else {}
        for bit, q in enumerate(registers[1][1]):
            self.wires[q] = {1 << bit: self.mask}
        self.initial = {q: p.copy() for q, p in self.wires.items()}

    def value(self, q):
        return self.wires.get(q, {})

    def add_phase(self, poly, outcome):
        if outcome is None:
            self.phase = self.xor(self.phase, poly)
        else:
            if outcome not in self.outcomes:
                raise Incorrect("Phase control uses an undefined measurement outcome")
            for draw in self.outcomes.get(outcome, set()):
                debt = self.xor(self.debts.get(draw, {}), poly)
                if debt:
                    self.debts[draw] = debt
                else:
                    self.debts.pop(draw, None)

    def apply(self, name, qa, qb, qt, ot, oc):
        if oc is not None and name not in ("NEG", "Z", "CZ", "CCZ", "OXOR"):
            raise Unsupported("Only phase corrections may be classically controlled")
        if name in ("X", "CX", "CCX"):
            value = self.one if name == "X" else self.value(qb)
            if name == "CCX":
                value = self.mul(self.value(qa), value)
            self.wires[qt] = self.xor(self.value(qt), value)
        elif name in ("NEG", "Z", "CZ", "CCZ"):
            value = self.one if name == "NEG" else self.value(qt)
            if name in ("CZ", "CCZ"):
                value = self.mul(self.value(qb), value)
            if name == "CCZ":
                value = self.mul(self.value(qa), value)
            self.add_phase(value, oc)
        elif name == "SWAP":
            self.wires[qb], self.wires[qt] = self.value(qt), self.value(qb)
        elif name == "HMR":
            if qt in self.protected:
                raise Unsupported("Measuring logical input/output wires is not supported")
            draw = self.draws
            self.draws += 1
            if self.value(qt):
                self.debts[draw] = self.value(qt).copy()
            self.outcomes[ot] = {draw}
            self.wires[qt] = {}
        elif name == "OXOR":
            self.outcomes[ot] = self.outcomes.get(ot, set()) ^ self.outcomes.get(oc, set())
        elif name == "R":
            if self.value(qt):
                raise Incorrect(f"Reset of nonzero wire {qt}")
        else:
            raise Unsupported(f"Operation {name} has no exact lookup semantics yet")

    def assert_clean(self):
        """Require zero workspace and no residual logical or measurement phase."""
        if any(value for q, value in self.wires.items() if q not in self.protected):
            raise Incorrect("Workspace not clean for all logical inputs")
        if self.phase:
            raise Incorrect("Nonzero logical phase polynomial")
        if any(self.debts.values()):
            raise Incorrect("Uncancelled phase for at least one measurement draw")

    def apply_elementary(self, gates):
        """Exact all-outcome propagation of the dynamic classical sublanguage."""
        for name, wires, value in gates:
            if name == "oxor":
                target, left, right = value
                if (
                    target in self.outcomes
                    or (left != -1 and left not in self.outcomes)
                    or right not in self.outcomes
                ):
                    raise Incorrect("Invalid outcome dependency")
                self.outcomes[target] = (
                    set() if left == -1 else self.outcomes[left]
                ) ^ self.outcomes[right]
                continue
            condition = value["condition"] if isinstance(value, dict) else None
            if condition is not None and condition not in self.outcomes:
                raise Incorrect("Undefined classical condition")
            if name == "assert_zero":
                if self.value(wires[0]):
                    raise Incorrect("Released wire is not zero")
                continue
            if name not in ("x", "cx", "ccx", "z", "cz", "hmr"):
                raise Unsupported("Nonclassical gate in symbolic component proof")
            qa = wires[0] if len(wires) == 3 else 0
            qb = wires[-2] if len(wires) >= 2 else 0
            self.apply(name.upper(), qa, qb, wires[-1], value if name == "hmr" else 0, condition)


class GaussianSimulator:
    """Number-preserving mode rotations with vacuum phase fixed to +1.

    One run fixes address/angle and enable values. Coherent controls are verified
    by the evaluator's exact composition contract, not by treating them as free
    classical controls. This simulator supplies a numerical cross-check.
    """

    def __init__(self, modes):
        if type(modes) is not int or not 1 <= modes <= 64:
            raise ValueError("Gaussian mode limit exceeded")
        self.modes = modes
        self.matrix = np.eye(modes)

    def givens(self, p, q, turns, angle_bits):
        if not 0 <= p < self.modes or not 0 <= q < self.modes or p == q:
            raise ValueError("Invalid mode pair")
        theta = math.tau * (turns % (1 << angle_bits)) / (1 << angle_bits)
        c, s = math.cos(theta), math.sin(theta)
        a, b = self.matrix[p].copy(), self.matrix[q].copy()
        self.matrix[p], self.matrix[q] = c * a + s * b, -s * a + c * b

    def run(self, gates, word, angle_bits, *, enable=True):
        for gate in gates:
            if word >> gate["angle_bit"] & 1 and (enable or not gate["controlled"]):
                self.givens(gate["p"], gate["q"], gate["turns"], angle_bits)
        return self.matrix


class DenseFockSimulator:
    """Small full Fock-space unitary, applying occupation-basis gates directly.

    Mode p is occupation bit p. Nonadjacent Givens include the parity between
    modes. Vacuum and double occupancy are fixed exactly, including their phase.
    """

    def __init__(self, modes):
        if type(modes) is not int or not 1 <= modes <= 8:
            raise ValueError("Dense Fock simulation is restricted to at most 8 modes")
        self.modes = modes
        self.matrix = np.eye(1 << modes)

    def givens(self, p, q, turns, angle_bits):
        if not 0 <= p < self.modes or not 0 <= q < self.modes or p == q:
            raise ValueError("Invalid mode pair")
        if p > q:
            p, q, turns = q, p, -turns
        theta = math.tau * (turns % (1 << angle_bits)) / (1 << angle_bits)
        c, s = math.cos(theta), math.sin(theta)
        middle = ((1 << q) - 1) ^ ((1 << (p + 1)) - 1)
        for state in range(1 << self.modes):
            if state >> p & 1 and not state >> q & 1:
                other = state ^ (1 << p) ^ (1 << q)
                sign = -1 if (state & middle).bit_count() % 2 else 1
                a, b = self.matrix[state].copy(), self.matrix[other].copy()
                self.matrix[state] = c * a + sign * s * b
                self.matrix[other] = -sign * s * a + c * b

    def run(self, gates, word, angle_bits, *, enable=True):
        for gate in gates:
            if word >> gate["angle_bit"] & 1 and (enable or not gate["controlled"]):
                self.givens(gate["p"], gate["q"], gate["turns"], angle_bits)
        return self.matrix


def fock_lift(mode_matrix):
    """Independent exterior-power reference using determinant minors; <=8 modes."""
    mode_matrix = np.asarray(mode_matrix)
    n = mode_matrix.shape[0]
    if mode_matrix.shape != (n, n) or not 1 <= n <= 8:
        raise ValueError("Fock lift requires a square matrix of at most 8 modes")
    states = [[p for p in range(n) if word >> p & 1] for word in range(1 << n)]
    full = np.zeros((1 << n, 1 << n), dtype=mode_matrix.dtype)
    for row, occupied_out in enumerate(states):
        for col, occupied_in in enumerate(states):
            if len(occupied_out) == len(occupied_in):
                full[row, col] = (
                    np.linalg.det(mode_matrix[np.ix_(occupied_out, occupied_in)])
                    if occupied_in
                    else 1
                )
    return full


def jordan_wigner(modes, mode, pauli):
    """Small full occupation-basis I, Z, or Majorana X/Y matrix."""
    if not 0 <= mode < modes <= 8 or pauli not in ("I", "X", "Y", "Z"):
        raise ValueError("Invalid small Jordan-Wigner operator")
    matrix = np.zeros((1 << modes, 1 << modes), dtype=complex)
    for state in range(1 << modes):
        occupied = state >> mode & 1
        if pauli in ("I", "Z"):
            matrix[state, state] = 1 if pauli == "I" else (-1) ** occupied
        else:
            phase = (-1) ** ((state & ((1 << mode) - 1)).bit_count() % 2)
            if pauli == "Y":
                phase *= -1j if occupied else 1j
            matrix[state ^ (1 << mode), state] = phase
    return matrix


def exact_clifford_t_columns(gates, num_qubits, columns):
    """Phase-sensitive exact unitary columns, limited to four wires / 128 gates."""
    return _exact_columns(gates, num_qubits, columns)[0]


def exact_instrument_columns(gates, num_qubits, columns, outcomes, external):
    """Unnormalized Kraus columns and classical outputs for a specified record.

    The caller enumerates ALL measurement and external-classical assignments.
    No branch is normalized or phase-aligned, including zero-probability branches.
    """
    return _exact_columns(gates, num_qubits, columns, outcomes=outcomes, external=external)


def _exact_columns(gates, num_qubits, columns, *, outcomes=None, external=None):
    """Exact local Clifford+T columns in Q[omega]/(omega**4+1).

    Every amplitude is a tuple of four Fractions in the basis 1,omega,omega^2,
    omega^3. H=(omega-omega^3)/2. This deliberately bounded primitive supports
    at most four wires and 128 unconditional gates, and never aligns phases.
    """
    from fractions import Fraction
    from itertools import islice

    if type(num_qubits) is not int or not 1 <= num_qubits <= 4:
        raise Unsupported("Exact local matrix supports one to four wires")
    size = 1 << num_qubits
    columns = tuple(islice(iter(columns), size + 1))
    if (
        not 1 <= len(columns) <= size
        or len(set(columns)) != len(columns)
        or any(type(c) is not int or not 0 <= c < size for c in columns)
    ):
        raise Incorrect("Invalid exact-matrix input columns")
    gates = list(islice(iter(gates), 129))
    if len(gates) > 128:
        raise Unsupported("Exact local matrix gate limit exceeded")
    zero = (Fraction(0),) * 4
    one = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))
    state = [[one if row == col else zero for col in columns] for row in range(size)]

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b, strict=True))

    def negate(a):
        return tuple(-x for x in a)

    def omega(a, power):
        result = [Fraction(0)] * 4
        for i, value in enumerate(a):
            quotient, remainder = divmod(i + power, 4)
            result[remainder] += (-1 if quotient % 2 else 1) * value
        return tuple(result)

    def hadamard_factor(a):
        return tuple(x / 2 for x in add(omega(a, 1), negate(omega(a, 3))))

    powers = {"t": 1, "tdg": -1, "s": 2, "sdg": -2, "z": 4}
    record = dict(external or {})
    for name, wires, value in gates:
        dynamic = outcomes is not None
        allowed = {*powers, "h", "x", "cx", "cz"}
        if dynamic:
            allowed |= {"hmr", "oxor", "neg", "assert_zero"}
        if name not in allowed or (not dynamic and value is not None):
            raise Unsupported("Unsupported operation in exact matrix proof")
        arity = 0 if name in ("oxor", "neg") else 2 if name in ("cx", "cz") else 1
        if (
            len(wires) != arity
            or len(set(wires)) != arity
            or any(type(q) is not int or not 0 <= q < num_qubits for q in wires)
        ):
            raise Incorrect("Invalid exact-matrix gate wires")
        if name == "oxor":
            target, left, right = value
            if target in record or right not in record or (left != -1 and left not in record):
                raise Incorrect("Invalid classical dependency in exact instrument")
            record[target] = (0 if left == -1 else record[left]) ^ record[right]
            continue
        if isinstance(value, dict):
            if set(value) != {"condition"} or value["condition"] not in record:
                raise Incorrect("Undefined classical condition in exact instrument")
            if not record[value["condition"]]:
                continue
            value = None
        if name == "hmr":
            if value in record or value not in outcomes:
                raise Incorrect("Invalid measurement identity in exact instrument")
            record[value] = outcomes[value]
            q = wires[0]
            for low in range(size):
                if low >> q & 1:
                    continue
                high = low | (1 << q)
                a, b = state[low], state[high]
                state[low] = [
                    hadamard_factor(add(x, negate(y) if record[value] else y))
                    for x, y in zip(a, b, strict=True)
                ]
                state[high] = [zero] * len(columns)
        elif name == "assert_zero":
            if any(x != zero for i, row in enumerate(state) if i >> wires[0] & 1 for x in row):
                raise Incorrect("Unproved zero assertion in exact instrument")
        elif name == "neg":
            state = [[negate(x) for x in row] for row in state]
        elif value is not None:
            raise Incorrect("Unexpected exact-matrix gate value")
        elif name == "h":
            q = wires[0]
            for low in range(size):
                if low >> q & 1:
                    continue
                high = low | (1 << q)
                a, b = state[low], state[high]
                state[low] = [hadamard_factor(add(x, y)) for x, y in zip(a, b, strict=True)]
                state[high] = [
                    hadamard_factor(add(x, negate(y))) for x, y in zip(a, b, strict=True)
                ]
        elif name in ("x", "cx"):
            target = wires[-1]
            state = [
                state[row ^ (1 << target)] if name == "x" or row >> wires[0] & 1 else state[row]
                for row in range(size)
            ]
        elif name == "cz":
            state = [
                [negate(x) for x in values] if all(row >> q & 1 for q in wires) else values
                for row, values in enumerate(state)
            ]
        else:
            state = [
                [omega(x, powers[name]) for x in values] if row >> wires[0] & 1 else values
                for row, values in enumerate(state)
            ]
    return tuple(tuple(row) for row in state), tuple(sorted(record.items()))


def verify_zx_equivalence(before, after):
    """Use pinned PyZX rewriting, then check identity wiring and its EXACT scalar.

    No QASM import, simulation verdict, tensor tolerance, or global-phase quotient
    is used. A failed reduction is inconclusive, not a claim of non-equivalence.
    """
    from fractions import Fraction

    import pyzx as zx

    if zx.__version__ != "0.10.7":
        raise Unsupported("ZX proof requires pinned PyZX 0.10.7")
    if max(len(before), len(after)) > 4096:
        raise Unsupported("ZX block exceeds 4096 gates")
    wires = sorted({q for _, qs, _ in before + after for q in qs})
    if not 1 <= len(wires) <= 32:
        raise Unsupported("ZX block requires one to 32 wires")
    mapping = {q: i for i, q in enumerate(wires)}
    phases = {
        "t": Fraction(1, 4),
        "tdg": Fraction(-1, 4),
        "s": Fraction(1, 2),
        "sdg": Fraction(-1, 2),
        "z": Fraction(1),
    }

    def circuit(gates):
        result = zx.Circuit(len(wires))
        for name, qs, value in gates:
            if value is not None or name not in {*phases, "h", "x", "cx", "cz"}:
                raise Unsupported("ZX blocks require unconditional Clifford+T gates")
            if len(qs) != (2 if name in ("cx", "cz") else 1) or len(set(qs)) != len(qs):
                raise Incorrect("Invalid ZX gate wires")
            if name in phases:
                result.add_gate("ZPhase", mapping[qs[0]], phase=phases[name])
            else:
                result.add_gate(
                    {"h": "HAD", "x": "NOT", "cx": "CNOT", "cz": "CZ"}[name],
                    *(mapping[q] for q in qs),
                )
        return result

    miter = circuit(before).adjoint()
    miter.add_circuit(circuit(after))
    graph = miter.to_graph(backend="simple")
    zx.simplify.full_reduce(graph)
    inputs, outputs = tuple(graph.inputs()), tuple(graph.outputs())
    if (
        graph.num_vertices() != 2 * len(wires)
        or graph.num_edges() != len(wires)
        or len(inputs) != len(wires)
        or len(outputs) != len(wires)
        or set(inputs) & set(outputs)
        or any(graph.type(v) != zx.VertexType.BOUNDARY for v in graph.vertices())
        or any(
            not graph.connected(a, b)
            or graph.vertex_degree(a) != 1
            or graph.vertex_degree(b) != 1
            or graph.edge_type(graph.edge(a, b)) != zx.EdgeType.SIMPLE
            for a, b in zip(inputs, outputs, strict=True)
        )
    ):
        raise Unsupported("ZX reduction did not prove identity with the fixed wire order")
    scalar = graph.scalar
    if scalar.is_unknown or scalar.is_zero or scalar.floatfactor != 1:
        raise Unsupported("ZX scalar is unknown, zero, or contains a floating factor")
    if type(scalar.power2) is not int or abs(scalar.power2) > 32768:
        raise Unsupported("ZX scalar exponent exceeds exact proof budget")
    zero, one = (Fraction(0),) * 4, (Fraction(1), Fraction(0), Fraction(0), Fraction(0))

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b, strict=True))

    def mul(a, b):
        result = list(zero)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                result[(i + j) % 4] += (-1 if i + j >= 4 else 1) * x * y
        return tuple(result)

    def phase(p):
        if not isinstance(p, (int, Fraction)) or (4 * p).denominator != 1:
            raise Unsupported("ZX scalar left the exact Clifford+T number field")
        power = int(4 * p) % 8
        result = list(zero)
        result[power % 4] = Fraction(-1 if power >= 4 else 1)
        return tuple(result)

    value = tuple(x * Fraction(2) ** (scalar.power2 // 2) for x in phase(scalar.phase))
    if scalar.power2 % 2:
        value = mul(value, (Fraction(0), Fraction(1), Fraction(0), Fraction(-1)))
    for p in scalar.phasenodes:
        value = mul(value, add(one, phase(p)))
    if scalar.sum_of_phases:
        total = zero
        for p, coefficient in scalar.sum_of_phases.items():
            if type(coefficient) is not int:
                raise Unsupported("Noninteger coefficient in exact ZX scalar")
            total = add(total, tuple(coefficient * x for x in phase(p)))
        value = mul(value, total)
    if value != one:
        raise Incorrect("ZX identity has a nonunit scalar or a different global phase")
    return {
        "method": "PyZX identity reduction + exact cyclotomic scalar",
        "version": zx.__version__,
        "wires": len(wires),
        "scalar": "1",
        "verified": True,
    }


class StatevectorSimulator:
    """Elementary-gate simulator retaining relative and global complex phases."""

    def __init__(self, qubits):
        if type(qubits) is not int or not 1 <= qubits <= 18:
            raise ValueError("Elementary simulation limited to 18 qubits")
        self.qubits = qubits

    def run(self, gates, state, *, measurement_outcomes=None, seed=0, return_record=False):
        state = np.array(state, dtype=complex, copy=True)
        if state.shape[0] != 1 << self.qubits:
            raise ValueError("State dimension mismatch")
        indices = np.arange(1 << self.qubits)
        record = {}
        prescribed = iter(measurement_outcomes) if measurement_outcomes is not None else None
        random = np.random.default_rng(seed)
        for name, wires, angle in gates:
            if len(set(wires)) != len(wires) or any(not 0 <= q < self.qubits for q in wires):
                raise ValueError("Invalid elementary gate wires")
            if name == "oxor":
                target, left, right = angle
                if target in record or (left != -1 and left not in record) or right not in record:
                    raise Incorrect("Invalid classical outcome dependency")
                record[target] = (0 if left == -1 else record[left]) ^ record[right]
                continue
            if isinstance(angle, dict):
                if set(angle) != {"condition"} or angle["condition"] not in record:
                    raise Incorrect("Undefined classical control")
                if not record[angle["condition"]]:
                    continue
                angle = None
            if name == "global_phase":
                state *= np.exp(1j * angle)
                continue
            if name == "neg":
                state *= -1
                continue
            target = wires[-1]
            low = indices[(indices >> target & 1) == 0]
            if name in ("cx", "ccx"):
                for control in wires[:-1]:
                    low = low[(low >> control & 1) == 1]
            high = low ^ (1 << target)
            a, b = state[low].copy(), state[high].copy()
            if name == "assert_zero":
                if np.max(np.abs(b)) > 1e-9:
                    raise Incorrect("Released wire is not zero")
            elif name == "hmr":
                if angle in record:
                    raise Incorrect("Measurement overwrites a classical outcome")
                norm = float(np.sum(np.abs(state) ** 2))
                plus, minus = (a + b) / math.sqrt(2), (a - b) / math.sqrt(2)
                probability = float(np.sum(np.abs(plus) ** 2)) / norm
                outcome = (
                    next(prescribed)
                    if prescribed is not None
                    else int(random.random() >= probability)
                )
                if type(outcome) is not int or outcome not in (0, 1):
                    raise ValueError("Measurement outcome must be zero or one")
                probability = probability if outcome == 0 else 1 - probability
                if probability < 1e-14:
                    raise Incorrect("Requested measurement branch has zero probability")
                state[low] = (plus if outcome == 0 else minus) / math.sqrt(probability)
                state[high] = 0
                record[angle] = outcome
            elif name == "cz":
                state[high[(high >> wires[0] & 1) == 1]] *= -1
            elif name in ("x", "cx", "ccx"):
                state[low], state[high] = b, a
            elif name == "h":
                state[low], state[high] = (a + b) / math.sqrt(2), (a - b) / math.sqrt(2)
            elif name == "ry":
                c, s = math.cos(angle / 2), math.sin(angle / 2)
                state[low], state[high] = c * a - s * b, s * a + c * b
            elif name in ("z", "s", "sdg", "t", "tdg"):
                state[high] *= {
                    "z": -1,
                    "s": 1j,
                    "sdg": -1j,
                    "t": np.exp(1j * math.pi / 4),
                    "tdg": np.exp(-1j * math.pi / 4),
                }[name]
            else:
                raise Unsupported(f"Unsupported elementary gate: {name}")
        return (state, record) if return_record else state


class SparseStatevectorSimulator:
    """Bounded physical trajectory simulator for wide but sparse small walks.

    Values may be vectors of input columns. Measurements normalize the whole
    Kraus matrix with one scalar, never each column independently.
    """

    def __init__(self, qubits, max_states=131072):
        if not 1 <= qubits <= 8192:
            raise ValueError("Invalid sparse simulator width")
        self.qubits, self.max_states = qubits, max_states

    def run(self, gates, state, *, seed=0, measurement_outcomes=None):
        state = {key: np.asarray(value, dtype=complex).copy() for key, value in state.items()}
        random, record = np.random.default_rng(seed), {}
        prescribed = iter(measurement_outcomes) if measurement_outcomes is not None else None

        def norm(values):
            return math.fsum(float(np.sum(np.abs(v) ** 2)) for v in values)

        for name, wires, value in gates:
            if len(set(wires)) != len(wires) or any(not 0 <= q < self.qubits for q in wires):
                raise ValueError("Invalid sparse gate wires")
            if name == "oxor":
                target, a, b = value
                if target in record or (a != -1 and a not in record) or b not in record:
                    raise Incorrect("Invalid classical dependency")
                record[target] = (0 if a == -1 else record[a]) ^ record[b]
                continue
            if isinstance(value, dict):
                if set(value) != {"condition"} or value["condition"] not in record:
                    raise Incorrect("Undefined classical control")
                if not record[value["condition"]]:
                    continue
                value = None
            if name in ("global_phase", "neg"):
                phase = -1 if name == "neg" else np.exp(1j * value)
                state = {s: phase * a for s, a in state.items()}
                continue
            target = 1 << wires[-1]
            if name == "assert_zero":
                if any(np.max(np.abs(a)) > 1e-9 for s, a in state.items() if s & target):
                    raise Incorrect("Released physical wire is not zero")
            elif name in ("x", "cx", "ccx"):
                controls = sum(1 << q for q in wires[:-1])
                state = {
                    s ^ (target if s & controls == controls else 0): a for s, a in state.items()
                }
            elif name in ("z", "s", "sdg", "t", "tdg", "cz"):
                mask = sum(1 << q for q in wires)
                phase = {
                    "z": -1,
                    "cz": -1,
                    "s": 1j,
                    "sdg": -1j,
                    "t": np.exp(1j * math.pi / 4),
                    "tdg": np.exp(-1j * math.pi / 4),
                }[name]
                state = {s: a * (phase if s & mask == mask else 1) for s, a in state.items()}
            elif name in ("h", "ry", "hmr"):
                pairs = {s & ~target for s in state}
                zero = np.zeros_like(next(iter(state.values())))
                output, minus = {}, {}
                for low in pairs:
                    a, b = state.get(low, zero), state.get(low | target, zero)
                    if name in ("h", "hmr"):
                        c, d = (a + b) / math.sqrt(2), (a - b) / math.sqrt(2)
                    else:
                        cos, sin = math.cos(value / 2), math.sin(value / 2)
                        c, d = cos * a - sin * b, sin * a + cos * b
                    if np.max(np.abs(c)) > 1e-14:
                        output[low] = c
                    if np.max(np.abs(d)) > 1e-14:
                        (minus if name == "hmr" else output)[
                            low if name == "hmr" else low | target
                        ] = d
                if name == "hmr":
                    probability = norm(output.values()) / norm(state.values())
                    draw = (
                        next(prescribed)
                        if prescribed is not None
                        else int(random.random() >= probability)
                    )
                    if type(draw) is not int or draw not in (0, 1) or value in record:
                        raise Incorrect("Invalid measurement outcome")
                    probability = probability if draw == 0 else 1 - probability
                    if probability < 1e-14:
                        raise Incorrect("Zero-probability measurement branch")
                    output = output if draw == 0 else minus
                    output = {s: a / math.sqrt(probability) for s, a in output.items()}
                    record[value] = draw
                state = output
            else:
                raise Unsupported(f"Unsupported sparse gate: {name}")
            if len(state) > self.max_states:
                raise Unsupported("Sparse state limit exceeded")
        return state


class SOSSimulator:
    """Small rectangular-encoding simulation on label/term/system registers.

    Uses direct occupation-basis Givens/Pauli action, independent of elementary
    synthesis. Batch columns permit complete-system checks without a huge
    full ancilla-space unitary. Dense checks are capped before allocation.
    """

    def __init__(self, task, artifact):
        if task.mode_count > 6 or task.label_bits + task.term_bits > 8:
            raise ValueError("SOS dense simulation limit exceeded")
        self.task, self.artifact = task, artifact
        self.labels, self.terms, self.dimension = (
            1 << task.label_bits,
            1 << task.term_bits,
            1 << task.mode_count,
        )
        self.selected = []
        n = task.model["spatial_modes"]
        for group in artifact["select"]:
            mode = group["spin"] * n
            rotation = DenseFockSimulator(task.mode_count)
            if group["orbital"] >= 0:
                for p, q, turn in artifact["orbitals"][group["orbital"]]:
                    rotation.givens(p + mode, q + mode, turn, task.model["angle_bits"])
            r = rotation.matrix
            for branch in group["branches"]:
                operator = (
                    (1j ** branch["phase"])
                    * r
                    @ jordan_wigner(task.mode_count, mode, branch["pauli"])
                    @ r.T
                )
                self.selected.append((branch["label"], branch["term"], operator))

    def embedding(self):
        state = np.zeros((self.labels, self.terms, self.dimension, self.dimension), dtype=complex)
        state[0, 0] = np.eye(self.dimension)
        return state

    def _prepare(self, state, name, inverse):
        from qpe_lab.circuits.gadgets.prepare import (
            prepare_weights,
            tree_controls,
            walk_distributions,
        )

        source = self.artifact[name]
        if self.artifact.get("implementation", {}).get("prepare") == "alias":
            outer, inner = walk_distributions(self.task.model)
            source = (
                prepare_weights(outer, self.task.label_bits)
                if name == "outer"
                else [
                    gate
                    for label, weights in enumerate(inner)
                    for gate in prepare_weights(weights, self.task.term_bits, label=label)
                ]
            )

        gates = reversed(source) if inverse else source
        for gate in gates:
            total = gate["left"] + gate["right"]
            c, s = math.sqrt(gate["left"] / total), math.sqrt(gate["right"] / total)
            if inverse:
                s = -s
            bit = gate["target"]
            bits = self.task.label_bits if name == "outer" else self.task.term_bits
            controls = tree_controls(gate, bits)
            for zero in range(1 << bits):
                if zero >> bit & 1 or any((zero >> q & 1) != value for q, value in controls):
                    continue
                one = zero | (1 << bit)
                if name == "outer":
                    a, b = state[zero].copy(), state[one].copy()
                    state[zero], state[one] = c * a - s * b, s * a + c * b
                else:
                    label = gate["label"]
                    a, b = state[label, zero].copy(), state[label, one].copy()
                    state[label, zero], state[label, one] = c * a - s * b, s * a + c * b

    def run(self, state, *, encoding_only=False, enable=True):
        state = np.array(state, dtype=complex, copy=True)
        steps = self.artifact["steps"][:4] if encoding_only else self.artifact["steps"]
        for step in steps:
            name, inverse = step["block"], step["adjoint"]
            if name in ("outer", "inner"):
                self._prepare(state, name, inverse)
            elif name == "select":
                for label, term, operator in self.selected:
                    state[label, term] = (operator.conj().T if inverse else operator) @ state[
                        label, term
                    ]
            elif enable or not self.task.controlled:
                state *= -1
                if name == "rin":
                    state[0, 0] *= -1
                else:
                    state[:, 0] *= -1
        return state
