"""Exact nonnegative state preparation by a binary tree of algebraic RY gates."""

import math


def prepare_weights(weights, bits, *, label=None, order=None):
    """Prepare amplitudes sqrt(w/sum(w)); left/right encode exact integer ratios.

    A gate on bit t is controlled by the higher bits equaling prefix. Angles
    are 2*atan2(sqrt(right), sqrt(left)); zero-probability subtrees stay identity.
    """
    if (
        len(weights) > 1 << bits
        or any(type(w) is not int or w < 0 for w in weights)
        or not sum(weights)
    ):
        raise ValueError("Invalid state preparation weights")
    order = list(reversed(range(bits))) if order is None else list(order)
    if sorted(order) != list(range(bits)):
        raise ValueError("Preparation order must permute the register bits")
    # Reindex the distribution into tree coordinates; the physical output labels
    # never change. This permits different trees for the same fixed distribution.
    padded = list(weights) + [0] * ((1 << bits) - len(weights))
    padded = [
        padded[sum(((i >> (bits - j - 1)) & 1) << bit for j, bit in enumerate(order))]
        for i in range(1 << bits)
    ]
    gates = []
    for target in reversed(range(bits)):
        width = 1 << target
        for prefix in range(1 << (bits - target - 1)):
            start = prefix * 2 * width
            left = sum(padded[start : start + width])
            right = sum(padded[start + width : start + 2 * width])
            if not right:
                continue
            gcd = math.gcd(left, right)
            gates.append(
                {
                    "target": target,
                    "prefix": prefix,
                    "left": left // gcd,
                    "right": right // gcd,
                    "label": label,
                }
            )
    if order != list(reversed(range(bits))):
        for gate in gates:
            depth = bits - 1 - gate["target"]
            gate["target"] = order[depth]
            gate["controls"] = [
                [bit, (gate["prefix"] >> (depth - j - 1)) & 1]
                for j, bit in enumerate(order[:depth])
            ]
    return gates


def tree_controls(gate, bits):
    """Physical controls, including the legacy MSB-first tree convention."""
    return gate.get(
        "controls",
        [
            [q, (gate["prefix"] >> (q - gate["target"] - 1)) & 1]
            for q in range(gate["target"] + 1, bits)
        ],
    )


def quantize_distribution(weights, bits, *, pad=True):
    """Deterministic largest-remainder quantization to N * 2**bits cells.

    N is padded to a power of two. This is target preparation, never an agent
    permission to change a frozen task.
    """
    if not weights or any(type(w) is not int or w < 0 for w in weights) or not sum(weights):
        raise ValueError("Invalid distribution")
    if type(bits) is not int or not 1 <= bits <= 24:
        raise ValueError("Invalid alias precision")
    size = 1 << max(1, (len(weights) - 1).bit_length()) if pad else len(weights)
    weights = list(weights) + [0] * (size - len(weights))
    total, cells = sum(weights), size * (1 << bits)
    quotients = [divmod(w * cells, total) for w in weights]
    counts = [q for q, _ in quotients]
    for index in sorted(range(size), key=lambda i: (-quotients[i][1], i))[: cells - sum(counts)]:
        counts[index] += 1
    return counts


def alias_table(counts, bits, *, compact=False):
    """Construct a coherent alias table with integer, exactly checkable cells."""
    size, capacity = len(counts), 1 << bits
    if (
        size < 1
        or (not compact and (size < 2 or size & (size - 1)))
        or any(type(w) is not int or w < 0 for w in counts)
        or sum(counts) != size * capacity
    ):
        raise ValueError("Alias cells do not form the fixed uniform domain")
    work = list(counts)
    small = [i for i, w in enumerate(work) if w < capacity]
    large = [i for i, w in enumerate(work) if w >= capacity]
    keep, alternate = [capacity] * size, list(range(size))
    while small:
        low, high = small.pop(), large.pop()
        keep[low], alternate[low] = work[low], high
        work[high] -= capacity - work[low]
        (small if work[high] < capacity else large).append(high)
    if compact:
        # A full bucket aliases to itself. Always swapping identical labels
        # represents it exactly without an extra endpoint bit.
        keep = [0 if k == capacity else k for k in keep]
    return {"keep": keep, "alt": alternate}


def alias_counts(table, bits, *, compact=False):
    """Recover the actual distribution, allowing noncanonical alias schedules."""
    if not isinstance(table, dict) or set(table) != {"keep", "alt"}:
        raise ValueError("Invalid alias fields")
    keep, alternate = table["keep"], table["alt"]
    if (
        not isinstance(keep, list)
        or not isinstance(alternate, list)
        or len(keep) != len(alternate)
        or not (1 if compact else 2) <= len(keep) <= 512
        or (not compact and len(keep) & (len(keep) - 1))
    ):
        raise ValueError("Invalid alias domain")
    counts, capacity = [0] * len(keep), 1 << bits
    for i, (threshold, alt) in enumerate(zip(keep, alternate, strict=True)):
        if (
            type(threshold) is not int
            or not 0 <= threshold <= capacity
            or type(alt) is not int
            or not 0 <= alt < len(keep)
        ):
            raise ValueError("Invalid alias threshold or alternate")
        counts[i] += threshold
        counts[alt] += capacity - threshold
    return counts


def measured_compare(circuit, left, right, target):
    """XOR [left >= right], with one temporary AND per carry bit.

    Carry erasure is measurement plus phase correction, not a second Toffoli
    ripple. Both inputs and arbitrary target superpositions are preserved.
    """
    if not left or len(left) != len(right):
        raise ValueError("Comparator operand widths differ")
    initial = circuit.qubit()
    circuit.x(initial)
    for q in right:
        circuit.x(q)
    carry, trail = initial, []
    for a, b in zip(left, right, strict=True):
        circuit.cx(carry, a)
        circuit.cx(carry, b)
        nxt = circuit.and_compute(a, b)
        circuit.cx(carry, nxt)
        circuit.cx(carry, b)
        circuit.cx(carry, a)
        trail.append((carry, a, b, nxt))
        carry = nxt
    circuit.cx(carry, target)
    for prev, a, b, nxt in reversed(trail):
        circuit.cx(prev, nxt)
        circuit.cx(prev, a)
        circuit.cx(prev, b)
        circuit.and_release(nxt, a, b)
        circuit.cx(prev, b)
        circuit.cx(prev, a)
    for q in right:
        circuit.x(q)
    circuit.x(initial)
    circuit.recycle(initial)


def uniform_range(circuit, index, size, auxiliary=None, *, inverse=False):
    """Exact one-round amplitude amplification for a non-power-of-two range.

    The auxiliary is part of the logical projector, NOT disposable scratch:
    the inverse on a phased uniform state need not leave it zero. All index
    states, including padding, have a defined unitary action.
    """
    import math

    if not 1 <= size <= 1 << len(index):
        raise ValueError("Uniform range does not fit its register")
    if size == 1 << len(index):
        for q in index:
            circuit.h(q)
        return
    if auxiliary is None:
        raise ValueError("Nonbinary uniform range needs a persistent auxiliary")
    angle = 2 * math.acos(math.sqrt((1 << len(index)) / (4 * size)))

    def a(adjoint=False):
        if adjoint:
            circuit.ry(auxiliary, -angle)
        for q in index:
            circuit.h(q)
        if not adjoint:
            circuit.ry(auxiliary, angle)

    def good_phase():
        constant = circuit.qubits(len(index))
        flag = circuit.qubit()
        for i, q in enumerate(constant):
            if size >> i & 1:
                circuit.x(q)
        measured_compare(circuit, index, constant, flag)
        circuit.x(flag)
        circuit.x(auxiliary)
        circuit.cz(flag, auxiliary)
        circuit.x(auxiliary)
        circuit.x(flag)
        measured_compare(circuit, index, constant, flag)
        for i, q in enumerate(constant):
            if size >> i & 1:
                circuit.x(q)
        for q in [flag, *constant]:
            circuit.recycle(q)

    if inverse:
        a(True)
        circuit.reflection([*index, auxiliary])
        a()
        good_phase()
        a(True)
    else:
        a()
        good_phase()
        a(True)
        circuit.reflection([*index, auxiliary])
        a()


def flagged_uniform_parameters(size, width, precision=7):
    """Sanders et al. (2020), Eqs. (204)-(206); integer-rounded rotation."""
    if not 2 <= precision <= 16 or not 1 <= size <= 1 << width:
        raise ValueError("Invalid flagged uniform parameters")
    if size == 1 << width:
        return {"turns": 0, "success_probability": 1.0, "precision": precision}
    theta = math.asin(math.sqrt((1 << width) / (4 * size)))
    turns = round(theta / math.tau * (1 << precision))
    sine = math.sin(math.tau * turns / (1 << precision))
    ratio = size / (1 << width)
    probability = ratio * (
        (1 - 4 * ratio * sine**2 + 2 * sine**2) ** 2 + 4 * sine**2 * (1 - sine**2)
    )
    return {"turns": turns, "success_probability": probability, "precision": precision}


def flagged_uniform(circuit, index, size, auxiliary, success, gradient, *, inverse=False):
    """Two-rotation flagged preparation, Sec. III E 2 of arXiv:2007.07391.

    The final auxiliary is deliberately retained. Success tests the INDEX,
    not this auxiliary. The three comparisons are exact reversible XORs.
    """
    from qpe_lab.circuits.gadgets.givens import constant_gradient_rotation
    from qpe_lab.circuits.gadgets.qrom import lookup_xor

    parameters = flagged_uniform_parameters(size, len(index), len(gradient))
    if size == 1 << len(index):
        for q in index:
            circuit.h(q)
        circuit.x(success)
        return

    def compare():
        lookup_xor(circuit, index, [success], [int(i < size) for i in range(1 << len(index))])

    def hadamards():
        for q in index:
            circuit.h(q)

    def rotate(sign):
        constant_gradient_rotation(circuit, auxiliary, gradient, sign * parameters["turns"])

    if inverse:
        compare()
        hadamards()
        circuit.reflection([*index, auxiliary])
        hadamards()
        compare()
        rotate(1)
        circuit.cz(success, auxiliary)
        rotate(-1)
        compare()
        hadamards()
    else:
        hadamards()
        compare()
        rotate(1)
        circuit.cz(success, auxiliary)
        rotate(-1)
        compare()
        hadamards()
        circuit.reflection([*index, auxiliary])
        hadamards()
        compare()


def walk_distributions(model):
    """Fixed outer and conditional inner distributions in either task profile."""
    if "preparation" in model:
        return model["preparation"]["outer"], model["preparation"]["inner"]
    inner = [[t["weight"] for t in g["terms"]] for g in model["generators"]]
    return [sum(row) ** 2 for row in inner], inner


def compare_greater_equal(circuit, left, right, target):
    """XOR [left >= right] into target without changing either operand."""
    from uncompute.gadgets.arith import maj, maj_inv

    if len(left) != len(right) or not left:
        raise ValueError("Comparator operands must have equal, nonzero widths")
    carry = circuit.qubit()
    circuit.x(carry)
    for q in right:
        circuit.x(q)
    previous = [carry] + list(right[:-1])
    for c, a, b in zip(previous, left, right, strict=True):
        maj(circuit, c, a, b)
    circuit.cx(right[-1], target)
    for c, a, b in reversed(list(zip(previous, left, right, strict=True))):
        maj_inv(circuit, c, a, b)
    for q in right:
        circuit.x(q)
    circuit.x(carry)
    circuit.recycle(carry)


def alias_prepare(
    circuit,
    index,
    sigma,
    word,
    flag,
    tables,
    *,
    labels=(),
    inverse=False,
    block_size=1,
    unlookup_size=1,
    program=None,
):
    """Executable uniform/lookup/compare/swap preparation and explicit inverse.

    The selected index is swapped with alt, so original bucket, threshold,
    sigma and comparison flag remain coherent garbage until the inverse.
    """
    from qpe_lab.circuits.gadgets.qrom import dynamic_lookup

    bits, width = len(sigma), len(index)
    size = 1 << width
    if len(tables) != 1 << len(labels):
        raise ValueError("Conditional alias table count does not match labels")
    words = [
        tables[label]["alt"][term] | (tables[label]["keep"][term] << width)
        for term in range(size)
        for label in range(len(tables))
    ]
    if len(word) != width + bits + 1:
        raise ValueError("Alias word must include the full-keep endpoint bit")
    alternate, keep = word[:width], word[width:]

    def compare():
        zero = circuit.qubit()
        compare_greater_equal(circuit, sigma + [zero], keep, flag)
        circuit.recycle(zero)

    def swap():
        for a, b in zip(index, alternate, strict=True):
            circuit.cswap(flag, a, b)

    if inverse:
        swap()
        compare()
        dynamic_lookup(
            circuit,
            list(labels) + index,
            word,
            words,
            erase=True,
            block_size=block_size,
            unlookup_size=unlookup_size,
            program=program,
        )
        for q in sigma + index:
            circuit.h(q)
    else:
        for q in index + sigma:
            circuit.h(q)
        dynamic_lookup(
            circuit,
            list(labels) + index,
            word,
            words,
            block_size=block_size,
            unlookup_size=unlookup_size,
            program=program,
        )
        compare()
        swap()


def paper_layout(model):
    """Validate spin-free factor structure and recover Appendix B's compact x.

    The original expanded generators remain the independent Hamiltonian input.
    No orbital identifiers or signs supplied by a candidate are trusted here.
    """
    sf, single = {}, {}
    for index, generator in enumerate(model["generators"]):
        parts = generator["name"].split("-")
        if len(parts) != 3 or parts[0] not in ("SF", "D", "Q"):
            raise ValueError("Paper layout requires named DFTHC factors")
        a, b = map(int, parts[1:])
        destination = sf if parts[0] == "SF" else single
        if (a, b) in destination:
            raise ValueError("Duplicate DFTHC factor")
        destination[a, b] = index
    if not sf:
        raise ValueError("Missing spin-free factors")
    rank = 1 + max(r for r, _ in sf)
    copies = 1 + max(c for _, c in sf)
    bases = (len(model["generators"][sf[0, 0]]["terms"]) - 1) // 2
    n = model["spatial_modes"]
    if set(sf) != {(r, c) for r in range(rank) for c in range(copies)} or set(single) != {
        (j, s) for j in range(n) for s in range(2)
    }:
        raise ValueError("Paper layout requires complete rectangular factors")
    for (r, _), index in sf.items():
        terms = model["generators"][index]["terms"]
        if (
            len(terms) != 2 * bases + 1
            or not bases
            or (terms[0]["pauli"], terms[0]["orbital"], terms[0]["phase"])
            not in (("I", -1, 0), ("I", -1, 2))
        ):
            raise ValueError("Invalid spin-free identity")
        for b in range(bases):
            up, down = terms[1 + 2 * b : 3 + 2 * b]
            if (
                up["spin"] != 0
                or down != {**up, "spin": 1}
                or up["pauli"] != "Z"
                or up["orbital"] != r * bases + b
                or up["phase"] not in (0, 2)
            ):
                raise ValueError("Spin-free terms do not share one orbital and coefficient")
    for j in range(n):
        g0, g1 = [model["generators"][single[j, s]] for s in range(2)]
        if g0["name"][0] != g1["name"][0]:
            raise ValueError("Creation/annihilation spin pair differs")
        for spin, generator in enumerate((g0, g1)):
            terms = generator["terms"]
            if len(terms) != 2:
                raise ValueError("Majorana pair must have two terms")
            a, b = terms
            if (
                a["pauli"] != "X"
                or a["phase"] != 0
                or a["spin"] != spin
                or a["orbital"] != rank * bases + j
                or b != {**a, "pauli": "Y", "phase": 1 if g0["name"][0] == "D" else 3}
            ):
                raise ValueError("Invalid Majorana pair")
        if g0["terms"][0]["weight"] != g1["terms"][0]["weight"]:
            raise ValueError("One-body spin weights differ")
    if len(model["orbital_angles"]) != rank * bases + n:
        raise ValueError("Unexpected paper orbital table")
    return {
        "rank": rank,
        "bases": bases,
        "copies": copies,
        "sf": [sf[r, c] for r in range(rank) for c in range(copies)],
        "single": [[single[j, s] for s in range(2)] for j in range(n)],
    }
