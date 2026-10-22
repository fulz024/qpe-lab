"""Total-domain XOR lookups composed from the pinned upstream QROM components."""

from uncompute.gadgets.qrom import pruned_walk, selectswap_lookup, xor_word


def ordered_walk(circuit, address, count, leaf, control=None):
    """Ascending unary traversal, retaining padding controls on the full domain.

    Unlike the upstream bounded-window iterator, an out-of-range input fires
    no leaf. Ascending order permits the Appendix B SF/D1 storage handoff.
    """

    def visit(bits, lo, span, ctrl):
        if lo >= count:
            return
        if span == 1:
            leaf(lo, ctrl)
            return
        q, half = bits[-1], span // 2
        if ctrl is None:
            circuit.x(q)
            visit(bits[:-1], lo, half, q)
            circuit.x(q)
            visit(bits[:-1], lo + half, half, q)
        else:
            child = circuit.and_compute(ctrl, q, pol=(True, False))
            visit(bits[:-1], lo, half, child)
            if lo + half < count:
                circuit.cx(ctrl, child)
                visit(bits[:-1], lo + half, half, child)
                circuit.and_release(child, ctrl, q)
            else:
                circuit.and_release(child, ctrl, q, pol=(True, False))

    visit(list(address), 0, 1 << len(address), control)


def phase_table(circuit, address, phases, *, window, signs=(), control=None, high_bound=None):
    """Combined deterministic signs and measurement phases in ONE unary pass.

    Binary-to-unary conversion of the low register plus unary traversal of
    the high register is also Appendix B's two-dimensional angle unlookup.
    """
    low = window.bit_length() - 1
    if window < 1 or window & (window - 1) or low > len(address):
        raise ValueError("Invalid phase table window")
    unary = circuit.qubits(window)
    circuit.x(unary[0])
    for bit in reversed(range(low)):
        for slot in range(0, window, 1 << (bit + 1)):
            circuit.cswap(address[bit], unary[slot], unary[slot + (1 << bit)])
    signs = set(signs)
    occupied = set(phases) | signs
    count = 0 if not occupied else (max(occupied) >> low) + 1

    def leaf(high, ctrl):
        for slot, wire in enumerate(unary):
            key = (high << low) | slot
            if key in signs:
                circuit.cz(ctrl, wire)
            if key in phases:
                circuit.cz(ctrl, wire, cond=phases[key])

    if count:
        if high_bound is None:
            ordered_walk(circuit, address[low:], count, leaf, control)
        else:
            from uncompute.gadgets.qrom import walk

            high = address[low:]
            walk(
                circuit,
                high,
                control,
                len(high) - 1,
                0,
                1 << len(high),
                0,
                high_bound,
                lambda ctrl, value: leaf(value, ctrl),
            )
    for bit in range(low):
        for slot in range(0, window, 1 << (bit + 1)):
            other = unary[slot + (1 << bit)]
            circuit.cx(other, unary[slot])
            circuit.and_release(other, address[bit], unary[slot])
    circuit.x(unary[0])
    circuit.recycle(unary[0])


def mixed_inner_erase(
    circuit, labels, terms, sf_count, total_count, phases, one_phases, signs, *, window
):
    """Appendix B50: grouped SF phase fixup followed by single D1/Q1 leaves.

    The SF one-hot window is retired at the SF/D1 boundary. D1/Q1 phases
    do not depend on b and use the continued x traversal directly.
    """
    size = 1 << len(terms)
    window = max(size, window)
    while sf_count % (window // size):
        window //= 2
    low_x = (window // size).bit_length() - 1
    address = list(terms) + list(labels[:low_x])
    unary = circuit.qubits(window)
    circuit.x(unary[0])
    for bit in reversed(range(len(address))):
        for slot in range(0, window, 1 << (bit + 1)):
            circuit.cswap(address[bit], unary[slot], unary[slot + (1 << bit)])

    def retire():
        for bit in range(len(address)):
            for slot in range(0, window, 1 << (bit + 1)):
                other = unary[slot + (1 << bit)]
                circuit.cx(other, unary[slot])
                circuit.and_release(other, address[bit], unary[slot])
        circuit.x(unary[0])
        circuit.recycle(unary[0])

    def leaf(high, ctrl):
        base_x = high << low_x
        if base_x < sf_count:
            for slot, q in enumerate(unary):
                address_value = (base_x << len(terms)) | slot
                if address_value in signs:
                    circuit.cz(ctrl, q)
                if address_value in phases:
                    circuit.cz(ctrl, q, cond=phases[address_value])
            if base_x + (1 << low_x) == sf_count:
                retire()
        else:

            def one_leaf(low, flag):
                phase = one_phases.get(base_x + low)
                if phase is not None:
                    if flag is None:
                        circuit.gate("neg", [], {"condition": phase.index})
                    else:
                        circuit.z(flag, cond=phase)

            ordered_walk(
                circuit, labels[:low_x], min(1 << low_x, total_count - base_x), one_leaf, ctrl
            )

    ordered_walk(circuit, labels[low_x:], (total_count + (1 << low_x) - 1) >> low_x, leaf)


def plane_phase_table(circuit, address, phases, *, window, plane_bits=2, signs=()):
    """Joint phase repair with the discovered circuit's low-bit Boolean planes.

    Adapted from chemistry-fe2s2/circuits/discovered/circuit.py:b2_fixup.
    Unlike that task's free padding, every address here has an exact value:
    missing rows contribute zero. Classical expressions are reduced to their
    original measurement draws before the Boolean transform, so equal parity
    expressions cancel even when they have different temporary outcome ids.
    """
    from uncompute.gadgets.qrom import xor_outcomes

    from qpe_lab.circuits.gadgets.synthesis import Measurement

    if (
        type(plane_bits) is not int
        or not 0 <= plane_bits <= min(2, len(address))
        or type(window) is not int
        or window < 1
        or window & (window - 1)
        or window > 1 << (len(address) - plane_bits)
    ):
        raise ValueError("Invalid Boolean-plane repair layout")
    support, draws = {}, []
    for name, _, value in circuit.gates:
        if name == "hmr":
            draws.append(Measurement(circuit, value))
            support[value] = 1 << len(draws)  # bit zero is the constant sign
        elif name == "oxor":
            output, left, right = value
            support[output] = (0 if left == -1 else support[left]) ^ support[right]
    outcomes = {mask >> 1: Measurement(circuit, key) for key, mask in support.items() if mask}
    words = [0] * (1 << len(address))
    for row, outcome in phases.items():
        if not 0 <= row < len(words) or outcome.circuit is not circuit:
            raise ValueError("Invalid phase table address or outcome")
        words[row] ^= support[outcome.index]
    for row in signs:
        if not 0 <= row < len(words):
            raise ValueError("Invalid sign address")
        words[row] ^= 1
    low = list(address[:plane_bits])
    slots = list(address[plane_bits : plane_bits + window.bit_length() - 1])
    high = list(address[plane_bits + len(slots) :])
    planes = 1 << plane_bits
    tables = []
    for base in range(0, len(words), window * planes):
        coefficients = []
        for slot in range(window):
            row = words[base + slot * planes : base + (slot + 1) * planes]
            for bit in range(plane_bits):
                for mask in range(planes):
                    if mask >> bit & 1:
                        left, right = row[mask], row[mask ^ (1 << bit)]
                        value = left ^ right
                        parity = value >> 1
                        # Each coefficient needs at most one XOR of existing
                        # parities, rather than another reduction of all draws.
                        if parity and parity not in outcomes:
                            outcomes[parity] = xor_outcomes(
                                [outcomes[w >> 1] for w in (left, right) if w >> 1]
                            )
                        row[mask] = value
            coefficients.append(row)
        tables.append(coefficients)
    count = max((i + 1 for i, table in enumerate(tables) if any(map(any, table))), default=0)
    unary = circuit.qubits(window)
    circuit.x(unary[0])
    for bit in reversed(range(len(slots))):
        for slot in range(0, window, 1 << (bit + 1)):
            circuit.cswap(slots[bit], unary[slot], unary[slot + (1 << bit)])

    def emit(flag, table, mask):
        for wire, row in zip(unary, table, strict=True):
            word = row[mask]
            if word & 1:
                circuit.cz(flag, wire)
            parity = word >> 1
            if parity:
                if parity not in outcomes:
                    outcomes[parity] = xor_outcomes(
                        [outcome for i, outcome in enumerate(draws) if parity >> i & 1]
                    )
                circuit.cz(flag, wire, cond=outcomes[parity])

    def leaf(index, flag):
        table = tables[index]
        needed = {mask for mask in range(1, planes) if any(row[mask] for row in table)}
        emit(flag, table, 0)
        if 3 in needed:
            first = 1 if 1 in needed else 2
            a, b = (low[0], low[1]) if first == 1 else (low[1], low[0])
            single = a if flag is None else circuit.and_compute(flag, a)
            pair = circuit.and_compute(single, b)
            emit(pair, table, 3)
            circuit.and_release(pair, single, b)
            if first in needed:
                emit(single, table, first)
                needed.remove(first)
            if flag is not None:
                circuit.and_release(single, flag, a)
            needed.remove(3)
        for mask in sorted(needed):
            wire = low[mask.bit_length() - 1]
            single = wire if flag is None else circuit.and_compute(flag, wire)
            emit(single, table, mask)
            if flag is not None:
                circuit.and_release(single, flag, wire)

    if count:
        ordered_walk(circuit, high, count, leaf)
    for bit in range(len(slots)):
        for slot in range(0, window, 1 << (bit + 1)):
            other = unary[slot + (1 << bit)]
            circuit.cx(other, unary[slot])
            circuit.and_release(other, slots[bit], unary[slot])
    circuit.x(unary[0])
    circuit.recycle(unary[0])


def bounded_window_address(value, width, count):
    """Total-domain extension of upstream walk's omitted one-sided controls."""
    lo, hi, mapped = 0, 1 << width, 0
    for bit in reversed(range(width)):
        mid = (lo + hi) // 2
        if count <= mid:
            hi = mid
        elif value >> bit & 1:
            mapped |= 1 << bit
            lo = mid
        else:
            hi = mid
    return mapped


def deferred_phases(circuit, target, record):
    """Measure the selected bank and return its full deferred phase function."""
    from uncompute.gadgets.qrom import xor_outcomes

    words, records, permutations = record["words"], record["measurements"], record["permutation"]
    measurements = [[circuit.hmr(q) for q in target], *records[1:]]
    count = len(measurements)
    phases = {}
    for addr in range(len(words)):
        base, low = addr & ~(count - 1), addr & (count - 1)
        terms = []
        for bank, outcomes in enumerate(measurements):
            word = words[base | permutations[low][bank]]
            terms.extend(outcomes[i] for i in range(len(target)) if word >> i & 1)
        parity = xor_outcomes(terms)
        if parity is not None:
            phases[addr] = parity
    return phases


def deferred_lookup(circuit, address, target, words, *, block_size=1):
    """Clean, in-place QROAM, retaining its phase debt until matching erase.

    Bank zero is the output itself. Other banks are X-measured immediately so
    their wires can become angle storage. The returned classical record must
    be retired at the ORIGINAL address before any Hadamards on that address.
    Unlike a general XOR lookup this requires target initially zero.
    """
    from uncompute.gadgets.qrom import swap_perm

    address, target, words = list(address), list(target), tuple(words)
    if (
        len(words) != 1 << len(address)
        or not target
        or block_size < 1
        or block_size & (block_size - 1)
        or block_size > len(words)
        or any(type(w) is not int or not 0 <= w < 1 << len(target) for w in words)
    ):
        raise ValueError("Invalid deferred QROAM")
    low = block_size.bit_length() - 1
    banks = [target] + [circuit.qubits(len(target)) for _ in range(block_size - 1)]
    chunks = [words[i : i + block_size] for i in range(0, len(words), block_size)]

    def leaf(index, control):
        for bank, word in zip(banks, chunks[index], strict=True):
            xor_word(circuit, control, word, bank)

    pruned_walk(
        circuit,
        address[low:][::-1],
        0,
        len(chunks),
        leaf,
        lambda lo, span: any(any(row) for row in chunks[lo : lo + span]),
        lambda lo, span: len(set(chunks[lo : lo + span])) == 1,
    )
    for i in range(low):
        for slot in range(0, block_size, 1 << (i + 1)):
            for a, b in zip(banks[slot], banks[slot + (1 << i)], strict=True):
                circuit.cswap(address[i], a, b)
    measurements = [None] + [[circuit.hmr(q) for q in bank] for bank in banks[1:]]
    for bank in reversed(banks[1:]):
        for q in reversed(bank):
            circuit.recycle(q)
    return {"words": words, "measurements": measurements, "permutation": swap_perm(block_size)}


def deferred_erase(circuit, address, target, record, *, block_size=1):
    """Erase selected bank and repay ALL deferred bank phases together."""
    from uncompute.gadgets.qrom import phase_fixup, xor_outcomes

    words, records, permutations = record["words"], record["measurements"], record["permutation"]
    measurements = [[circuit.hmr(q) for q in target], *records[1:]]
    count = len(measurements)
    phases = {}
    for addr in range(len(words)):
        base, low = addr & ~(count - 1), addr & (count - 1)
        terms = []
        for bank, outcomes in enumerate(measurements):
            word = words[base | permutations[low][bank]]
            terms.extend(outcomes[i] for i in range(len(target)) if word >> i & 1)
        parity = xor_outcomes(terms)
        if parity is not None:
            phases[addr] = parity
    phase_fixup(circuit, list(address), len(words), block_size, phases)


def erase_sum_lookups(circuit, target, tables):
    """One measurement of XORed disjoint tables, with each phase fixed exactly.

    tables contains (address, full-domain words, unlookup window). The action
    also holds when the tables overlap: phase kickback is linear over XOR.
    """
    from uncompute.gadgets.qrom import phase_fixup, xor_outcomes

    outcomes = [circuit.hmr(q) for q in target]
    for address, words, window in tables:
        phases = {}
        for addr, word in enumerate(words):
            parity = xor_outcomes([o for i, o in enumerate(outcomes) if word >> i & 1])
            if parity is not None:
                phases[addr] = parity
        phase_fixup(circuit, list(address), len(words), window, phases)


def dynamic_lookup(
    circuit, address, target, words, *, block_size=1, unlookup_size=1, erase=False, program=None
):
    """Measured QROAM; erase requires target == table[address], not arbitrary XOR."""
    from uncompute.gadgets.qrom import erase_register

    address, target, words = list(address), list(target), tuple(words)
    if (
        not address
        or not target
        or len(words) != 1 << len(address)
        or len(set(address + target)) != len(address + target)
        or any(type(w) is not int or not 0 <= w < 1 << len(target) for w in words)
        or any(
            type(k) is not int or k < 1 or k & (k - 1) or k > len(words)
            for k in (block_size, unlookup_size)
        )
    ):
        raise ValueError("Invalid dynamic lookup domain or block size")
    if program is not None:
        append_lookup_program(circuit, address, target, words, program, erase=erase)
        return
    if erase:
        erase_register(
            circuit,
            target,
            address,
            len(words),
            words,
            unlookup_size,
            range(len(words)),
            "unlookup",
        )
    elif block_size == 1:
        lookup_xor(circuit, address, target, words)
    else:
        selectswap_lookup(
            circuit,
            address,
            len(words),
            words,
            len(target),
            block_size,
            unlookup_size,
            range(len(words)),
            "qroam",
            out=target,
        )


def append_lookup_program(circuit, address, target, words, encoded, *, erase):
    """Compose the actual certified operations, with disjoint wires/outcome ids."""
    import base64
    import hashlib

    from qpe_lab.circuits.checks import verify_lookup_program

    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > 32 * 1024**2:
        raise ValueError("Embedded lookup program is too large")
    key = (raw, tuple(words), erase)
    if key not in circuit.component_cache:
        circuit.component_cache[key] = verify_lookup_program(
            raw, len(address), len(target), words, erase=erase
        )
        _, _, cost, evidence = circuit.component_cache[key]
        circuit.component_evidence.append(
            {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "erase": erase,
                "cost": cost,
                "evidence": evidence,
            }
        )
    ops, registers, cost, _ = circuit.component_cache[key]
    mapping = dict(zip(registers[0][1] + registers[1][1], address + target, strict=True))
    scratch = []
    for local in range(cost["qubits"]):
        if local not in mapping:
            mapping[local] = circuit.qubit()
            scratch.append(mapping[local])
    outcomes = {}
    for name, qa, qb, qt, ot, oc in ops:
        condition = None if oc is None else {"condition": outcomes[oc].index}
        if name == "HMR":
            outcomes[ot] = circuit.hmr(mapping[qt])
        elif name == "OXOR":
            result = circuit.outcome()
            circuit.gates.append(
                (
                    "oxor",
                    (),
                    [
                        result.index,
                        outcomes[ot].index if ot in outcomes else -1,
                        outcomes[oc].index,
                    ],
                )
            )
            outcomes[ot] = result
        elif name == "R":
            circuit.gate("assert_zero", [mapping[qt]])
        elif name == "SWAP":
            circuit.cx(mapping[qb], mapping[qt])
            circuit.cx(mapping[qt], mapping[qb])
            circuit.cx(mapping[qb], mapping[qt])
        elif name == "NEG":
            circuit.gate("neg", [], condition)
        elif name == "CCZ":
            circuit.h(mapping[qt])
            circuit.gate("ccx", [mapping[qa], mapping[qb], mapping[qt]], condition)
            circuit.h(mapping[qt])
        else:
            arity = {"X": 1, "Z": 1, "CX": 2, "CZ": 2, "CCX": 3}[name]
            circuit.gate(name.lower(), [mapping[q] for q in (qa, qb, qt)[-arity:]], condition)
    for wire in reversed(scratch):
        circuit.recycle(wire)


def lookup_xor(b, address, target, words, *, method="unary", block_size=1):
    """Append |a>|z> -> |a>|z XOR words[a]> to an existing circuit.

    Address and target are disjoint, little-endian wire sequences. The caller owns
    their allocation and registration. This component does not load tasks or emit
    streams, and restores its own workspace to zero.
    """
    address, target, words = list(address), list(target), tuple(words)
    if not address or not target or len(words) != 1 << len(address):
        raise ValueError("Lookup must define every address and have a nonempty target")
    wires = address + target
    if len(wires) != len(set(wires)):
        raise ValueError("Lookup registers must contain distinct wires")
    if any(type(word) is not int or word < 0 or word.bit_length() > len(target) for word in words):
        raise ValueError("Lookup words must be nonnegative integers fitting the target")
    if method == "unary":

        def leaf(lo, ctrl):
            xor_word(b, ctrl, words[lo], target)

        pruned_walk(
            b,
            address[::-1],
            0,
            len(words),
            leaf,
            lambda lo, span: any(words[lo : lo + span]),
            lambda lo, span: len(set(words[lo : lo + span])) == 1,
        )
    elif method == "selectswap":
        if (
            type(block_size) is not int
            or block_size < 1
            or block_size & (block_size - 1)
            or block_size > len(words)
        ):
            raise ValueError("block_size must be a power of two within the table")
        selectswap_lookup(
            b,
            address,
            len(words),
            words,
            len(target),
            block_size,
            1,
            range(len(words)),
            "lookup",
            out=target,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def coherent_lookup(expansion, address, target, words, *, direct=False, block_size=1):
    """Unitary XOR QROM with clean workspace; safe to adjoint inside SELECT.

    Unary traversal shares prefix flags. Select-swap banks provide an explicit
    width/Toffoli tradeoff, including their load, swap, copy and unload costs.
    """
    address, target, words = list(address), list(target), tuple(words)
    if (
        len(words) != 1 << len(address)
        or not target
        or len(set(address + target)) != len(address + target)
        or any(type(w) is not int or not 0 <= w < 1 << len(target) for w in words)
        or type(block_size) is not int
        or block_size < 1
        or block_size & (block_size - 1)
        or block_size > len(words)
    ):
        raise ValueError("Invalid coherent QROM registers/table/block")

    def xor(word, flag):
        while word:
            bit = (word & -word).bit_length() - 1
            yield ("x", (target[bit],), None) if flag is None else ("cx", (flag, target[bit]), None)
            word &= word - 1

    if block_size > 1:
        k, width = block_size.bit_length() - 1, len(target)
        start = expansion.scratch_start
        banks = [list(range(start + i * width, start + (i + 1) * width)) for i in range(block_size)]
        expansion.scratch_start += block_size * width
        expansion.peak_qubits = max(expansion.peak_qubits, expansion.scratch_start)
        packed = tuple(
            sum(words[(i << k) + j] << (j * width) for j in range(block_size))
            for i in range(1 << (len(address) - k))
        )
        load = list(
            coherent_lookup(
                expansion, address[k:], [q for bank in banks for q in bank], packed, direct=direct
            )
        )
        swaps = []
        for bit in reversed(range(k)):
            for bank in range(1 << bit):
                for a, b in zip(banks[bank], banks[bank + (1 << bit)], strict=True):
                    swaps += [
                        ("cx", (b, a), None),
                        ("ccx", (address[bit], a, b), None),
                        ("cx", (b, a), None),
                    ]
        yield from load
        yield from swaps
        for a, b in zip(banks[0], target, strict=True):
            yield "cx", (a, b), None
        yield from reversed(swaps)
        yield from reversed(load)
        expansion.scratch_start = start
    elif direct:
        for index, word in enumerate(words):
            if word:
                before, flag = expansion._conjunction(
                    [(q, index >> i & 1) for i, q in enumerate(address)]
                )
                yield from before
                yield from xor(word, flag)
                yield from reversed(before)
    else:

        def visit(lo, span, depth, flag):
            section = words[lo : lo + span]
            if not any(section):
                return
            if all(word == section[0] for word in section):
                yield from xor(section[0], flag)
                return
            wire = address[len(address) - depth - 1]
            for value in (0, 1):
                child = lo + value * (span // 2)
                if not any(words[child : child + span // 2]):
                    continue
                if not value:
                    yield "x", (wire,), None
                if flag is None:
                    yield from visit(child, span // 2, depth + 1, wire)
                else:
                    scratch = expansion.scratch_start + depth - 1
                    expansion.peak_qubits = max(expansion.peak_qubits, scratch + 1)
                    yield "ccx", (flag, wire, scratch), None
                    yield from visit(child, span // 2, depth + 1, scratch)
                    yield "ccx", (flag, wire, scratch), None
                if not value:
                    yield "x", (wire,), None

        yield from visit(0, len(words), 0, None)
