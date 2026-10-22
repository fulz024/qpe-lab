"""Rotated Majorana/number-reflection SELECT with optional shared basis changes."""


def orbital_network(angles, beta, *, bitwise=False):
    """Chronological R with R e_0 = u; integer turns are in units 2*pi/2**beta."""
    n = len(angles) + 1
    gates = []
    for q in range(1, n):
        value = angles[n - 1 - q]
        turns = [-(1 << bit) for bit in range(beta) if value >> bit & 1] if bitwise else [-value]
        gates.extend([q - 1, q, turn] for turn in turns if turn % (1 << beta))
    return gates


def select_groups(model, *, grouped=True):
    groups = []
    by_basis = {}
    for label, generator in enumerate(model["generators"]):
        for term_index, term in enumerate(generator["terms"]):
            if not term["weight"]:
                continue
            key = (term["orbital"], term["spin"])
            if not grouped or key not in by_basis:
                group = {"orbital": key[0], "spin": key[1], "branches": []}
                groups.append(group)
                by_basis[key] = group
            by_basis[key]["branches"].append(
                {"label": label, "term": term_index, "pauli": term["pauli"], "phase": term["phase"]}
            )
    return groups


def select_table(task, groups):
    """Total-domain angle/spin/Pauli/phase words for coherent SELECT.

    Padding is exactly identity, including its phase. Address is label followed
    by term; the returned word layout is shared by the executable consumer.
    """
    n, beta = task.model["spatial_modes"], task.model["angle_bits"]
    angle_width = (n - 1) * beta
    words = [0] * (1 << (task.label_bits + task.term_bits))
    for group in groups:
        orbital, spin = group["orbital"], group["spin"]
        angles = task.model["orbital_angles"][orbital] if orbital >= 0 else [0] * (n - 1)
        word = sum(
            ((-value) % (1 << beta)) << (q * beta) for q, value in enumerate(reversed(angles))
        )
        word |= spin << angle_width
        for branch in group["branches"]:
            packed = word | ("IXYZ".index(branch["pauli"]) << (angle_width + 1))
            packed |= branch["phase"] << (angle_width + 3)
            words[branch["label"] | (branch["term"] << task.label_bits)] = packed
    return words, angle_width + 5


def qrom_select(expansion, task, artifact, label_reg, term_reg, system_start):
    """Load, consume quantum angles and metadata, and coherently clear the word."""
    import math

    from qpe_lab.circuits.gadgets.qrom import coherent_lookup
    from qpe_lab.circuits.gadgets.synthesis import adjoint

    words, width = select_table(task, artifact["select"])
    n, beta = task.model["spatial_modes"], task.model["angle_bits"]
    start = expansion.scratch_start
    word_reg = list(range(start, start + width))
    expansion.scratch_start += width
    expansion.peak_qubits = max(expansion.peak_qubits, expansion.scratch_start)
    spin_wire = word_reg[(n - 1) * beta]
    pauli_reg, phase_reg = word_reg[-4:-2], word_reg[-2:]
    impl = artifact["select_implementation"]
    lookup = list(
        coherent_lookup(
            expansion,
            label_reg + term_reg,
            word_reg,
            words,
            direct=impl["kind"] == "qrom_direct",
            block_size=impl["block_size"],
        )
    )
    rotation = []
    for spin in range(2):
        for q in range(1, n):
            left, right = system_start + spin * n + q - 1, system_start + spin * n + q
            for bit in range(beta):
                rotation.append(("cx", (left, right), None))
                rotation.extend(
                    expansion.ry(
                        left,
                        2 * math.tau * (1 << bit) / (1 << beta),
                        [(right, 1), (word_reg[(q - 1) * beta + bit], 1), (spin_wire, spin)],
                    )
                )
                rotation.append(("cx", (left, right), None))
    yield from lookup
    yield from adjoint(rotation)
    yield "s", (phase_reg[0],), None
    yield "z", (phase_reg[1],), None
    for spin in range(2):
        for code, pauli in enumerate("XYZ", 1):
            controls = [(spin_wire, spin)] + [
                (wire, code >> i & 1) for i, wire in enumerate(pauli_reg)
            ]
            yield from expansion.branch(
                system_start + spin * n, pauli, 0, controls, system_start=system_start
            )
    yield from rotation
    yield from reversed(lookup)
    expansion.scratch_start = start


def descriptor_tables(task, groups):
    """Separate the small SELECT descriptor from shared orbital angle data."""
    n, beta = task.model["spatial_modes"], task.model["angle_bits"]
    orbital_bits = max(1, len(task.model["orbital_angles"]).bit_length())
    descriptors = [0] * (1 << (task.label_bits + task.term_bits))
    for group in groups:
        base = group["orbital"] + 1
        base |= group["spin"] << orbital_bits
        for branch in group["branches"]:
            word = base | ("IXYZ".index(branch["pauli"]) << (orbital_bits + 1))
            word |= branch["phase"] << (orbital_bits + 3)
            descriptors[branch["label"] | (branch["term"] << task.label_bits)] = word
    angles = [0] * (1 << orbital_bits)
    for orbital, row in enumerate(task.model["orbital_angles"], 1):
        angles[orbital] = sum(
            ((-turn) % (1 << beta)) << (j * beta) for j, turn in enumerate(reversed(row))
        )
    return descriptors, angles, orbital_bits, (n - 1) * beta


def dynamic_select(circuit, task, artifact, labels, terms, system, gradient, *, inverse=False):
    """Structured SELECT with one shared spin rotation network and measured erasure."""
    from uncompute.gadgets.arith import and_chain, release_chain

    from qpe_lab.circuits.gadgets.givens import phase_gradient_givens
    from qpe_lab.circuits.gadgets.qrom import dynamic_lookup

    impl = artifact["implementation"]
    words, angle_words, orbital_bits, angle_width = descriptor_tables(task, artifact["select"])
    descriptor = circuit.qubits(orbital_bits + 5)
    angle_register = circuit.qubits(angle_width)
    n, beta = task.model["spatial_modes"], task.model["angle_bits"]
    lookup_options = {
        "block_size": impl["lookup_block_size"],
        "unlookup_size": impl["unlookup_block_size"],
    }
    programs = artifact.get("lookup_programs", {})
    dynamic_lookup(
        circuit,
        labels + terms,
        descriptor,
        words,
        **lookup_options,
        program=programs.get("descriptor", {}).get("load"),
    )
    dynamic_lookup(
        circuit,
        descriptor[:orbital_bits],
        angle_register,
        angle_words,
        block_size=impl["angle_block_size"],
        unlookup_size=min(impl["unlookup_block_size"], len(angle_words)),
        program=programs.get("angles", {}).get("load"),
    )
    spin, pauli, phase = descriptor[orbital_bits], descriptor[-4:-2], descriptor[-2:]
    for p in range(n):
        circuit.cswap(spin, system[p], system[n + p])

    def rotations(reverse):
        order = reversed(range(1, n)) if reverse else range(1, n)
        for q in order:
            angle = angle_register[(q - 1) * beta : q * beta]
            if impl["rotations"] == "phase_gradient":
                phase_gradient_givens(
                    circuit, system[q - 1], system[q], angle, gradient, inverse=reverse
                )
            else:
                import math

                for bit, control in enumerate(angle):
                    circuit.cx(system[q - 1], system[q])
                    circuit.controlled_ry(
                        system[q - 1],
                        (-1 if reverse else 1) * 2 * math.tau * (1 << bit) / (1 << beta),
                        [(system[q], True), (control, True)],
                    )
                    circuit.cx(system[q - 1], system[q])

    rotations(True)
    for code, letter in enumerate("XYZ", 1):
        flag, trail = and_chain(
            circuit, [(wire, bool(code >> bit & 1)) for bit, wire in enumerate(pauli)]
        )
        if letter == "Y":
            circuit.sdg(system[0])
        if letter == "Z":
            circuit.cz(flag, system[0])
        else:
            circuit.cx(flag, system[0])
        if letter == "Y":
            circuit.s(system[0])
        release_chain(circuit, trail)
    rotations(False)
    for p in reversed(range(n)):
        circuit.cswap(spin, system[p], system[n + p])
    # X/Y on the down-spin block carries the parity of the original up block.
    # Pauli code 1 or 2 iff XOR(code bits) is one.
    parity_flag = circuit.qubit()
    circuit.cx(pauli[0], parity_flag)
    circuit.cx(pauli[1], parity_flag)
    flag = circuit.and_compute(spin, parity_flag)
    for wire in system[:n]:
        circuit.cz(flag, wire)
    circuit.and_release(flag, spin, parity_flag)
    circuit.cx(pauli[1], parity_flag)
    circuit.cx(pauli[0], parity_flag)
    circuit.recycle(parity_flag)
    (circuit.sdg if inverse else circuit.s)(phase[0])
    circuit.z(phase[1])
    dynamic_lookup(
        circuit,
        descriptor[:orbital_bits],
        angle_register,
        angle_words,
        erase=True,
        unlookup_size=min(impl["unlookup_block_size"], len(angle_words)),
        program=programs.get("angles", {}).get("erase"),
    )
    for wire in reversed(angle_register):
        circuit.recycle(wire)
    dynamic_lookup(
        circuit,
        labels + terms,
        descriptor,
        words,
        erase=True,
        **lookup_options,
        program=programs.get("descriptor", {}).get("erase"),
    )
    for wire in reversed(descriptor):
        circuit.recycle(wire)


def paper_select(
    circuit,
    task,
    artifact,
    labels,
    terms,
    spins,
    system,
    gradient,
    metadata,
    sign_bits,
    alias_flag,
    *,
    inverse=False,
):
    """Appendix B compact SELECT: factor angles, spin multiplexing, Majoranas.

    spins[0] is the INNER spin/Majorana selector; spins[1] is the OUTER
    one-body spin. Their different reflection membership is essential.
    """
    from qpe_lab.circuits.gadgets.givens import orbital_basis
    from qpe_lab.circuits.gadgets.qrom import dynamic_lookup, erase_sum_lookups, lookup_xor

    prep = task.model["paper"]
    n, beta = task.model["spatial_modes"], task.model["angle_bits"]
    rank, bases, copies = prep["rank"], prep["bases"], prep["copies"]
    sf, qtype, rbits = metadata[0], metadata[1], metadata[2:]
    start = len(circuit.gates)
    angle = circuit.qubits((n - 1) * beta)

    def word(orbital):
        return sum(v << (i * beta) for i, v in enumerate(artifact["orbital_angles"][orbital]))

    sf_address = terms + rbits + [sf]
    sf_words = [0] * (1 << len(sf_address))
    for r in range(rank):
        for b in range(bases):
            sf_words[b | (r << len(terms)) | (1 << (len(terms) + len(rbits)))] = word(r * bases + b)
    one_words = [0] * (1 << len(labels))
    for j in range(n):
        one_words[rank * copies + j] = word(rank * bases + j)
    lookup_xor(circuit, sf_address, angle, sf_words)
    lookup_xor(circuit, labels, angle, one_words)
    circuit.blocks.append(("angle_load", start, len(circuit.gates)))
    start = len(circuit.gates)
    spin = circuit.qubit()
    circuit.cx(spins[1], spins[0])
    circuit.ccx(sf, spins[0], spin)
    circuit.cx(spins[1], spins[0])
    circuit.cx(spins[1], spin)
    for p in range(n):
        circuit.cswap(spin, system[p], system[n + p])
    orbital_basis(circuit, system[:n], angle, gradient, inverse=True)
    # SF identity is b=B. Padding also acts as identity, on the total domain.
    active = circuit.qubit()
    valid = [int(b < bases) for b in range(1 << len(terms))]
    lookup_xor(circuit, terms, [active], valid)
    sf_active = circuit.and_compute(sf, active)
    circuit.cz(sf_active, system[0])
    circuit.and_release(sf_active, sf, active)
    dynamic_lookup(
        circuit,
        terms,
        [active],
        valid,
        erase=True,
        unlookup_size=min(4, len(valid)),
    )
    circuit.recycle(active)
    # X followed by Z is iY. The adjoint must reverse this order:
    # D=(X+iY)/2 and Q=(X-iY)/2, with Q's minus controlled by qtype.
    circuit.x(sf)
    xy = circuit.and_compute(sf, spins[0])
    if not inverse:
        circuit.cx(sf, system[0])
    circuit.cz(xy, system[0])
    if inverse:
        circuit.cx(sf, system[0])
    circuit.and_release(xy, sf, spins[0])
    circuit.x(sf)
    circuit.cz(qtype, spins[0])
    orbital_basis(circuit, system[:n], angle, gradient)
    for p in reversed(range(n)):
        circuit.cswap(spin, system[p], system[n + p])
    parity = circuit.and_compute(sf, spins[1], pol=(False, True))
    for wire in system[:n]:
        circuit.cz(parity, wire)
    circuit.and_release(parity, sf, spins[1], pol=(False, True))
    outcome = circuit.hmr(spin)
    circuit.z(spins[1], cond=outcome)
    circuit.cz(sf, spins[0], cond=outcome)
    circuit.cz(sf, spins[1], cond=outcome)
    circuit.recycle(spin)
    circuit.z(sign_bits[0])
    circuit.cz(alias_flag, sign_bits[1])
    circuit.blocks.append(("majorana_and_rotations", start, len(circuit.gates)))
    start = len(circuit.gates)
    # A single angle-word measurement serves the two disjoint angle tables.
    erase_sum_lookups(
        circuit,
        angle,
        [
            (sf_address, sf_words, min(16, len(sf_words))),
            (labels, one_words, min(16, len(one_words))),
        ],
    )
    for wire in reversed(angle):
        circuit.recycle(wire)
    circuit.blocks.append(("angle_erase", start, len(circuit.gates)))


def strict_paper_select(
    circuit,
    task,
    labels,
    terms,
    spins,
    system,
    gradient,
    metadata,
    angle,
    active,
    sf_words,
    success,
    delta_sign,
    alias_flag,
    *,
    inverse=False,
    specialize=False,
    gradient_adder="general",
    sf_lookup_control="explicit",
):
    """Appendix B SELECT with success controls and a shared angle lifetime.

    One-body angles have already been loaded during the inner x traversal.
    Only SF angles are erased here; one-body measurement phases are returned
    to that same traversal's inverse, together with metadata and alias signs.
    The zero-address option requires b=r=0 on every non-SF branch; the strict
    inner preparation establishes this invariant, including padded labels.
    """
    from uncompute.gadgets.qrom import walk, xor_outcomes, xor_word

    from qpe_lab.circuits.gadgets.givens import orbital_basis
    from qpe_lab.circuits.gadgets.qrom import ordered_walk, phase_table

    n = task.model["spatial_modes"]
    if sf_lookup_control not in ("explicit", "zero_address"):
        raise ValueError("Invalid SF lookup control")
    sf, qtype, rbits = metadata[0], metadata[1], metadata[2:]
    address = terms + rbits
    start = len(circuit.gates)
    # B42: b-first, then bounded r traversal. Its folded padding extension
    # is explicitly present in sf_words and shared by the inverse fixup.
    prep = task.model["paper"]
    offset = 1 << len(address)

    def b_leaf(b, ctrl):
        if b < prep["bases"]:
            circuit.cx(ctrl, active)
            walk(
                circuit,
                rbits,
                ctrl,
                len(rbits) - 1,
                0,
                1 << len(rbits),
                0,
                prep["rank"],
                lambda flag, r: xor_word(
                    circuit,
                    flag,
                    sf_words[offset | (r << len(terms)) | b] & ((1 << len(angle)) - 1),
                    angle,
                ),
            )

    zero_address = sf_lookup_control == "zero_address"
    ordered_walk(circuit, terms, prep["bases"] + 1, b_leaf, None if zero_address else sf)
    if zero_address:
        circuit.x(sf)
        xor_word(circuit, sf, sf_words[offset], angle + [active])
        circuit.x(sf)
    circuit.blocks.append(("angle_load", start, len(circuit.gates)))
    start = len(circuit.gates)
    spin = circuit.qubit()
    if specialize:
        circuit.cx(spins[1], spins[0])
        circuit.ccx(sf, spins[0], spin)
        circuit.cx(spins[1], spins[0])
        circuit.cx(spins[1], spin)
    else:
        circuit.ccx(sf, spins[0], spin)
        circuit.x(sf)
        circuit.ccx(sf, spins[1], spin)
        circuit.x(sf)
    for p in range(n):
        circuit.cswap(spin, system[p], system[n + p])
    orbital_basis(circuit, system[:n], angle, gradient, inverse=True, adder=gradient_adder)
    if len(success) not in (1, 2):
        raise ValueError("SELECT requires one or two explicit success controls")
    good = success[0] if len(success) == 1 else circuit.and_compute(*success)
    # angle_load already produces active = sf AND (b < B).
    sf_active = active if specialize else circuit.and_compute(sf, active)
    controlled_sf = circuit.and_compute(good, sf_active)
    circuit.cz(controlled_sf, system[0])
    circuit.and_release(controlled_sf, good, sf_active)
    if not specialize:
        circuit.and_release(sf_active, sf, active)
    one = circuit.and_compute(good, sf, pol=(True, False))
    xy = None if specialize else circuit.and_compute(sf, spins[0], pol=(False, True))
    controlled_xy = (
        circuit.and_compute(one, spins[0]) if specialize else circuit.and_compute(good, xy)
    )
    if not inverse:
        circuit.cx(one, system[0])
    circuit.cz(controlled_xy, system[0])
    if inverse:
        circuit.cx(one, system[0])
    if specialize:
        # qtype is zero on SF rows, so this reuses the existing controlled XY.
        circuit.cz(qtype, controlled_xy)
        circuit.and_release(controlled_xy, one, spins[0])
        circuit.and_release(one, good, sf, pol=(True, False))
    else:
        circuit.and_release(controlled_xy, good, xy)
        circuit.and_release(xy, sf, spins[0], pol=(False, True))
        circuit.and_release(one, good, sf, pol=(True, False))
        qsign = circuit.and_compute(qtype, spins[0])
        circuit.cz(good, qsign)
        circuit.and_release(qsign, qtype, spins[0])
    if len(success) == 2:
        circuit.and_release(good, *success)
    orbital_basis(circuit, system[:n], angle, gradient, adder=gradient_adder)
    for p in reversed(range(n)):
        circuit.cswap(spin, system[p], system[n + p])
    parity = circuit.and_compute(sf, spins[1], pol=(False, True))
    for q in system[:n]:
        circuit.cz(parity, q)
    circuit.and_release(parity, sf, spins[1], pol=(False, True))
    outcome = circuit.hmr(spin)
    circuit.cz(sf, spins[0], cond=outcome)
    circuit.x(sf)
    circuit.cz(sf, spins[1], cond=outcome)
    circuit.x(sf)
    circuit.recycle(spin)
    circuit.cz(alias_flag, delta_sign)
    circuit.blocks.append(("majorana_and_rotations", start, len(circuit.gates)))
    start = len(circuit.gates)
    outcomes = [circuit.hmr(q) for q in angle + [active]]
    # Two-dimensional unlookup: unary b, iterate r, controlled by G0.
    phases = {}
    offset = 1 << len(address)
    for a, word in enumerate(sf_words[offset:]):
        phase = xor_outcomes([o for bit, o in enumerate(outcomes) if word >> bit & 1])
        if phase is not None:
            phases[a] = phase
    phase_table(
        circuit,
        address,
        phases,
        window=1 << len(terms),
        control=None if zero_address else sf,
        high_bound=prep["rank"],
    )
    if zero_address and 0 in phases:
        circuit.x(sf)
        circuit.z(sf, cond=phases[0])
        circuit.x(sf)
    for q in reversed(angle + [active]):
        circuit.recycle(q)
    circuit.blocks.append(("angle_erase", start, len(circuit.gates)))
    return outcomes[:-1]
