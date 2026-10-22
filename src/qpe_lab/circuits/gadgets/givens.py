"""Bit-controlled orbital rotation components; angles remain quantum registers."""


def variable_givens(builder, p, q, angle_offset, angle_bits, *, inverse=False):
    """Append one rotation as commuting controls on the bits of its angle word.

    G(theta) acts on the one-particle sector as [[cos,sin],[-sin,cos]],
    fixes vacuum and double occupancy, and includes fermionic parity for
    nonadjacent modes. Each step is an ideal logical rotation, not a free gate
    or a Clifford+T synthesis cost estimate.
    """
    sign = -1 if inverse else 1
    for bit in range(angle_bits):
        builder.givens(p, q, angle_offset + bit, sign * (1 << bit))


def inverse_elimination_network(builder, spatial_modes, spin_copies, angle_bits):
    """Map orbital e0 to the frozen shared vector in each spin block."""
    for spin in range(spin_copies):
        offset = spin * spatial_modes
        for j in range(1, spatial_modes):
            angle_index = spatial_modes - 1 - j
            variable_givens(
                builder,
                offset + j - 1,
                offset + j,
                angle_index * angle_bits,
                angle_bits,
                inverse=True,
            )


def prepare_phase_gradient(circuit, register):
    """Prepare sum_y exp(-2*pi*i*y/2**b)|y>/sqrt(2**b), up to a scalar.

    Its adjoint is charged when the catalyst is retired. The scalar cancels
    between preparation and retirement; no data-dependent phase is discarded.
    """
    import math

    start = len(circuit.gates)
    for bit, q in enumerate(register):
        circuit.h(q)
        # RZ(phi) = H Sdag RY(phi) S H, in chronological order below.
        circuit.h(q)
        circuit.s(q)
        circuit.ry(q, -math.tau * (1 << bit) / (1 << len(register)))
        circuit.sdg(q)
        circuit.h(q)
    return list(circuit.gates[start:])


def constant_gradient_rotation(circuit, target, gradient, turns):
    """RY(4*pi*turns/2**s), Sanders et al. Appendix A, with classical carries.

    The high gradient bit is |->: its carry is a phase, not a computed AND.
    The first nonzero constant bit supplies a carry without an AND. Thus an
    s-bit odd constant needs at most s-3 Toffolis. Every temporary carry is
    measured and phase-corrected before its input bits change.
    """
    import math

    size = len(gradient)
    if size < 2:
        raise ValueError("Constant gradient rotation needs at least two bits")
    turns %= 1 << size
    circuit.sdg(target)
    circuit.h(target)
    for q in gradient:
        circuit.x(q)
        circuit.cx(target, q)
    if turns >> (size - 1) & 1:
        circuit.gate("global_phase", [], math.pi)

    def ripple(p, carry):
        bit, wire = (turns >> p) & 1, gradient[p]
        if p == size - 2:
            if carry is None:
                if bit:
                    circuit.z(wire)
            elif bit:
                circuit.x(carry)
                circuit.x(wire)
                circuit.cz(carry, wire)
                circuit.x(wire)
                circuit.x(carry)
                circuit.gate("global_phase", [], math.pi)
            else:
                circuit.cz(carry, wire)
        elif carry is None:
            ripple(p + 1, wire if bit else None)
        else:
            polarity = (not bit, not bit)
            nxt = circuit.and_compute(carry, wire, pol=polarity)
            if bit:
                circuit.x(nxt)
            ripple(p + 1, nxt)
            if bit:
                circuit.x(nxt)
            circuit.and_release(nxt, carry, wire, pol=polarity)
        if bit:
            circuit.x(wire)
        if carry is not None:
            circuit.cx(carry, wire)

    ripple(0, None)
    for q in reversed(gradient):
        circuit.cx(target, q)
        circuit.x(q)
    circuit.h(target)
    circuit.s(target)


def phase_gradient_add(circuit, target, operand, control):
    """Controlled addition with the highest target bit fixed in |->.

    Keep the lower-bit permutation of modular addition. The dropped high sum
    bit contributes (-1)**(controlled_high_addend XOR carry_out). Its addend
    is a CZ; its last carry is another CZ on the transformed ripple inputs,
    with the incoming carry's Z. No high addend/carry ancillas are needed.

    This is not a general adder: the highest target bit must be |->. All other
    input bits may be coherent and entangled. Costs max(0, 2*n-3) CCX.
    """
    size = len(target)
    wires = [*target, *operand, control]
    if not size or len(operand) != size or len(set(wires)) != len(wires):
        raise ValueError("Invalid phase-gradient adder registers")
    circuit.cz(operand[-1], control)

    def ripple(bit, carry=None):
        addend = circuit.and_compute(operand[bit], control)
        if carry is not None:
            circuit.cx(carry, addend)
            circuit.cx(carry, target[bit])
        if bit == size - 2:
            circuit.cz(addend, target[bit])
            if carry is not None:
                circuit.z(carry)
        else:
            nxt = circuit.and_compute(addend, target[bit])
            if carry is not None:
                circuit.cx(carry, nxt)
            ripple(bit + 1, nxt)
            if carry is not None:
                circuit.cx(carry, nxt)
            circuit.and_release(nxt, addend, target[bit])
        if carry is not None:
            circuit.cx(carry, addend)
        circuit.cx(addend, target[bit])
        circuit.and_release(addend, operand[bit], control)

    if size > 1:
        ripple(0)


def orbital_basis(circuit, system, angles, gradient, *, inverse=False, adder="general"):
    """R e0=u from stored high-to-low elimination angles; inverse emits R†.

    A positive elementary Givens maps (u_left,u_right) towards (norm,0).
    Preparing the orbital therefore uses negative turns in reversed *storage*
    order. Reversing just the gates or just the angle word gives a different
    orbital, even though the forward/adjoint sandwich still looks unitary.
    """
    beta = len(gradient)
    if len(angles) != (len(system) - 1) * beta:
        raise ValueError("Orbital angle word has the wrong width")
    pairs = range(1, len(system))
    for right in reversed(pairs) if inverse else pairs:
        offset = (len(system) - 1 - right) * beta
        phase_gradient_givens(
            circuit,
            system[right - 1],
            system[right],
            angles[offset : offset + beta],
            gradient,
            inverse=not inverse,
            adder=adder,
        )


def phase_gradient_givens(circuit, p, q, angle, gradient, *, inverse=False, adder="general"):
    """Exact dyadic Givens via a measured ripple adder and a restored catalyst.

    After CX(p,q), q selects the one-particle sector. Complementing the phase
    register when p=0 changes an addition into subtraction, giving RZ(2*theta)
    with both signs, and hence no spurious phase on the quantum angle register.
    """
    from uncompute.gadgets.arith import add

    if (
        adder not in ("general", "phase")
        or q != p + 1
        or not angle
        or len(angle) != len(gradient)
        or len(set([p, q] + list(angle) + list(gradient))) != 2 + 2 * len(angle)
    ):
        raise ValueError("Invalid phase-gradient Givens registers")
    if inverse:
        circuit.z(p)
    circuit.cx(p, q)
    circuit.sdg(p)
    circuit.h(p)
    for wire in gradient:
        circuit.x(wire)
        circuit.cx(p, wire)
    if adder == "phase":
        phase_gradient_add(circuit, gradient, angle, q)
    else:
        add(circuit, gradient, angle, ctrl=q)
    for wire in reversed(gradient):
        circuit.cx(p, wire)
        circuit.x(wire)
    circuit.h(p)
    circuit.s(p)
    circuit.cx(p, q)
    if inverse:
        circuit.z(p)
