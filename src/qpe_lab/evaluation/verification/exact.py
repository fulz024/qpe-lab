"""Exact proofs of one framework-generated verification goal."""

from qpe_lab.circuits.checks import canonical_records, verify_phase_fusion
from qpe_lab.circuits.sim import Incorrect, Unsupported
from qpe_lab.contracts import VerificationGoal, VerificationResult


def verify(goal: VerificationGoal, context=None) -> VerificationResult:
    try:
        try:
            evidence = prove(goal)
        except (Incorrect, Unsupported) as exc:
            if goal.semantics != "qpe.phase-sensitive-instrument.v1":
                raise
            # A counterexample on arbitrary gradient inputs is NOT a
            # counterexample on the frozen resource-state subspace.
            if isinstance(exc, Incorrect) and not goal.resources:
                raise
            if goal.external_records or goal.live_records:
                raise Unsupported(str(exc)) from exc
            from qpe_lab.circuits.sim import tensor_isometry_check, verify_gradient_monomial

            if goal.resources:
                try:
                    evidence = verify_gradient_monomial(
                        goal.reference["gates"], goal.proposed, goal.forms, goal.resources
                    )
                    return VerificationResult("proved", goal.sha256, evidence)
                except Unsupported:
                    pass
            evidence = tensor_isometry_check(
                goal.reference["gates"], goal.proposed, goal.forms, goal.resources
            )
        return VerificationResult("proved", goal.sha256, evidence)
    except Incorrect as exc:
        return VerificationResult("counterexample", goal.sha256, detail=str(exc))
    except Unsupported as exc:
        return VerificationResult("unknown", goal.sha256, detail=str(exc), reason="unsupported")


def prove(goal):
    from qpe_lab.circuits.sim import (
        exact_clifford_t_columns,
        verify_channel_equivalence,
        verify_monomial_equivalence,
        verify_zx_equivalence,
    )

    if goal.semantics not in {"qpe.phase-sensitive-instrument.v1", "qpe.clean-wire-allocation.v1"}:
        raise Unsupported("Exact backend does not cover this equivalence relation")
    before, after = goal.reference["gates"], goal.proposed
    if (goal.proof or {}).get("kind") == "reuse":
        if goal.reference.get("ancilla_initial") != "zero":
            raise Unsupported("Qubit reuse requires a frozen clean-ancilla contract")
        return verify_qubit_reuse(
            before, after, goal.proof["moves"], goal.reference["protected_wires"]
        )
    if before == after:
        return {"method": "identical concrete program"}
    # This structural proof retains all classical records and arbitrary inputs.
    try:
        words = (before, after)
        internal = not goal.external_records and not goal.live_records
        if internal:
            words = tuple(canonical_records(word) for word in words)
        return verify_phase_fusion(*words, affine=True, complements=True) | {
            "internal_record_alpha_renaming": internal,
        }
    except (Incorrect, Unsupported):
        pass
    wires = goal.wires
    try:
        if len(wires) <= 4 and max(len(before), len(after)) <= 128:
            columns = goal.columns
            if any(n in {"hmr", "oxor"} or v is not None for n, _, v in before + after):
                proof = verify_channel_equivalence(
                    before,
                    after,
                    wires,
                    columns,
                    external=goal.external_records,
                    outputs=goal.live_records,
                )
                return {"input_columns": columns, "instrument": proof, **proof}
            mapping = {q: i for i, q in enumerate(wires)}
            words = [
                [(n, tuple(mapping[q] for q in qs), v) for n, qs, v in word]
                for word in (before, after)
            ]
            if exact_clifford_t_columns(words[0], len(wires), columns) != exact_clifford_t_columns(
                words[1], len(wires), columns
            ):
                raise Incorrect("Circuit differs on an exact input column (amplitude or phase)")
            return {"method": "exact input columns", "input_columns": columns}
    except Unsupported:
        pass
    failures = []
    if not goal.external_records and not goal.live_records:
        try:
            return verify_monomial_equivalence(before, after, goal.inputs, forms=goal.forms)
        except Unsupported as exc:
            failures.append(str(exc))
    if len(wires) <= 32 and max(len(before), len(after)) <= 4096:
        try:
            return verify_zx_equivalence(before, after)
        except (Incorrect, Unsupported) as exc:
            failures.append(str(exc))
    raise Unsupported("Exact methods did not complete this goal; " + "; ".join(failures))


def verify_qubit_reuse(before, after, moves, protected):
    """Certify a physical register allocation on a clean-ancilla input subspace.

    Gates/measurements/classical records are unchanged except for wire names.
    A binding is injective, I/O locations stay fixed, and a released wire must
    be independently known to be |0>. This is an isometry, including on inputs
    entangled with an external system; no state is discarded using assertions.
    """
    if len(before) != len(after):
        raise Incorrect("Qubit reuse changed the number of operations")
    protected = set(protected)
    bindings = {q: q for q in protected}
    occupied = dict(bindings)
    facts = {q: None for q in protected}
    cursor, releases = 0, 0
    last_position = 0
    for move in moves:
        if (
            not isinstance(move, (list, tuple))
            or len(move) != 3
            or any(type(v) is not int for v in move)
            or not last_position <= move[0] <= len(before)
            or not 0 <= move[1] < 8192
            or not -1 <= move[2] < 8192
        ):
            raise Incorrect("Malformed qubit allocation event")
        last_position = move[0]
    for index in range(len(before) + 1):
        while cursor < len(moves) and moves[cursor][0] == index:
            _, logical, physical = moves[cursor]
            if logical in protected:
                raise Incorrect("Qubit reuse moved a protected input/output wire")
            if facts.get(logical, 0) != 0:
                raise Incorrect("Qubit reuse attempted to release or bind an unproved-clean wire")
            if physical == -1:
                if logical not in bindings:
                    raise Incorrect("Qubit reuse released an inactive wire")
                del occupied[bindings.pop(logical)]
                releases += 1
            else:
                if logical in bindings or physical in occupied:
                    raise Incorrect("Qubit reuse overlaps live logical wires")
                bindings[logical] = physical
                occupied[physical] = logical
            cursor += 1
        if index == len(before):
            break
        name, wires, value = before[index]
        if any(q not in bindings for q in wires):
            raise Incorrect("Qubit reuse omitted an active wire binding")
        expected = name, tuple(bindings[q] for q in wires), value
        if after[index] != expected:
            raise Incorrect(
                "Qubit reuse changed a gate, measurement record, or classical dependency"
            )
        # Only actual operations establish facts. In particular, assert_zero
        # does not change this ledger. Conditional basis changes are unknown.
        if name == "hmr":
            facts[wires[0]] = 0
        elif name == "x" and value is None:
            bit = facts.get(wires[0], 0)
            facts[wires[0]] = None if bit is None else bit ^ 1
        elif name == "cx" and value is None:
            a, b = (facts.get(q, 0) for q in wires)
            facts[wires[1]] = None if a is None or b is None else a ^ b
        elif name in ("h", "x", "cx"):
            facts[wires[-1]] = None
    if cursor != len(moves):
        raise Incorrect("Qubit reuse has unconsumed allocation events")
    return {
        "verified": True,
        "method": "injective clean-wire lifetime allocation",
        "initial_ancillas": "zero",
        "releases": releases,
        "events": len(moves),
        "protected_wires_fixed": True,
    }
