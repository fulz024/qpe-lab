"""Pure circuit checks shared by compilation and independent evaluation."""

from qpe_lab.circuits.sim import Incorrect, SymbolicSimulator


def elementary_cost(gates, protected_wires):
    """Count explicit physical wires and actual gates, independently of a builder.

    Physical line labels may have gaps. Every referenced line and every protected
    input/output line occupies a qubit; no claimed releases or costs are read.
    """
    from collections import Counter

    verify_phase_fusion(gates, gates)  # Includes classical dependency validation.
    counts = Counter(g[0] for g in gates)
    wires = set(protected_wires) | {q for _, qs, _ in gates for q in qs}
    t = counts["t"] + counts["tdg"]
    return {
        "status": "complete",
        "t_count": t,
        "qubits": len(wires),
        "tq_cost": t * len(wires),
        "operations": len(gates),
        "clifford": sum(
            n
            for gate, n in counts.items()
            if gate not in ("t", "tdg", "hmr", "oxor", "assert_zero")
        ),
        "measurements": counts["hmr"],
        "classical_xor": counts["oxor"],
        "compiled_elementary_gates": dict(sorted(counts.items())),
        "physical_wires": sorted(wires),
    }


def classical_interface(gates):
    """Classical definitions and reads; all records are local to one entry."""
    definitions, reads = set(), set()
    for name, _, value in gates:
        if name == "hmr":
            definitions.add(value)
        elif name == "oxor":
            definitions.add(value[0])
            reads.update(x for x in value[1:] if x != -1)
        elif isinstance(value, dict):
            reads.add(value["condition"])
    return definitions, reads


def canonical_records(gates):
    """Alpha-rename an entry's internal records, retaining every dependency.

    Only use when no classical inputs or observable outputs cross the goal
    boundary. The caller keeps the original gates for hashing and cost.
    """
    from qpe_lab.circuits.sim import Unsupported

    records, result = {}, []
    for name, wires, value in gates:
        if name in {"hmr", "oxor"}:
            target = value if name == "hmr" else value[0]
            if target in records:
                raise Incorrect("Repeated classical record identifier")
            new = len(records)
            if name == "hmr":
                value = new
            else:
                _, left, right = value
                if right not in records or left != -1 and left not in records:
                    raise Unsupported("Record normalization requires a closed classical interface")
                value = [new, -1 if left == -1 else records[left], records[right]]
            records[target] = new
        elif isinstance(value, dict):
            if value["condition"] not in records:
                raise Unsupported("Record normalization requires a closed classical interface")
            value = {"condition": records[value["condition"]]}
        result.append((name, wires, value))
    return result


def verify_zero_target_rewrites(before, after, witness):
    """Certify seven-T block rewrites by exact matrices and reset-derived facts.

    The optimizer's witness only locates blocks; it cannot assert zero states.
    Reference words are proved on all eight inputs, replacements on the four
    zero-target inputs, with exact cyclotomic amplitudes. Independently propagate
    sets of possible basis values from initially unknown wires. Measurement
    resets are the only source of fixed bits; zero assertions are not resets.
    """
    import hashlib
    import json
    from collections import Counter
    from itertools import islice, product

    from qpe_lab.circuits.sim import exact_clifford_t_columns

    before, after = list(before), list(after)
    validation = verify_phase_fusion(before, before)
    limit = len(before) // 15
    witness = list(islice(iter(witness), limit + 1))
    if len(witness) > limit:
        raise Incorrect("Too many zero-target witness blocks")
    boundary = 0
    for entry in witness:
        if not isinstance(entry, (tuple, list)) or len(entry) != 5:
            raise Incorrect("Malformed zero-target witness")
        start, a, b, target, replace = entry
        if (
            any(type(v) is not int or v < 0 for v in (start, a, b, target))
            or type(replace) is not bool
            or len({a, b, target}) != 3
            or start < boundary
            or start + 15 > len(before)
        ):
            raise Incorrect("Invalid or overlapping zero-target witness block")
        boundary = start + 15

    unknown = frozenset({0, 1})
    possible, reference_words, replacement_words = {}, set(), set()

    def bits(q):
        return possible.get(q, unknown)

    def assign(q, values):
        values = frozenset(values)
        if values == unknown:
            possible.pop(q, None)
        else:
            possible[q] = values

    def certify(word, wires, replaced):
        expected_length, expected_t = (10, 4) if replaced else (15, 7)
        if len(word) != expected_length or sum(g[0] in ("t", "tdg") for g in word) != expected_t:
            raise Incorrect("Zero-target rewrite changed the bounded seven-T/four-T word shape")
        mapping = {q: i for i, q in enumerate(wires)}
        if any(q not in mapping for _, ws, _ in word for q in ws):
            raise Incorrect("Zero-target word touches a different wire")
        if any(value is not None for _, _, value in word):
            raise Incorrect("Zero-target words must be unconditional")
        normalized = tuple((name, tuple(mapping[q] for q in ws), None) for name, ws, _ in word)
        cache = replacement_words if replaced else reference_words
        if normalized in cache:
            return
        columns = range(4) if replaced else range(8)
        actual = exact_clifford_t_columns(normalized, 3, columns)
        expected = tuple(
            tuple(
                (1, 0, 0, 0) if row == (col ^ 4 if col & 3 == 3 else col) else (0, 0, 0, 0)
                for col in columns
            )
            for row in range(8)
        )
        if actual != expected:
            raise Incorrect("Zero-target word has an incorrect exact amplitude or phase")
        cache.add(normalized)

    i = j = atom = replaced_count = 0
    while i < len(before):
        if atom < len(witness) and i == witness[atom][0]:
            _, a, b, target, replace = witness[atom]
            reference = before[i : i + 15]
            certify(reference, (a, b, target), False)
            length = 10 if replace else 15
            actual = after[j : j + length]
            if replace:
                if bits(target) != frozenset({0}):
                    raise Incorrect("Replacement target is not proven zero from physical resets")
                certify(actual, (a, b, target), True)
                replaced_count += 1
            elif actual != reference:
                raise Incorrect("An unchanged seven-T block was modified")
            assign(target, {z ^ (x & y) for x, y, z in product(bits(a), bits(b), bits(target))})
            i, j, atom = i + 15, j + length, atom + 1
            continue
        gate = before[i]
        if j >= len(after) or after[j] != gate:
            raise Incorrect("Zero-target rewrite changed an operation outside its witness blocks")
        name, wires, value = gate
        if name == "hmr":
            assign(wires[0], {0})
        elif name == "h":
            assign(wires[0], unknown)
        elif name == "x":
            old = bits(wires[0])
            assign(wires[0], {1 - v for v in old} | (old if value is not None else set()))
        elif name == "cx":
            control, target = wires
            old = bits(target)
            assign(
                target,
                {x ^ y for x, y in product(bits(control), old)}
                | (old if value is not None else set()),
            )
        # Diagonal gates, classical operations and assertions cannot change a
        # proven basis value. In particular an assertion never creates a fact.
        i, j = i + 1, j + 1
    if atom != len(witness) or j != len(after):
        raise Incorrect("Zero-target witness does not cover the actual rewritten stream")
    counts = Counter(g[0] for g in after)
    t_after = counts["t"] + counts["tdg"]
    if validation["t_before"] - t_after != 3 * replaced_count:
        raise Incorrect("Zero-target rewrite count disagrees with its actual gate stream")
    return {
        "verified": True,
        "method": "exact local cyclotomic matrices + independent possible-bit-set propagation",
        "scope": "all initial quantum inputs and all measurement outcomes; no phase alignment",
        "initial_known_bits": 0,
        "zero_source": "physical HMR resets only; assertions establish no facts",
        "reference_blocks": len(witness),
        "replaced_blocks": replaced_count,
        "reference_word_variants": len(reference_words),
        "replacement_word_variants": len(replacement_words),
        "reference_columns": 8,
        "replacement_columns": 4,
        "arithmetic": "exact rational Q[omega]/(omega^4+1)",
        "before_operations": len(before),
        "after_operations": len(after),
        "t_before": validation["t_before"],
        "t_after": t_after,
        "witness_sha256": hashlib.sha256(
            json.dumps(witness, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def verify_phase_fusion(before, after, *, affine=False, complements=False):
    """Exact certificate of diagonal phase transport between basis changes.

    Keep the entire nonphase operation sequence, including every measurement
    and classical instruction. On each wire, compare phase exponents modulo
    eight within epochs delimited by operations that change its Z basis.
    Alternatively track affine basis values with sets of independent path
    variables. Each unchanged nonphase tensor has identical input/output path
    labels on both sides. Equal phase polynomials therefore give identical
    amplitudes for each path, including each fixed measurement record. This
    check does not call the optimizer or use its output words or integer masks.
    Complement mode normalizes f XOR 1 as 1-f and keeps a scalar coefficient.
    It permits a terminal pair of identical unconditional X gates only when
    its intervening/following phases contribute a pure scalar.
    """
    import hashlib
    import json
    from collections import Counter, defaultdict
    from itertools import zip_longest

    if complements and not affine:
        raise Incorrect("Complement verification requires affine phase tracking")
    exponents = {"t": 1, "tdg": -1, "s": 2, "sdg": -2, "z": 4}
    allowed = {*exponents, "h", "x", "cx", "cz", "hmr", "oxor", "assert_zero", "neg"}
    ledgers, counts = [{}, {}], [Counter(), Counter()]

    def skeleton(gates, ledger, count):
        epoch = defaultdict(int)
        outcomes = set()
        forms = {}
        position = 0

        def form(q):
            return forms.get(q, frozenset({(0, q, 0)}))

        def add_phase(key, power):
            remainder = (ledger.get(key, 0) + power) % 8
            if remainder:
                ledger[key] = remainder
            else:
                ledger.pop(key, None)

        for name, wires, value in gates:
            if name not in allowed:
                raise Incorrect("Unsupported gate in phase-fusion certificate")
            arity = 0 if name in ("oxor", "neg") else 2 if name in ("cx", "cz") else 1
            if (
                len(wires) != arity
                or len(set(wires)) != arity
                or any(type(q) is not int or q < 0 for q in wires)
            ):
                raise Incorrect("Invalid phase-fusion gate wires")
            if name == "hmr":
                if type(value) is not int or value < 0 or value in outcomes:
                    raise Incorrect("Invalid phase-fusion measurement id")
                outcomes.add(value)
            elif name == "oxor":
                if not isinstance(value, (list, tuple)) or len(value) != 3:
                    raise Incorrect("Invalid phase-fusion XOR")
                target, left, right = value
                if (
                    any(type(v) is not int for v in value)
                    or target < 0
                    or target in outcomes
                    or (left != -1 and left not in outcomes)
                    or right not in outcomes
                ):
                    raise Incorrect("Invalid phase-fusion classical dependency")
                outcomes.add(target)
            elif value is not None:
                if (
                    not isinstance(value, dict)
                    or set(value) != {"condition"}
                    or type(value["condition"]) is not int
                    or value["condition"] not in outcomes
                ):
                    raise Incorrect("Invalid phase-fusion condition")
            count[name] += 1
            if name in exponents and value is None:
                if len(wires) != 1:
                    raise Incorrect("Invalid phase-fusion gate arity")
                key = form(wires[0]) if affine else (wires[0], epoch[wires[0]])
                power = exponents[name]
                if complements and (2, 0, 0) in key:
                    add_phase(frozenset(), power)
                    key = key - {(2, 0, 0)}
                    power = -power
                if affine and not key:
                    continue  # A zero computational-basis value has no phase.
                add_phase(key, power)
                continue
            if affine:
                if name == "cx" and value is None:
                    forms[wires[1]] = form(wires[1]) ^ form(wires[0])
                elif name == "x" and value is None:
                    forms[wires[0]] = form(wires[0]) ^ {(2, 0, 0)}
                elif name == "hmr":
                    forms[wires[0]] = frozenset()
                elif name in ("h", "x", "cx", "assert_zero"):
                    for q in wires[-1:] if name == "cx" else wires:
                        forms[q] = frozenset({(1, position, q)})
            else:
                if name == "cx":
                    epoch[wires[-1]] += 1
                elif name in ("h", "x", "hmr", "assert_zero"):
                    for q in wires:
                        epoch[q] += 1
            position += 1
            yield name, tuple(wires), value

    operations, extra_x, tail_start = 0, [], None
    for left, right in zip_longest(
        skeleton(before, ledgers[0], counts[0]), skeleton(after, ledgers[1], counts[1])
    ):
        if left != right:
            if (
                complements
                and left is None
                and right[0] == "x"
                and right[2] is None
                and len(extra_x) < 2
            ):
                if not extra_x:
                    tail_start = dict(ledgers[1])
                extra_x.append(right)
                continue
            raise Incorrect("Phase fusion changed a nonphase operation or classical dependency")
        operations += 1
    scalar_repair = 0
    if extra_x:
        if len(extra_x) != 2 or extra_x[0] != extra_x[1]:
            raise Incorrect(
                "Scalar repair must restore the same wire with two unconditional X gates"
            )
        if {k: v for k, v in tail_start.items() if k} != {k: v for k, v in ledgers[1].items() if k}:
            raise Incorrect("Scalar repair changed a nonconstant phase")
        scalar_repair = (ledgers[1].get(frozenset(), 0) - tail_start.get(frozenset(), 0)) % 8
    if ledgers[0] != ledgers[1]:
        raise Incorrect(
            "Phase fusion changed an exact phase or crossed a basis/measurement barrier"
        )
    canonical = sorted((sorted(key) if affine else key, power) for key, power in ledgers[0].items())
    return {
        "verified": True,
        "method": (
            "identical core nonphase sequence + exact normalized parity phases and scalar repair"
            if complements
            else "identical nonphase sequence + exact path-affine phase polynomial modulo eight"
            if affine
            else "identical nonphase sequence + exact per-wire epoch exponents modulo eight"
        ),
        "scope": "all quantum inputs and all measurement outcomes; conditional phases preserved",
        "nonphase_operations": operations,
        "nonzero_phase_regions": len(ledgers[0]),
        "phase_ledger_sha256": hashlib.sha256(
            json.dumps(canonical, separators=(",", ":")).encode()
        ).hexdigest(),
        "t_before": counts[0]["t"] + counts[0]["tdg"],
        "t_after": counts[1]["t"] + counts[1]["tdg"],
        **(
            {
                "scalar_repair_x_count": len(extra_x),
                "scalar_repair_pi_over_4_units": scalar_repair,
            }
            if complements
            else {}
        ),
    }


"""Lookup-specific acceptance rules over the independent symbolic simulator."""


class LookupChecker(SymbolicSimulator):
    def __init__(self, task):
        super().__init__(task.address_bits, task.word_bits)
        self.task = task

    def finish(self):
        for q in self.registers[0][1]:
            if self.value(q) != self.initial[q]:
                raise Incorrect(f"Address wire {q} not restored")
        for bit, q in enumerate(self.registers[1][1]):
            coefficient = sum(
                1 << address for address, word in enumerate(self.task.words) if word >> bit & 1
            )
            expected = self.xor(self.initial[q], {0: coefficient} if coefficient else {})
            if self.value(q) != expected:
                raise Incorrect(f"Target bit {bit} is not z XOR table[address] for every input")
        self.assert_clean()

    def evidence(self):
        return {
            "method": "all-address bitsets + symbolic target ANF + symbolic measurement debts",
            "addresses": len(self.task.words),
            "target_bits_symbolic": self.task.word_bits,
            "measurement_draws_symbolic": self.draws,
            "floating_point": False,
            "scope": "full lookup channel on arbitrary coherent address and target input; clean workspace",
        }


def verify_lookup_program(raw, address_bits, word_bits, words, *, erase=False, max_qubits=8192):
    """Certify an agent-authored upstream program before embedding it in a walk."""
    from types import SimpleNamespace

    from qpe_lab.circuits.compact_reader import interpret

    class Checker(LookupChecker):
        def initialize(self, registers):
            if self.registers is not None:
                return
            super().initialize(registers)
            if erase:
                self.protected = set(registers[0][1])
                for bit, wire in enumerate(registers[1][1]):
                    mask = sum(1 << a for a, word in enumerate(words) if word >> bit & 1)
                    self.wires[wire] = {0: mask} if mask else {}
                    self.initial[wire] = self.wires[wire].copy()

    checker = Checker(SimpleNamespace(address_bits=address_bits, word_bits=word_bits, words=words))
    ops = []
    cost = interpret(raw, checker, max_qubits, capture=ops)
    evidence = checker.evidence()
    evidence["scope"] = (
        "all addresses and measurement outcomes; target starts equal to table[address] and ends zero"
        if erase
        else evidence["scope"]
    )
    return ops, checker.registers, cost, evidence
