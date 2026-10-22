"""Bounded upstream stream reader feeding actual operations to a symbolic checker."""

from collections import Counter
from io import BytesIO

from uncompute.evalcore.compact_reader import _CompactVM
from uncompute.ops import NO_BIT, NO_OUT, OpKind, gate_class


def interpret(path, checker, max_qubits, max_ops=2_000_000, *, capture=None):
    """Reuse the normative parser; feed actual expanded ops into our exact checker.

    _CompactVM is private upstream API, intentionally confined to this reader.
    """
    counts = Counter()

    class Sink:
        def apply(self, ops, validate=False):
            checker.initialize(vm.registers)
            for op in ops:
                kind, qa, qb, qt, ot, oc, _, ct, cc = op
                if cc != NO_BIT or ct != NO_BIT:
                    checker.unsupported("Classical bit gates are not in this task contract")
                cls = gate_class(kind, conditioned=False)
                if cls:
                    counts[cls] += 1
                name = OpKind(kind).name
                counts["ops"] += 1
                counts["measurements"] += name == "HMR"
                checker.apply(name, qa, qb, qt, ot, None if oc == NO_OUT else oc)
                if capture is not None:
                    capture.append((name, qa, qb, qt, ot, None if oc == NO_OUT else oc))

    class BoundedVM(_CompactVM):
        def _check_caps(self, op_index=None):
            super()._check_caps(op_index)
            if self.num_qubits > max_qubits or self.num_outs > max_ops or self.num_bits:
                raise ValueError("Register space exceeds fixed task limits")

        def _exec_op(self, rec):
            if self.n_expanded >= max_ops:
                raise ValueError("Expanded operation limit exceeded")
            super()._exec_op(rec)

        def _read_exact(self, n, what):
            if n > 32 * 1024 * 1024:
                raise ValueError("Single stream field exceeds size limit")
            return super()._read_exact(n, what)

    with BytesIO(path) if isinstance(path, bytes) else open(path, "rb") as f:
        if f.read(8) != b"OPSTRM01":
            raise ValueError("Invalid upstream stream magic")
        vm = BoundedVM(f, [], Sink())
        vm.run()
        vm.flush()
        checker.initialize(vm.registers)
        checker.finish()
    return {
        "status": "counted",
        "toffoli": counts["toffoli"],
        "clifford": counts["clifford"],
        "ops": counts["ops"],
        "measurements": counts["measurements"],
        "qubits": vm.num_qubits,
        "workspace_qubits": vm.num_qubits - checker.task.address_bits - checker.task.word_bits,
        "tq_cost": counts["toffoli"] * vm.num_qubits,
        "scope": "upstream logical gate model; not complete DFTHC walk",
    }
