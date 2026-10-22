"""Constant-control simplification followed by certified ancillary-wire reuse."""

from qpe_lab import CircuitBuilder
from qpe_lab.circuits.gadgets.synthesis import constant_control_edits, reuse_clean_qubits

b = CircuitBuilder()
for entry in b.entries:
    b.select(entry)
    b.replace(constant_control_edits(b.gates))
    b.reuse(*reuse_clean_qubits(b.gates, b.protected_wires))
b.emit()
