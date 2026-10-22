from qpe_lab import CircuitBuilder
from qpe_lab.circuits.gadgets.synthesis import constant_control_edits

b = CircuitBuilder()
for entry in b.entries:
    b.select(entry)
    b.replace(constant_control_edits(b.gates))
b.emit()
