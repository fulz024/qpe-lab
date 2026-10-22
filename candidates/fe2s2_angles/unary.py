"""Baseline generator: content-pruned unary iteration over the fixed angle table."""

from qpe_lab import build_lookup

b, address, target = build_lookup(method="unary")
b.emit()
