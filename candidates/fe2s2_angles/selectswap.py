"""Alternative generator: two-word SelectSwap, with measurement-based cleanup."""

from qpe_lab import build_lookup

b, address, target = build_lookup(method="selectswap", block_size=2)
b.emit()
