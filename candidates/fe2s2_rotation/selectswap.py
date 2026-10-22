"""Load and unload the fixed rotation consumer using select-swap QROM, block 2."""

from qpe_lab import build_rotation

build_rotation(method="selectswap", block_size=2).emit()
