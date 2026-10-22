"""Load, apply the fixed controlled Givens network, and unload using unary QROM."""

from qpe_lab import build_rotation

build_rotation(method="unary").emit()
