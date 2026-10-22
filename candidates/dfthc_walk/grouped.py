"""Share orbital basis changes across SELECT branches with equal orbital/spin."""

from qpe_lab import build_walk

build_walk(grouped=True).emit()
