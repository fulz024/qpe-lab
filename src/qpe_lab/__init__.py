"""Circuit construction and simulation extensions alongside pinned uncompute."""

from qpe_lab.circuits import builder as _builder
from qpe_lab.circuits.builder import (
    Builder,
    CircuitBuilder,
    HybridBuilder,
    WalkBuilder,
)
from qpe_lab.circuits.sim import (
    DenseFockSimulator,
    GaussianSimulator,
    Simulator,
    SOSSimulator,
    StatevectorSimulator,
    SymbolicSimulator,
)

__all__ = [
    "Builder",
    "CircuitBuilder",
    "DenseFockSimulator",
    "GaussianSimulator",
    "HybridBuilder",
    "SOSSimulator",
    "Simulator",
    "StatevectorSimulator",
    "SymbolicSimulator",
    "WalkBuilder",
    "build_lookup",
    "build_rotation",
    "build_walk",
    "load_task",
]

__version__ = "0.1.0"


def load_task():
    """Read the task injected by the evaluator for a Python generator."""
    import os

    from qpe_lab.tasks.task import read_task

    return read_task(os.environ["QPE_TASK_PATH"])


def build_lookup(task=None, **options):
    """Construct a lookup using an explicit task or the injected task."""
    return _builder.build_lookup(load_task() if task is None else task, **options)


def build_rotation(task=None, **options):
    """Construct an orbital rotation using an explicit or injected task."""
    return _builder.build_rotation(load_task() if task is None else task, **options)


def build_walk(task=None, **options):
    """Construct a walk using an explicit or injected task."""
    return _builder.build_walk(load_task() if task is None else task, **options)
