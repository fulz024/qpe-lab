"""Compact DFTHC gate generator, with reusable measured QROAM components."""

from qpe_lab import build_walk, load_task

task = load_task()
build_walk(
    task,
    implementation="paper_strict" if "uniform_bits" in task.model["paper"] else "paper",
    outer_block_size=min(2, 1 << task.label_bits),
    qrom_block_size=min(8, 1 << task.term_bits),
    outer_unlookup_size=min(8, 1 << task.label_bits),
    unlookup_block_size=min(32, 1 << (task.label_bits + task.term_bits)),
).emit()
