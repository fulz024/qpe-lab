"""Strict walk with constant controls, a phase adder and joint Boolean repair."""

from qpe_lab import build_walk, load_task

task = load_task()
build_walk(
    task,
    implementation="paper_strict",
    outer_block_size=min(2, 1 << task.label_bits),
    qrom_block_size=min(8, 1 << task.term_bits),
    outer_unlookup_size=min(8, 1 << task.label_bits),
    unlookup_block_size=min(32, 1 << (task.label_bits + task.term_bits)),
    specialize_constants=True,
    gradient_adder="phase",
    phase_unlookup_bits=2,
    outer_phase_unlookup_bits=2,
    sf_lookup_control="zero_address",
    fuse_phases="parity",
    specialize_clean_targets=True,
).emit()
