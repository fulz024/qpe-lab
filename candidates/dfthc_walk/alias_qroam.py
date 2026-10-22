"""Executable alias/QROAM/phase-gradient walk; submit against a frozen alias task."""

from uncompute.gadgets.qrom import erase_register

from qpe_lab import Builder, build_walk
from qpe_lab.circuits.gadgets.qrom import lookup_xor

walk = build_walk(
    implementation="dynamic",
    prepare_method="alias",
    rotation_method="phase_gradient",
    qrom_block_size=4,
    unlookup_block_size=8,
    angle_block_size=1,
)
# These are actual upstream programs, not resource estimates. The judge proves
# their complete lookup/erase channels before composing them into the walk.
spec = walk.lookup_spec("angles")
for erase in (False, True):
    program = Builder()
    address = program.qubits(spec["address_bits"])
    target = program.qubits(spec["word_bits"])
    program.qubit_register(address)
    program.qubit_register(target)
    if erase:
        erase_register(
            program,
            target,
            address,
            len(spec["words"]),
            spec["words"],
            # The pinned upstream Builder requires a nonempty high address:
            # a whole-domain window emits CZ(None, q). DynamicCircuit handles
            # that shorthand, but a standalone upstream stream cannot.
            min(8, len(spec["words"]) // 2),
            range(len(spec["words"])),
            "angles-erase",
        )
    else:
        lookup_xor(program, address, target, spec["words"])
    walk.lookup_program("angles", program, erase=erase)
walk.emit()
