"""Construction APIs: original reversible Builder and hybrid rotation circuits."""

import base64
import io
import os
from pathlib import Path

from uncompute.builder import Builder

from qpe_lab.circuits.gadgets.qrom import lookup_xor
from qpe_lab.circuits.ops import encode_hybrid, encode_walk
from qpe_lab.contracts import RotationTask, WalkTask

__all__ = [
    "Builder",
    "CircuitBuilder",
    "HybridBuilder",
    "WalkBuilder",
    "build_lookup",
    "build_rotation",
    "build_walk",
]


def _upstream_bytes(builder):
    """Serialize embedded programs through the pinned normative Python encoder.

    The upstream native fast path calls a missing _OpTape._obj_is_reg for
    side-table records. Its documented UNCOMPUTE_KERNEL switch preserves the
    format and surfaces normative operand errors without modifying upstream.
    Only this in-memory serialization disables it; the independent C checker
    and the caller's environment remain unchanged.
    """
    stream = io.BytesIO()
    previous = os.environ.get("UNCOMPUTE_KERNEL")
    os.environ["UNCOMPUTE_KERNEL"] = "0"
    try:
        builder.emit(meta=False, dest=stream)
    finally:
        if previous is None:
            del os.environ["UNCOMPUTE_KERNEL"]
        else:
            os.environ["UNCOMPUTE_KERNEL"] = previous
    return stream.getvalue()


def build_lookup(task, *, method="unary", block_size=1):
    """Build a complete lookup circuit with registered address and target wires.

    Use gadgets.qrom.lookup_xor to insert the same operation into a larger circuit.
    Builder itself is the upstream class; no gate or serialization semantics change.
    """
    b = Builder()
    address, target = b.qubits(task.address_bits), b.qubits(task.word_bits)
    b.qubit_register(address)
    b.qubit_register(target)
    lookup_xor(b, address, target, task.words, method=method, block_size=block_size)
    return b, address, target


class HybridBuilder:
    """Compose clean lookup blocks and quantum-controlled dyadic Givens gates.

    Register names are persistent: address, angle, system, and optional enable.
    Embedded lookup I/O registers map to address/angle; their other local wires
    map to a reusable workspace pool disjoint from system/enable.
    """

    def __init__(self, task):
        if not isinstance(task, RotationTask):
            raise TypeError("HybridBuilder currently requires a RotationTask")
        self.task = task
        self.steps = []

    def lookup(self, builder):
        self.steps.append(
            {"kind": "lookup", "ops_b64": base64.b64encode(_upstream_bytes(builder)).decode()}
        )

    def givens(self, p, q, angle_bit, turns, *, controlled=None):
        """G(2*pi*turns/2**beta), controlled by angle_bit and optionally enable."""
        self.steps.append(
            {
                "kind": "cgivens",
                "p": p,
                "q": q,
                "angle_bit": angle_bit,
                "turns": turns,
                "controlled": self.task.controlled if controlled is None else controlled,
            }
        )

    def artifact(self):
        return {
            "schema": "qpe.hybrid.v1",
            "address_bits": self.task.address_bits,
            "word_bits": self.task.word_bits,
            "angle_bits": self.task.angle_bits,
            "mode_count": self.task.mode_count,
            "controlled": self.task.controlled,
            "steps": self.steps,
        }

    def emit(self, dest=None):
        data = encode_hybrid(self.artifact())
        if hasattr(dest, "write"):
            dest.write(data)
        else:
            Path(dest or os.environ.get("UNCOMPUTE_OPS_PATH", "circuit.ops")).write_bytes(data)


def build_rotation(task, *, method="unary", block_size=1, unload_method=None):
    """Load the angle word, apply the fixed orbital rotation, then clear the word."""
    from qpe_lab.circuits.gadgets.givens import inverse_elimination_network

    circuit = HybridBuilder(task)
    load, _, _ = build_lookup(task, method=method, block_size=block_size)
    circuit.lookup(load)
    inverse_elimination_network(circuit, task.spatial_modes, task.spin_copies, task.angle_bits)
    unload, _, _ = build_lookup(task, method=unload_method or method, block_size=block_size)
    circuit.lookup(unload)
    return circuit


class CircuitBuilder:
    """Submit complete circuits, optionally accompanied by equivalence evidence.

    Helpers are optional: an agent may construct the same data format directly.
    Only the separate evaluator supplies an equivalence verdict or a reward.
    """

    def __init__(self, reference=None):
        from qpe_lab.circuits.ops import read_reference
        from qpe_lab.contracts import digest

        path = reference or os.environ.get("QPE_REFERENCE_PATH")
        if path is None:
            raise ValueError("A frozen training reference is required")
        data = read_reference(path)
        self.reference_sha256 = digest(path)
        self.contract = data.get("contract", {"scope": "complete controlled walk"})
        self.references = data.get("entries", {"walk": data})
        self.entries = {
            name: {
                "gates": list(entry["gates"]),
                "certificate": {
                    "schema": "qpe.gate-rewrites.v1",
                    "reference_sha256": self.reference_sha256,
                    "stages": [],
                },
            }
            for name, entry in self.references.items()
        }
        self.select(next(iter(self.entries)))

    def select(self, name):
        """Select forward/adjoint; each has its own independent implementation."""
        if name not in self.entries:
            raise ValueError("Unknown circuit entry")
        self.entry = name
        self.protected_wires = self.references[name]["protected_wires"]
        self.ancilla_initial = self.references[name].get("ancilla_initial", "unknown")
        return self

    @property
    def gates(self):
        return self.entries[self.entry]["gates"]

    @gates.setter
    def gates(self, value):
        self.entries[self.entry]["gates"] = value

    @property
    def program(self):
        return self.entries[self.entry]["certificate"]

    def submit(self, gates):
        """Replace this entry wholesale. No rewrite history or template required."""
        from qpe_lab.circuits.ops import elementary_gates

        self.gates = elementary_gates(list(gates))
        self.program["stages"] = []
        return self

    def replace(self, edits, *, proof="local"):
        """Apply non-overlapping [start, stop) replacements in current coordinates."""
        import copy

        if proof not in {"local", "dynamic", "zx", "boolean"}:
            raise ValueError("Unknown block proof backend")
        edits = copy.deepcopy(edits)
        result, end = [], 0
        for edit in edits:
            start, stop = edit["start"], edit["stop"]
            if not end <= start < stop <= len(self.gates):
                raise ValueError("Invalid replacement range")
            result.extend(self.gates[end:start])
            result.extend((name, tuple(wires), value) for name, wires, value in edit["gates"])
            end = stop
        result.extend(self.gates[end:])
        self.gates = result
        self.program["stages"].append({"kind": proof, "edits": edits})
        return self

    def reuse(self, moves, gates):
        """Propose explicit wire allocation/release events and their actual gate stream.

        Each move is [position, logical_wire, physical_wire]; -1 releases a wire.
        The evaluator independently proves every release is clean and every binding
        is exclusive. An assertion in the gate stream never establishes cleanliness.
        """
        import copy

        if self.ancilla_initial != "zero":
            raise ValueError("Qubit reuse requires a v2 reference with clean ancillary inputs")
        stage = copy.deepcopy({"kind": "reuse", "moves": moves, "gates": gates})
        self.gates = [(name, tuple(wires), value) for name, wires, value in stage["gates"]]
        stage["gates"] = self.gates
        self.program["stages"].append(stage)
        return self

    def phases(self, gates):
        """Propose any phase transport; the judge independently compares path sums."""
        self.gates = [(name, tuple(wires), value) for name, wires, value in gates]
        self.program["stages"].append({"kind": "phase", "gates": self.gates})
        return self

    def emit(self, dest=None):
        import json

        from qpe_lab.circuits.ops import CIRCUIT_MAGIC

        entries = {
            name: {
                "gates": entry["gates"],
                **({"proofs": entry["proofs"]} if entry.get("proofs") else {}),
                **(
                    {"certificate": entry["certificate"]}
                    if entry["certificate"]
                    != {
                        "schema": "qpe.gate-rewrites.v1",
                        "reference_sha256": self.reference_sha256,
                        "stages": [],
                    }
                    else {}
                ),
            }
            for name, entry in self.entries.items()
        }
        data = (
            CIRCUIT_MAGIC
            + json.dumps(
                {
                    "schema": "qpe.circuit.v1",
                    "reference_sha256": self.reference_sha256,
                    "entries": entries,
                },
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
        if hasattr(dest, "write"):
            dest.write(data)
        else:
            Path(dest or os.environ.get("UNCOMPUTE_OPS_PATH", "circuit.ops")).write_bytes(data)

    def attach_lean(self, proof, *, reference_range=None, candidate_range=None):
        """Attach a Lean proof term to this entry or a pair of half-open ranges.

        The evaluator generates the statement and semantics from actual gates.
        For small identities a term such as ``by decide`` often suffices.
        """
        from qpe_lab.circuits.ops import validate_proofs

        attachments = self.entries[self.entry].get("proofs", []) + [
            {
                "backend": "lean",
                "source": proof,
                "reference_range": reference_range,
                "candidate_range": candidate_range,
            }
        ]
        self.entries[self.entry]["proofs"] = validate_proofs(attachments)
        return self


class WalkBuilder:
    """Data-only, executable SOS walk circuit; emit remains the agent interface."""

    def __init__(self, task, artifact):
        self.task, self.program = task, artifact

    def artifact(self):
        return self.program

    def elementary_gates(self):
        from qpe_lab.circuits.gadgets.synthesis import expand_walk

        return expand_walk(self.task, self.program)[0]

    def lookup_spec(self, name):
        """Return a fixed component's full-domain words for a custom generator."""
        from qpe_lab.circuits.gadgets.prepare import alias_table
        from qpe_lab.circuits.gadgets.select import descriptor_tables

        if self.program.get("implementation", {}).get("kind") != "dynamic":
            raise ValueError("Custom components require a dynamic walk")
        if name in ("descriptor", "angles"):
            descriptors, angles, bits, width = descriptor_tables(self.task, self.program["select"])
            words, word_bits = (descriptors, bits + 5) if name == "descriptor" else (angles, width)
        elif name in ("outer", "inner") and "alias" in self.program:
            bits = self.task.model["preparation"]["alias_bits"]
            index_bits = self.task.label_bits if name == "outer" else self.task.term_bits
            tables = (
                [self.program["alias"]["outer"]]
                if name == "outer"
                else list(self.program["alias"]["inner"])
            )
            if name == "inner":
                dummy = alias_table(
                    [1 << (bits + index_bits)] + [0] * ((1 << index_bits) - 1), bits
                )
                tables += [dummy] * ((1 << self.task.label_bits) - len(tables))
            words = [
                table["alt"][term] | (table["keep"][term] << index_bits)
                for term in range(1 << index_bits)
                for table in tables
            ]
            word_bits = index_bits + bits + 1
        else:
            raise ValueError("Unknown or inactive lookup component")
        return {
            "address_bits": (len(words) - 1).bit_length(),
            "word_bits": word_bits,
            "words": words,
        }

    def lookup_program(self, name, builder, *, erase=False):
        """Embed an uncompute-compatible program; the judge proves its channel.

        The builder must register address and target. A load works for arbitrary
        target input; an erase starts at target=table[address] and must clear it.
        """
        self.lookup_spec(name)
        self.program.setdefault("lookup_programs", {}).setdefault(name, {})[
            "erase" if erase else "load"
        ] = base64.b64encode(_upstream_bytes(builder)).decode()
        return self

    def emit(self, dest=None):
        data = encode_walk(self.program)
        if hasattr(dest, "write"):
            dest.write(data)
        else:
            Path(dest or os.environ.get("UNCOMPUTE_OPS_PATH", "circuit.ops")).write_bytes(data)


def build_walk(
    task,
    *,
    grouped=True,
    bitwise_rotations=False,
    outer_order=None,
    inner_order=None,
    select_method="grouped",
    qrom_block_size=1,
    implementation="unitary",
    prepare_method="tree",
    rotation_method="phase_gradient",
    unlookup_block_size=1,
    angle_block_size=1,
    outer_block_size=1,
    outer_unlookup_size=1,
    specialize_constants=False,
    gradient_adder="general",
    phase_unlookup_bits=None,
    outer_phase_unlookup_bits=None,
    sf_lookup_control="explicit",
    fuse_phases=False,
    specialize_clean_targets=False,
):
    """Build PREPARE/SELECT, rectangular encoding U, and controlled SOS walk.

    grouped shares basis changes across terms using the same orbital/spin.
    bitwise_rotations exposes a deliberately simple dyadic rotation baseline.
    """
    from qpe_lab.circuits.gadgets.prepare import alias_table, prepare_weights, walk_distributions
    from qpe_lab.circuits.gadgets.select import orbital_network, select_groups

    if not isinstance(task, WalkTask):
        raise TypeError("build_walk requires a WalkTask")
    if type(specialize_clean_targets) is not bool or (
        specialize_clean_targets
        and (implementation != "paper_strict" or task.contract["cost_model"] != "clifford_t")
    ):
        raise ValueError("Zero-target specialization requires a strict Clifford+T walk")
    # Preserve both older modes; "parity" also combines complementary paths.
    if (type(fuse_phases) is not bool and fuse_phases not in ("affine", "parity")) or (
        fuse_phases
        and (implementation != "paper_strict" or task.contract["cost_model"] != "clifford_t")
    ):
        raise ValueError("Phase fusion requires a strict Clifford+T walk")
    if type(specialize_constants) is not bool or (
        specialize_constants and implementation != "paper_strict"
    ):
        raise ValueError("Constant specialization requires paper_strict")
    if gradient_adder not in ("general", "phase") or (
        gradient_adder != "general" and implementation != "paper_strict"
    ):
        raise ValueError("Phase-gradient adder requires paper_strict")
    if sf_lookup_control not in ("explicit", "zero_address") or (
        sf_lookup_control != "explicit" and implementation != "paper_strict"
    ):
        raise ValueError("Zero-address SF lookup requires paper_strict")
    for plane_bits in (phase_unlookup_bits, outer_phase_unlookup_bits):
        if plane_bits is not None and (
            type(plane_bits) is not int
            or not 0 <= plane_bits <= 2
            or implementation != "paper_strict"
        ):
            raise ValueError(
                "Boolean-plane unlookup requires paper_strict and zero to two plane bits"
            )
    if implementation in ("paper", "paper_strict"):
        if "paper" not in task.model:
            raise ValueError("Paper circuit requires a separately frozen compact target")
        prep = task.model["paper"]
        if (implementation == "paper_strict") != ("uniform_bits" in prep):
            raise ValueError("Implementation and frozen uniform projector differ")
        return WalkBuilder(
            task,
            {
                "schema": "qpe.walk-circuit.v3",
                "model_sha256": task.table_hash,
                "label_bits": task.label_bits,
                "term_bits": task.term_bits,
                "mode_count": task.mode_count,
                "controlled": task.controlled,
                "implementation": {
                    "kind": implementation,
                    "outer_block": outer_block_size,
                    "inner_block": qrom_block_size,
                    "outer_unlookup": outer_unlookup_size,
                    "inner_unlookup": unlookup_block_size,
                    **({"specialize_constants": True} if specialize_constants else {}),
                    **({"fuse_phases": fuse_phases} if fuse_phases else {}),
                    **({"specialize_clean_targets": True} if specialize_clean_targets else {}),
                    **({"gradient_adder": gradient_adder} if gradient_adder != "general" else {}),
                    **(
                        {"sf_lookup_control": sf_lookup_control}
                        if sf_lookup_control != "explicit"
                        else {}
                    ),
                    **(
                        {"phase_unlookup_bits": phase_unlookup_bits}
                        if phase_unlookup_bits is not None
                        else {}
                    ),
                    **(
                        {"outer_phase_unlookup_bits": outer_phase_unlookup_bits}
                        if outer_phase_unlookup_bits is not None
                        else {}
                    ),
                },
                "alias": {
                    "outer": alias_table(prep["outer"], prep["alias_bits"], compact=True),
                    "inner": [
                        alias_table(row, prep["alias_bits"], compact=True) for row in prep["inner"]
                    ],
                },
                "orbital_angles": [list(row) for row in task.model["orbital_angles"]],
            },
        )
    if "paper" in task.model:
        raise ValueError("Compact task requires the paper implementation")
    outer_weights, inner_weights = walk_distributions(task.model)
    outer = prepare_weights(outer_weights, task.label_bits, order=outer_order)
    inner = []
    for label, weights in enumerate(inner_weights):
        inner.extend(
            prepare_weights(
                weights,
                task.term_bits,
                label=label,
                order=inner_order,
            )
        )
    program = {
        "schema": "qpe.walk-circuit.v1",
        "model_sha256": task.table_hash,
        "label_bits": task.label_bits,
        "term_bits": task.term_bits,
        "mode_count": task.mode_count,
        "controlled": task.controlled,
        "outer": outer,
        "inner": inner,
        "orbitals": [
            orbital_network(row, task.model["angle_bits"], bitwise=bitwise_rotations)
            for row in task.model["orbital_angles"]
        ],
        "select": select_groups(task.model, grouped=grouped),
        "steps": [
            {"block": block, "adjoint": inverse}
            for block, inverse in (
                ("outer", False),
                ("inner", False),
                ("select", False),
                ("inner", True),
                ("rout", False),
                ("inner", False),
                ("select", True),
                ("inner", True),
                ("outer", True),
                ("rin", False),
            )
        ],
    }
    if select_method != "grouped":
        if select_method not in ("qrom", "qrom_direct"):
            raise ValueError("Unknown SELECT implementation")
        program["select_implementation"] = {
            "kind": select_method,
            "block_size": qrom_block_size,
        }
    if implementation == "dynamic":
        if select_method != "grouped":
            raise ValueError("Dynamic SELECT uses its own descriptor/angle lookups")
        program["schema"] = "qpe.walk-circuit.v2"
        program["implementation"] = {
            "kind": "dynamic",
            "prepare": prepare_method,
            "rotations": rotation_method,
            "lookup_block_size": qrom_block_size,
            "unlookup_block_size": unlookup_block_size,
            "angle_block_size": angle_block_size,
        }
        if prepare_method == "alias":
            if "preparation" not in task.model:
                raise ValueError("Alias preparation requires a separately frozen alias target")
            bits = task.model["preparation"]["alias_bits"]
            program["alias"] = {
                "outer": alias_table(outer_weights, bits),
                "inner": [alias_table(row, bits) for row in inner_weights],
            }
            program["outer"], program["inner"] = [], []
    elif implementation != "unitary" or prepare_method != "tree":
        raise ValueError("Unsupported walk implementation/preparation combination")
    return WalkBuilder(task, program)
