"""Data-only hybrid circuit format: embedded OPSTRM01 plus dyadic Givens gates.

The original gate format remains untouched. Hybrid circuits use a distinct magic
and explicit quantum angle-bit controls. No code, paths, or claimed costs are read.
"""

import json
from pathlib import Path

HYBRID_MAGIC = b"QPECIR01"
WALK_MAGIC = b"QPEWLK01"
CIRCUIT_MAGIC = b"QPEGEN01"
MAX_ARTIFACT_BYTES = 64 * 1024**2
MAX_STEPS = 20_000
MAX_ELEMENTARY_GATES = 2_000_000
BOOLEAN_BLOCK_GATES = 131_072

# Shared by the parser and task prompt. Values describe the emitted gate record,
# not a separate learner-facing dialect.
GATE_SCHEMA = {
    "h": (1, "Hadamard"),
    "x": (1, "Pauli X"),
    "z": (1, "Pauli Z"),
    "s": (1, "diag(1,i)"),
    "sdg": (1, "diag(1,-i)"),
    "t": (1, "diag(1,exp(i*pi/4))"),
    "tdg": (1, "diag(1,exp(-i*pi/4))"),
    "cx": (2, "controlled X; wires=(control,target)"),
    "cz": (2, "controlled Z"),
    "neg": (0, "global minus sign"),
    "hmr": (1, "X-basis measurement and reset to zero; value=fresh outcome id"),
    "oxor": (0, "classical XOR; value=[new_id,left_id,right_id]; left_id=-1 means zero"),
    "assert_zero": (1, "assert zero, without resetting or changing the state"),
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate artifact field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def elementary_gates(records):
    """Validate data-only elementary operations; no implicit macros or costs."""
    if not isinstance(records, list) or len(records) > MAX_ELEMENTARY_GATES:
        raise ValueError("Invalid elementary gate count")
    result = []
    for record in records:
        if not isinstance(record, (list, tuple)) or len(record) != 3:
            raise ValueError("Malformed elementary operation")
        name, wires, value = record
        if not isinstance(name, str) or name not in GATE_SCHEMA:
            raise ValueError("Unsupported elementary operation")
        arity = GATE_SCHEMA[name][0]
        if (
            not isinstance(wires, (list, tuple))
            or len(wires) != arity
            or any(type(q) is not int or not 0 <= q < 8192 for q in wires)
            or len(set(wires)) != arity
        ):
            raise ValueError("Invalid elementary wires")
        if name == "hmr":
            if type(value) is not int or not 0 <= value < MAX_ELEMENTARY_GATES:
                raise ValueError("Invalid measurement identifier")
        elif name == "oxor":
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 3
                or any(type(v) is not int or not -1 <= v < MAX_ELEMENTARY_GATES for v in value)
                or value[0] < 0
                or value[2] < 0
            ):
                raise ValueError("Invalid classical XOR")
            value = list(value)
        elif value is not None and (
            name == "assert_zero"
            or not isinstance(value, dict)
            or set(value) != {"condition"}
            or type(value["condition"]) is not int
            or not 0 <= value["condition"] < MAX_ELEMENTARY_GATES
        ):
            raise ValueError("Invalid classical condition")
        result.append((name, tuple(wires), value))
    return result


def read_reference(path):
    """Read a maintainer-pinned concrete gate stream, never recompile its source."""
    path = Path(path)
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Reference exceeds size cap")
    doc = json.loads(
        path.read_bytes(), object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    if isinstance(doc, dict) and doc.get("schema") == "qpe.core-reference.v1":
        if (
            set(doc) != {"schema", "contract", "entries"}
            or not isinstance(doc["entries"], dict)
            or not isinstance(doc["contract"], dict)
            or set(doc["entries"]) != {"forward", "adjoint"}
        ):
            raise ValueError("Invalid core reference")
        for entry in doc["entries"].values():
            validate_reference(entry)
        return doc
    return validate_reference(doc)


def validate_reference(doc):
    if (
        not isinstance(doc, dict)
        or doc.get("schema") != "qpe.gate-reference.v2"
        or set(doc) != {"schema", "protected_wires", "gates", "ancilla_initial"}
        or doc["ancilla_initial"] != "zero"
        or not isinstance(doc["protected_wires"], list)
        or not doc["protected_wires"]
        or any(type(q) is not int or not 0 <= q < 8192 for q in doc["protected_wires"])
        or len(set(doc["protected_wires"])) != len(doc["protected_wires"])
    ):
        raise ValueError("Invalid frozen gate reference")
    doc["gates"] = elementary_gates(doc["gates"])
    return doc


def read_circuit(path):
    """A complete candidate is mandatory; optional certificates are only hints."""
    path = Path(path)
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Circuit artifact exceeds size cap")
    raw = path.read_bytes()
    if not raw.startswith(CIRCUIT_MAGIC):
        raise ValueError("Expected QPEGEN01 complete circuit artifact")
    doc = json.loads(
        raw[len(CIRCUIT_MAGIC) :], object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    if (
        not isinstance(doc, dict)
        or set(doc) != {"schema", "reference_sha256", "entries"}
        or doc["schema"] != "qpe.circuit.v1"
        or not isinstance(doc["reference_sha256"], str)
        or len(doc["reference_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in doc["reference_sha256"])
        or not isinstance(doc["entries"], dict)
        or set(doc["entries"]) not in ({"walk"}, {"forward", "adjoint"})
    ):
        raise ValueError("Invalid complete circuit header")
    total = 0
    for entry in doc["entries"].values():
        if (
            not isinstance(entry, dict)
            or "gates" not in entry
            or set(entry) - {"gates", "certificate", "proofs"}
        ):
            raise ValueError("Invalid circuit entry")
        entry["gates"] = elementary_gates(entry["gates"])
        total += len(entry["gates"])
        if "certificate" in entry:
            validate_rewrite_proof(entry["certificate"])
        if "proofs" in entry:
            validate_proofs(entry["proofs"])
    if total > MAX_ELEMENTARY_GATES:
        raise ValueError("Combined candidate exceeds operation limit")
    return doc


def validate_proofs(proofs):
    """Bounded Lean proof terms with optional region coordinates; no assumptions."""
    if not isinstance(proofs, list) or len(proofs) > 128:
        raise ValueError("Invalid proof attachments")
    seen = set()
    for proof in proofs:
        if (
            not isinstance(proof, dict)
            or set(proof) != {"backend", "source", "reference_range", "candidate_range"}
            or proof["backend"] != "lean"
            or not isinstance(proof["source"], str)
            or len(proof["source"].encode()) > 65536
        ):
            raise ValueError("Invalid Lean proof attachment")
        ranges = (proof["reference_range"], proof["candidate_range"])
        if ranges != (None, None):
            for span in ranges:
                if (
                    not isinstance(span, (list, tuple))
                    or len(span) != 2
                    or any(type(i) is not int for i in span)
                    or not 0 <= span[0] <= span[1] <= MAX_ELEMENTARY_GATES
                ):
                    raise ValueError("Invalid proof region")
        key = repr(ranges)
        if key in seen:
            raise ValueError("Duplicate proof region")
        seen.add(key)
    return proofs


def validate_rewrite_proof(doc):
    """Validate optional data-only evidence, not a separate submission format."""
    if (
        not isinstance(doc, dict)
        or set(doc) != {"schema", "reference_sha256", "stages"}
        or doc["schema"] != "qpe.gate-rewrites.v1"
        or not isinstance(doc["reference_sha256"], str)
        or len(doc["reference_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in doc["reference_sha256"])
        or not isinstance(doc["stages"], list)
        or len(doc["stages"]) > 32
    ):
        raise ValueError("Invalid gate rewrite header")
    total_edits = 0
    for stage in doc["stages"]:
        if not isinstance(stage, dict):
            raise ValueError("Malformed rewrite stage")  # noqa: TRY004 -- serialized data validation
        if stage.get("kind") in {"local", "dynamic", "zx", "boolean"}:
            if set(stage) != {"kind", "edits"} or not isinstance(stage["edits"], list):
                raise ValueError("Invalid local rewrite stage")
            total_edits += len(stage["edits"])
            if total_edits > 8192:
                raise ValueError("Too many rewrite edits")
            bound = (
                BOOLEAN_BLOCK_GATES
                if stage["kind"] == "boolean"
                else 4096
                if stage["kind"] == "zx"
                else 128
            )
            end = 0
            for edit in stage["edits"]:
                if (
                    not isinstance(edit, dict)
                    or set(edit) != {"start", "stop", "gates"}
                    or any(type(edit[k]) is not int for k in ("start", "stop"))
                    or not end <= edit["start"] < edit["stop"] <= MAX_ELEMENTARY_GATES
                    or edit["stop"] - edit["start"] > bound
                ):
                    raise ValueError("Invalid or overlapping local rewrite range")
                edit["gates"] = elementary_gates(edit["gates"])
                if len(edit["gates"]) > bound:
                    raise ValueError("Replacement exceeds local gate bound")
                end = edit["stop"]
        elif stage.get("kind") == "phase":
            if set(stage) != {"kind", "gates"}:
                raise ValueError("Invalid phase rewrite stage")
            stage["gates"] = elementary_gates(stage["gates"])
        elif stage.get("kind") == "reuse":
            if set(stage) != {"kind", "moves", "gates"} or not isinstance(stage["moves"], list):
                raise ValueError("Invalid qubit-reuse stage")
            if len(stage["moves"]) > MAX_ELEMENTARY_GATES:
                raise ValueError("Too many qubit-lifetime events")
            previous = 0
            for move in stage["moves"]:
                if (
                    not isinstance(move, list)
                    or len(move) != 3
                    or any(type(v) is not int for v in move)
                    or not previous <= move[0] <= MAX_ELEMENTARY_GATES
                    or not 0 <= move[1] < 8192
                    or not -1 <= move[2] < 8192
                ):
                    raise ValueError("Invalid qubit-lifetime event")
                previous = move[0]
            stage["gates"] = elementary_gates(stage["gates"])
        else:
            raise ValueError("Unknown rewrite proof kind")
    return doc


def encode_hybrid(circuit):
    return HYBRID_MAGIC + json.dumps(circuit, separators=(",", ":"), allow_nan=False).encode()


def read_hybrid(path):
    path = Path(path)
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Hybrid artifact exceeds size limit")
    raw = path.read_bytes()
    if not raw.startswith(HYBRID_MAGIC):
        raise ValueError("Expected QPECIR01 hybrid circuit")
    doc = json.loads(raw[len(HYBRID_MAGIC) :], object_pairs_hook=_unique_object)
    fields = {
        "schema",
        "address_bits",
        "word_bits",
        "angle_bits",
        "mode_count",
        "controlled",
        "steps",
    }
    if not isinstance(doc, dict) or set(doc) != fields or doc["schema"] != "qpe.hybrid.v1":
        raise ValueError("Invalid hybrid circuit header")
    for key in ("address_bits", "word_bits", "angle_bits", "mode_count"):
        if type(doc[key]) is not int or not 1 <= doc[key] <= 4096:
            raise ValueError(f"Invalid hybrid dimension: {key}")
    if type(doc["controlled"]) is not bool:
        raise ValueError("Invalid control declaration")
    steps = doc["steps"]
    if not isinstance(steps, list) or not 3 <= len(steps) <= MAX_STEPS:
        raise ValueError("Invalid hybrid step count")
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Malformed circuit step")  # noqa: TRY004 -- serialized data validation
        if step.get("kind") == "lookup":
            if set(step) != {"kind", "ops_b64"} or not isinstance(step["ops_b64"], str):
                raise ValueError("Invalid embedded lookup")
        elif step.get("kind") == "cgivens":
            if set(step) != {"kind", "p", "q", "angle_bit", "turns", "controlled"}:
                raise ValueError("Invalid controlled Givens fields")
            for key in ("p", "q", "angle_bit", "turns"):
                if type(step[key]) is not int or abs(step[key]) > 2**40:
                    raise ValueError(f"Invalid Givens integer: {key}")
            if not 0 <= step["p"] < doc["mode_count"] or not 0 <= step["q"] < doc["mode_count"]:
                raise ValueError("Givens mode outside system register")
            if step["p"] == step["q"] or not 0 <= step["angle_bit"] < doc["word_bits"]:
                raise ValueError("Invalid Givens pair or angle control")
            if type(step["controlled"]) is not bool:
                raise ValueError("Invalid Givens enable control")
        else:
            raise ValueError(f"Unsupported hybrid operation: {step.get('kind')}")
    return doc


def encode_walk(circuit):
    return WALK_MAGIC + json.dumps(circuit, separators=(",", ":"), allow_nan=False).encode()


def read_walk(path):
    """Bounded data-only parser. Semantic dimensions are checked against the task."""
    path = Path(path)
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Walk artifact exceeds size limit")
    raw = path.read_bytes()
    if not raw.startswith(WALK_MAGIC):
        raise ValueError("Expected QPEWLK01 walk circuit")
    doc = json.loads(raw[len(WALK_MAGIC) :], object_pairs_hook=_unique_object)
    if isinstance(doc, dict) and doc.get("schema") == "qpe.walk-circuit.v3":
        fields = {
            "schema",
            "model_sha256",
            "label_bits",
            "term_bits",
            "mode_count",
            "controlled",
            "implementation",
            "alias",
            "orbital_angles",
        }
        if set(doc) != fields or not isinstance(doc["model_sha256"], str):
            raise ValueError("Invalid compact paper walk header")
        for key in ("label_bits", "term_bits", "mode_count"):
            if type(doc[key]) is not int or not 1 <= doc[key] <= 64:
                raise ValueError("Invalid paper walk dimension")
        if type(doc["controlled"]) is not bool:
            raise ValueError("Invalid paper control")
        impl = doc["implementation"]
        if (
            not isinstance(impl, dict)
            or set(impl)
            - {
                "specialize_constants",
                "gradient_adder",
                "phase_unlookup_bits",
                "outer_phase_unlookup_bits",
                "sf_lookup_control",
                "fuse_phases",
                "specialize_clean_targets",
            }
            != {"kind", "outer_block", "inner_block", "outer_unlookup", "inner_unlookup"}
            or impl["kind"] not in ("paper", "paper_strict")
            or (
                "specialize_clean_targets" in impl
                and (
                    type(impl["specialize_clean_targets"]) is not bool
                    or impl["kind"] != "paper_strict"
                )
            )
            or (
                "fuse_phases" in impl
                and (
                    (
                        type(impl["fuse_phases"]) is not bool
                        and impl["fuse_phases"] not in ("affine", "parity")
                    )
                    or impl["kind"] != "paper_strict"
                )
            )
            or (
                "specialize_constants" in impl
                and (
                    type(impl["specialize_constants"]) is not bool or impl["kind"] != "paper_strict"
                )
            )
            or (
                "gradient_adder" in impl
                and (
                    impl["gradient_adder"] not in ("general", "phase")
                    or impl["kind"] != "paper_strict"
                )
            )
            or any(
                key in impl
                and (
                    type(impl[key]) is not int
                    or not 0 <= impl[key] <= 2
                    or impl["kind"] != "paper_strict"
                )
                for key in ("phase_unlookup_bits", "outer_phase_unlookup_bits")
            )
            or (
                "sf_lookup_control" in impl
                and (
                    impl["sf_lookup_control"] not in ("explicit", "zero_address")
                    or impl["kind"] != "paper_strict"
                )
            )
            or any(
                type(impl[k]) is not int or not 1 <= impl[k] <= 64 or impl[k] & (impl[k] - 1)
                for k in impl
                if k
                not in (
                    "kind",
                    "specialize_constants",
                    "gradient_adder",
                    "phase_unlookup_bits",
                    "outer_phase_unlookup_bits",
                    "sf_lookup_control",
                    "fuse_phases",
                    "specialize_clean_targets",
                )
            )
        ):
            raise ValueError("Invalid compact QROAM implementation")
        tables = doc["alias"]
        if (
            not isinstance(tables, dict)
            or set(tables) != {"outer", "inner"}
            or not isinstance(tables["inner"], list)
            or len(tables["inner"]) > 512
        ):
            raise ValueError("Invalid compact alias tables")
        from qpe_lab.circuits.gadgets.prepare import alias_counts

        for table in [tables["outer"], *tables["inner"]]:
            alias_counts(
                table, 24, compact=True
            )  # bounded structural check; judge uses task precision
        angles = doc["orbital_angles"]
        if (
            not isinstance(angles, list)
            or len(angles) > 1024
            or any(
                not isinstance(row, list)
                or len(row) > 31
                or any(type(x) is not int or not 0 <= x < 1 << 24 for x in row)
                for row in angles
            )
        ):
            raise ValueError("Invalid compact angle table")
        return doc
    fields = {
        "schema",
        "model_sha256",
        "label_bits",
        "term_bits",
        "mode_count",
        "controlled",
        "outer",
        "inner",
        "orbitals",
        "select",
        "steps",
    }
    optional = {"select_implementation", "implementation", "alias", "lookup_programs"}
    if (
        not isinstance(doc, dict)
        or set(doc) - optional != fields
        or doc["schema"] not in ("qpe.walk-circuit.v1", "qpe.walk-circuit.v2")
    ):
        raise ValueError("Invalid walk header")
    if doc["schema"] == "qpe.walk-circuit.v2":
        impl = doc.get("implementation")
        if (
            not isinstance(impl, dict)
            or set(impl)
            != {
                "kind",
                "prepare",
                "rotations",
                "lookup_block_size",
                "unlookup_block_size",
                "angle_block_size",
            }
            or impl["kind"] != "dynamic"
            or impl["prepare"] not in ("tree", "alias")
            or impl["rotations"] not in ("phase_gradient", "bitwise")
            or "select_implementation" in doc
            or any(
                type(impl[k]) is not int or not 1 <= impl[k] <= 16 or impl[k] & (impl[k] - 1)
                for k in ("lookup_block_size", "unlookup_block_size", "angle_block_size")
            )
        ):
            raise ValueError("Invalid dynamic implementation")
        if impl["prepare"] == "alias":
            tables = doc.get("alias")
            if (
                not isinstance(tables, dict)
                or set(tables) != {"outer", "inner"}
                or not isinstance(tables["inner"], list)
                or len(tables["inner"]) > 512
                or doc["outer"]
                or doc["inner"]
            ):
                raise ValueError("Invalid alias preparation structure")
        elif "alias" in doc:
            raise ValueError("Unexpected alias tables")
        programs = doc.get("lookup_programs", {})
        allowed = {"descriptor", "angles"} | (
            {"outer", "inner"} if impl["prepare"] == "alias" else set()
        )
        if not isinstance(programs, dict) or set(programs) - allowed:
            raise ValueError("Invalid custom component slots")
        for program in programs.values():
            if (
                not isinstance(program, dict)
                or not program
                or set(program) - {"load", "erase"}
                or any(
                    not isinstance(raw, str) or len(raw) > 44 * 1024**2 for raw in program.values()
                )
            ):
                raise ValueError("Invalid embedded component")
    elif "implementation" in doc or "alias" in doc or "lookup_programs" in doc:
        raise ValueError("Dynamic operations require the v2 artifact schema")
    if "select_implementation" in doc:
        impl = doc["select_implementation"]
        if (
            not isinstance(impl, dict)
            or set(impl) != {"kind", "block_size"}
            or impl["kind"] not in ("qrom", "qrom_direct")
            or type(impl["block_size"]) is not int
            or not 1 <= impl["block_size"] <= 16
            or impl["block_size"] & (impl["block_size"] - 1)
        ):
            raise ValueError("Invalid SELECT implementation")
    for key in ("label_bits", "term_bits", "mode_count"):
        if type(doc[key]) is not int or not 1 <= doc[key] <= 64:
            raise ValueError("Invalid walk dimension")
    if type(doc["controlled"]) is not bool or not isinstance(doc["model_sha256"], str):
        raise ValueError("Invalid walk control or model hash")
    for name in ("outer", "inner"):
        if not isinstance(doc[name], list) or len(doc[name]) > 65536:
            raise ValueError("Preparation gate limit exceeded")
        for gate in doc[name]:
            if not isinstance(gate, dict) or set(gate) - {"controls"} != {
                "target",
                "prefix",
                "left",
                "right",
                "label",
            }:
                raise ValueError("Invalid preparation gate")
            if "controls" in gate:
                controls = gate["controls"]
                if (
                    not isinstance(controls, list)
                    or len(controls) > 12
                    or any(
                        not isinstance(c, list)
                        or len(c) != 2
                        or type(c[0]) is not int
                        or not 0 <= c[0] <= 12
                        or type(c[1]) is not int
                        or c[1] not in (0, 1)
                        for c in controls
                    )
                ):
                    raise ValueError("Invalid preparation controls")
            for key in ("target", "prefix", "left", "right"):
                if type(gate[key]) is not int or not 0 <= gate[key] < 2**120:
                    raise ValueError("Invalid preparation integer")
            if not gate["right"] or (gate["label"] is not None and type(gate["label"]) is not int):
                raise ValueError("Invalid preparation ratio or label")
    if not isinstance(doc["orbitals"], list) or len(doc["orbitals"]) > 1024:
        raise ValueError("Orbital table limit exceeded")
    for network in doc["orbitals"]:
        if not isinstance(network, list) or len(network) > 4096:
            raise ValueError("Orbital network limit exceeded")
        for gate in network:
            if (
                not isinstance(gate, list)
                or len(gate) != 3
                or any(type(x) is not int or abs(x) > 2**40 for x in gate)
            ):
                raise ValueError("Invalid dyadic Givens gate")
    if not isinstance(doc["select"], list) or len(doc["select"]) > 65536:
        raise ValueError("SELECT group limit exceeded")
    branches = 0
    for group in doc["select"]:
        if not isinstance(group, dict) or set(group) != {"orbital", "spin", "branches"}:
            raise ValueError("Invalid SELECT group")
        if (
            type(group["orbital"]) is not int
            or type(group["spin"]) is not int
            or not isinstance(group["branches"], list)
        ):
            raise ValueError("Invalid SELECT selector")
        branches += len(group["branches"])
        for branch in group["branches"]:
            if not isinstance(branch, dict) or set(branch) != {"label", "term", "pauli", "phase"}:
                raise ValueError("Invalid SELECT branch")
            if any(type(branch[k]) is not int for k in ("label", "term", "phase")) or branch[
                "pauli"
            ] not in ("I", "X", "Y", "Z"):
                raise ValueError("Invalid SELECT branch values")
    if branches > 65536:
        raise ValueError("SELECT branch limit exceeded")
    if not isinstance(doc["steps"], list) or len(doc["steps"]) != 10:
        raise ValueError("Invalid walk call sequence")
    for step in doc["steps"]:
        if (
            not isinstance(step, dict)
            or set(step) != {"block", "adjoint"}
            or type(step["adjoint"]) is not bool
            or step["block"] not in ("outer", "inner", "select", "rin", "rout")
        ):
            raise ValueError("Invalid walk call")
    return doc
