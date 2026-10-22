"""Load and validate fixed lookup, rotation and DFTHC task inputs."""

import json
import math
from dataclasses import asdict
from pathlib import Path

from qpe_lab.contracts import LookupTask, RotationTask, WalkTask, digest


def load_lookup(path, *, _schema="qpe.lookup.v1"):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if spec.get("schema") != _schema:
        raise ValueError(f"Expected {_schema} task")
    table = (path.parent / spec["table"]).resolve()
    if not table.is_relative_to(path.parent):
        raise ValueError("Table must be inside task directory")
    if digest(table) != spec["table_sha256"]:
        raise ValueError("Fixed table hash mismatch")
    data = json.loads(table.read_text())
    a, w = spec["address_bits"], spec["word_bits"]
    if type(a) is not int or not 1 <= a <= 12 or type(w) is not int or not 1 <= w <= 4096:
        raise ValueError("Unsupported lookup dimensions")
    words = tuple(int(x, 16) for x in data["words_hex"])
    if len(words) != 1 << a or any(x < 0 or x.bit_length() > w for x in words):
        raise ValueError("Table must define every address and fit its word width")
    q = spec["max_qubits"]
    if type(q) is not int or not a + w <= q <= 8192:
        raise ValueError("Invalid logical qubit budget")
    return LookupTask(spec["name"], a, w, words, q, digest(path), digest(table), path)


def load_rotation(path):
    base = load_lookup(path, _schema="qpe.rotation.v1")
    spec = json.loads(base.path.read_text())
    beta, modes, spins = spec["angle_bits"], spec["spatial_modes"], spec["spin_copies"]
    if type(beta) is not int or not 1 <= beta <= 24:
        raise ValueError("Unsupported dyadic angle precision")
    if type(modes) is not int or not 2 <= modes <= 32:
        raise ValueError("Unsupported spatial mode count")
    if type(spins) is not int or spins not in (1, 2) or type(spec["controlled"]) is not bool:
        raise ValueError("Invalid spin copies or control contract")
    if base.word_bits != beta * (modes - 1):
        raise ValueError("Angle register does not match the fixed rotation network")
    if spec.get("consumer") != "inverse-elimination-givens-v1":
        raise ValueError("Unsupported rotation consumer")
    if spec.get("mode_order") != "spin-blocked" or spec.get("angle_initial") != "zero":
        raise ValueError("Unsupported mode order or angle input")
    task = RotationTask(
        **asdict(base),
        angle_bits=beta,
        spatial_modes=modes,
        spin_copies=spins,
        controlled=spec["controlled"],
    )
    if base.max_qubits < base.address_bits + base.word_bits + task.reserved_qubits:
        raise ValueError("Qubit budget cannot hold the persistent registers")
    table = json.loads((base.path.parent / spec["table"]).read_text())
    if table.get("angle_bits") != beta or table.get("angles_per_word") != modes - 1:
        raise ValueError("Angle table metadata and rotation task disagree")
    return task


def read_task(path):
    schema = json.loads(Path(path).read_text()).get("schema")
    if schema == "qpe.lookup.v1":
        return load_lookup(path)
    if schema == "qpe.rotation.v1":
        return load_rotation(path)
    if schema == "qpe.walk.v1":
        return load_walk(path)
    raise ValueError(f"Unsupported task schema: {schema}")


def load_walk(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if spec.get("schema") != "qpe.walk.v1":
        raise ValueError("Expected qpe.walk.v1 task")
    table = (path.parent / spec["table"]).resolve()
    if not table.is_relative_to(path.parent) or digest(table) != spec["table_sha256"]:
        raise ValueError("Invalid SOS model location or hash")
    if table.stat().st_size > 8 * 1024**2:
        raise ValueError("SOS model too large")
    model = json.loads(table.read_text())
    if model.get("schema") != "qpe.sos-model.v1" or model.get("mode_order") != "spin-blocked":
        raise ValueError("Unsupported SOS model")
    for key, lower, upper in (
        ("spatial_modes", 2, 32),
        ("angle_bits", 1, 24),
        ("coefficient_bits", 1, 40),
    ):
        if type(model[key]) is not int or not lower <= model[key] <= upper:
            raise ValueError(f"Unsupported SOS dimension: {key}")
    angles = model["orbital_angles"]
    if not isinstance(angles, list) or not 1 <= len(angles) <= 1024:
        raise ValueError("Invalid orbital table")
    for row in angles:
        if len(row) != model["spatial_modes"] - 1 or any(
            type(x) is not int or not 0 <= x < 1 << model["angle_bits"] for x in row
        ):
            raise ValueError("Invalid orbital angles")
    generators = model["generators"]
    if not isinstance(generators, list) or not 1 <= len(generators) <= 512:
        raise ValueError("Invalid generator count")
    for generator in generators:
        terms = generator["terms"]
        if not isinstance(terms, list) or not 1 <= len(terms) <= 128:
            raise ValueError("Invalid LCU term count")
        for term in terms:
            if set(term) != {"weight", "orbital", "spin", "pauli", "phase"}:
                raise ValueError("Invalid SOS term fields")
            for key in ("weight", "orbital", "spin", "phase"):
                if type(term[key]) is not int:
                    raise ValueError("SOS indices and weights must be integers")
            if (
                not 0 <= term["weight"] < 2**48
                or term["spin"] not in (0, 1)
                or term["phase"] not in range(4)
            ):
                raise ValueError("Invalid weight, spin, or phase")
            if term["pauli"] not in ("I", "X", "Y", "Z"):
                raise ValueError("Unknown SOS Pauli operator")
            if (term["pauli"] == "I" and term["orbital"] != -1) or (
                term["pauli"] != "I" and not 0 <= term["orbital"] < len(angles)
            ):
                raise ValueError("Invalid orbital reference")
        if not sum(t["weight"] for t in terms):
            raise ValueError("Zero-norm SOS generator")
    if type(model["energy_offset"]) not in (int, float) or not math.isfinite(
        model["energy_offset"]
    ):
        raise ValueError("Nonfinite energy offset")
    expected_input, expected_output = (
        (
            "x,b,spin_outer,spin_inner,sigma_outer,sigma_inner,uniform=0",
            "b,spin_inner,sigma_inner,uniform_inner=0; retain outer",
        )
        if "paper" in model
        else ("label=0,term=0", "term=0; retain label")
    )
    if "uniform_bits" in model.get("paper", {}):
        expected_output += "; success_outer=1"
    if spec.get("walk_convention") != "Rin Udag Rout U; R=2P-I" or (
        spec.get("input_projector") != expected_input
        or spec.get("output_projector") != expected_output
    ):
        raise ValueError("Unsupported rectangular projector convention")
    if type(spec.get("controlled")) is not bool:
        raise ValueError("Invalid walk control")
    task = WalkTask(
        spec["name"],
        model,
        spec["max_qubits"],
        spec["controlled"],
        digest(path),
        digest(table),
        path,
        spec.get("optimization"),
    )
    if type(task.max_qubits) is not int or not task.logical_qubits <= task.max_qubits <= 8192:
        raise ValueError("Invalid walk qubit budget")
    if "preparation" in model:
        prep = model["preparation"]
        if (
            not isinstance(prep, dict)
            or set(prep) != {"schema", "alias_bits", "outer", "inner"}
            or prep["schema"] != "qpe.alias-target.v1"
            or type(prep["alias_bits"]) is not int
            or not 1 <= prep["alias_bits"] <= 24
            or not isinstance(prep["inner"], list)
            or len(prep["inner"]) != len(generators)
        ):
            raise ValueError("Invalid fixed alias preparation target")
        distributions = [(prep["outer"], task.label_bits, len(generators))]
        distributions += [
            (row, task.term_bits, len(g["terms"]))
            for row, g in zip(prep["inner"], generators, strict=True)
        ]
        for row, width, populated in distributions:
            if (
                not isinstance(row, list)
                or len(row) != 1 << width
                or any(type(w) is not int or w < 0 for w in row)
                or sum(row) != 1 << (width + prep["alias_bits"])
                or any(row[populated:])
            ):
                raise ValueError("Invalid frozen alias cell distribution")
        if any(
            count and not term["weight"]
            for row, generator in zip(prep["inner"], generators, strict=True)
            for count, term in zip(row, generator["terms"])
        ):
            raise ValueError("Alias target assigns mass to an absent SELECT term")
    if "paper" in model:
        from qpe_lab.tasks.data.hamiltonian import with_paper_preparation

        prep = model["paper"]
        extra = {"uniform_bits"} if prep.get("schema") == "qpe.paper-target.v2" else set()
        if (
            "preparation" in model
            or not isinstance(prep, dict)
            or set(prep) - extra
            != {"schema", "alias_bits", "rank", "bases", "copies", "outer", "inner"}
            or prep["schema"] not in ("qpe.paper-target.v1", "qpe.paper-target.v2")
            or (extra and "uniform_bits" not in prep)
            or type(prep["alias_bits"]) is not int
            or not 1 <= prep["alias_bits"] <= 24
        ):
            raise ValueError("Invalid paper preparation target")
        rebuilt = with_paper_preparation(
            {k: v for k, v in model.items() if k != "paper"},
            prep["alias_bits"],
            uniform_bits=prep.get("uniform_bits"),
        )["paper"]
        if prep != rebuilt:
            raise ValueError("Paper target differs from the frozen factor quantization")
    contract = task.contract
    if (
        set(contract)
        != {
            "cost_model",
            "walk_error_budget",
            "objective",
            "synthesis_backend",
            "ccx_decomposition",
        }
        or contract["cost_model"] not in ("ideal", "clifford_t")
        or contract["objective"] not in ("t_count", "tq_cost")
        or contract["synthesis_backend"] != "pygridsynth-2.0.0"
        or contract["ccx_decomposition"] != "7t-no-ancilla"
        or type(contract["walk_error_budget"]) not in (int, float)
        or not math.isfinite(contract["walk_error_budget"])
        or (contract["cost_model"] == "ideal" and contract["walk_error_budget"] != 0)
        or (
            contract["cost_model"] == "clifford_t"
            and not 1e-7 <= contract["walk_error_budget"] <= 0.1
        )
    ):
        raise ValueError("Invalid fixed optimization contract")
    return task
