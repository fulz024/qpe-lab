"""Shared task specifications, verification goals and frozen policy data."""

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal


def project_root():
    return Path(__file__).resolve().parents[2]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def sampling_policy(acceptance="exact_or_sampled", *, lean=False):
    if acceptance not in {"exact_or_sampled", "lean_required"}:
        raise ValueError("Unknown verification acceptance policy")
    return {
        "acceptance": acceptance,
        "backends": ["exact", "lean", "sampled"]
        if lean or acceptance == "lean_required"
        else ["exact", "sampled"],
        "campaign_confidence_budget": 1e-9,
        "sampled_claim": "complete_map_or_channel",
        "basis_diagnostic_shots": 4096,
        "exact_seconds_per_entry": 60,
        "schedule": {
            "version": "staged-v1",
            "quick_seconds": 2,
            "certificate_seconds": 10,
            "planning_seconds": 3,
            "local_seconds": 15,
            "whole_sampled_fraction": 0.5,
        },
        "deferred_timeout_seconds": 1800,
        "coherent_max_wires": 12,
        "coherent_max_records": 8,
        "tensor_max_frontier": 16,
        "tensor_max_products": 2000000,
        "attempt_allocation": "delta/(i*(i+1))",
        "query_allocation": "attempt_delta/(j*(j+1))",
    }


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def affine_columns(forms, wires):
    """Enumerate a small affine image in the supplied local wire order."""
    one = (2,)
    offset = sum(1 << i for i, q in enumerate(wires) if one in forms[q])
    variables = set().union(*(forms[q] - {one} for q in wires))
    span = {0}
    for variable in variables:
        mask = sum(1 << i for i, q in enumerate(wires) if variable in forms[q])
        span |= {value ^ mask for value in tuple(span)}
    return sorted(value ^ offset for value in span)


@dataclass(frozen=True)
class VerificationGoal:
    """A maintainer-generated proposition, independent of its proof backend.

    Affine input forms describe a sound prefix-derived subspace. Outputs name
    classical records observable by the continuation; all quantum wires remain
    observable. Missing forms use the frozen reference's clean-input contract.
    """

    reference: dict
    proposed: list
    reference_sha256: str
    max_qubits: int
    certificate: dict | None = None
    scope: dict | None = None
    input_conditions: dict | None = None
    output_requirements: dict | None = None
    semantics: str = "qpe.phase-sensitive-instrument.v1"
    proof: dict | None = None
    attachments: list | None = None
    resource_inputs: dict | None = None

    @property
    def statement_sha256(self):
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        for key in ("scope", "certificate", "proof", "attachments"):
            data.pop(key)
        return self._hash(data)

    @staticmethod
    def _hash(data):
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    @property
    def sha256(self):
        return self._hash({f.name: getattr(self, f.name) for f in fields(self)})

    @property
    def wires(self):
        return sorted({q for g in self.reference["gates"] + self.proposed for q in g[1]})

    @property
    def forms(self):
        if self.input_conditions is not None:
            return {q: frozenset(map(tuple, terms)) for q, terms in self.input_conditions["affine"]}
        clean = self.reference.get("ancilla_initial") == "zero"
        protected = self.reference["protected_wires"]
        return {
            q: frozenset({(0, q)}) if not clean or q in protected else frozenset()
            for q in self.wires
        }

    @property
    def inputs(self):
        return sorted(q for q, form in self.forms.items() if form)

    @property
    def columns(self):
        """Small exact backends call this only after checking their wire cap."""
        return affine_columns(self.forms, self.wires)

    @property
    def external_records(self):
        return (self.input_conditions or {}).get("external_records", [])

    @property
    def live_records(self):
        return (self.output_requirements or {}).get("classical_records", [])

    @property
    def resources(self):
        """Trusted product amplitudes |0> + exp(2*pi*i*p/2**b)|1>.

        Normalization is common to both programs. Entries come exclusively
        from the frozen task, or a proved reference prefix, never the proposal.
        """
        wires = set(self.wires)
        return {
            int(q): tuple(value)
            for q, value in (self.resource_inputs or {}).items()
            if int(q) in wires
        }


@dataclass(frozen=True)
class VerificationResult:
    """Backend evidence only. Cost, reward and scientific review belong elsewhere."""

    status: Literal["proved", "sampled", "counterexample", "unknown", "error"]
    goal_sha256: str
    evidence: dict | None = None
    detail: str | None = None
    reason: Literal["unsupported", "timeout", "tool_error"] | None = None


@dataclass(frozen=True)
class LookupTask:
    name: str
    address_bits: int
    word_bits: int
    words: tuple[int, ...]
    max_qubits: int
    task_hash: str
    table_hash: str
    path: Path


@dataclass(frozen=True)
class RotationTask(LookupTask):
    """A fixed dyadic-angle orbital rotation, with a clean angle register."""

    angle_bits: int
    spatial_modes: int
    spin_copies: int
    controlled: bool

    @property
    def mode_count(self):
        return self.spatial_modes * self.spin_copies

    @property
    def reserved_qubits(self):
        return self.mode_count + int(self.controlled)

    def lookup_contract(self):
        fields = {key: getattr(self, key) for key in LookupTask.__dataclass_fields__}
        fields["max_qubits"] -= self.reserved_qubits
        return LookupTask(**fields)


@dataclass(frozen=True)
class WalkTask:
    """Fixed rectangular SOS encoding and its spectrum-amplified walk."""

    name: str
    model: dict
    max_qubits: int
    controlled: bool
    task_hash: str
    table_hash: str
    path: Path
    optimization: dict | None = None

    @property
    def contract(self):
        return (
            self.optimization
            if self.optimization is not None
            else {
                "cost_model": "ideal",
                "walk_error_budget": 0.0,
                "objective": "t_count",
                "synthesis_backend": "pygridsynth-2.0.0",
                "ccx_decomposition": "7t-no-ancilla",
            }
        )

    @property
    def label_bits(self):
        if "paper" in self.model:
            return max(1, (len(self.model["paper"]["outer"]) - 1).bit_length())
        return max(1, (len(self.model["generators"]) - 1).bit_length())

    @property
    def term_bits(self):
        if "paper" in self.model:
            return max(1, self.model["paper"]["bases"].bit_length())
        return max(1, (max(len(g["terms"]) for g in self.model["generators"]) - 1).bit_length())

    @property
    def mode_count(self):
        return 2 * self.model["spatial_modes"]

    @property
    def logical_qubits(self):
        return (
            self.label_bits
            + self.term_bits
            + self.mode_count
            + int(self.controlled)
            + (2 if "paper" in self.model else 0)
        )
