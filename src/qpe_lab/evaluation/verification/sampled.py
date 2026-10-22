"""One-goal sampling and persistent private challenge allocation."""

import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path

from qpe_lab.circuits.sim import Incorrect, Unsupported
from qpe_lab.contracts import VerificationGoal, VerificationResult, affine_columns, sampling_policy


def verify(goal: VerificationGoal, context) -> VerificationResult:
    if goal.semantics != "qpe.phase-sensitive-instrument.v1":
        return VerificationResult(
            "unknown",
            goal.sha256,
            detail="Sampling does not cover this equivalence relation",
            reason="unsupported",
        )
    if context is None:
        return VerificationResult(
            "unknown", goal.sha256, detail="No sampling challenge supplied", reason="unsupported"
        )
    try:
        evidence = context.compare(
            goal.reference["gates"],
            goal.proposed,
            goal.inputs,
            external=goal.external_records,
            outputs=goal.live_records,
            forms=goal.forms,
            resources=goal.resources,
        )
        bound, budget = (
            evidence.get("false_accept_bound"),
            evidence.get("confidence_failure_budget"),
        )
        if (
            evidence.get("claim") not in {"complete isometry equality", "complete channel equality"}
            or not isinstance(bound, (int, float))
            or not isinstance(budget, (int, float))
            or not 0 <= bound <= budget < 1
        ):
            raise Unsupported("Sampled acceptance requires a complete-map false-acceptance bound")
        return VerificationResult("sampled", goal.sha256, evidence)
    except Incorrect as exc:
        return VerificationResult("counterexample", goal.sha256, detail=str(exc))
    except Unsupported as exc:
        return VerificationResult("unknown", goal.sha256, detail=str(exc), reason="unsupported")


class SamplingContext:
    """Trainer-owned random draws and a summable confidence budget per query."""

    def __init__(self, seed, alpha, *, shots=4096):
        self.seed, self.alpha, self.shots = seed, alpha, shots
        self.queries = 0

    def compare(
        self, before, after, inputs, *, external=(), outputs=(), forms=None, resources=None
    ):
        import hashlib

        from qpe_lab.circuits.sim import sample_monomial_equivalence

        self.queries += 1
        if self.queries > 512:
            raise Unsupported("Sampling decomposition exceeds 512 comparisons")
        index = self.queries
        alpha = self.alpha / (index * (index + 1))
        seed = hashlib.sha256(self.seed + index.to_bytes(8, "little")).digest()
        # Fingerprints cover the complete map, including rare input branches.
        # Basis probes can diagnose a failed comparison, never certify it.
        try:
            return self.fingerprint(
                before,
                after,
                inputs,
                seed,
                alpha,
                external=external,
                outputs=outputs,
                forms=forms,
                resources=resources,
            )
        except Unsupported as failure:
            reason = str(failure)
        basis_domain_known = forms is None or not any(
            term[0] == 1 for terms in forms.values() for term in terms
        )
        if not external and not outputs and basis_domain_known and not resources:
            try:
                sample_monomial_equivalence(
                    before, after, inputs, seed=seed, shots=self.shots, forms=forms
                )
                reason += (
                    "; basis probes found no counterexample; complete-map verification unfinished"
                )
            except Unsupported:
                pass
        raise Unsupported(reason)

    def fingerprint(
        self,
        before,
        after,
        inputs,
        seed,
        alpha,
        *,
        external=(),
        outputs=(),
        forms=None,
        resources=None,
    ):
        from qpe_lab.circuits.sim import (
            sample_channel_equivalence,
            tensor_channel_fingerprint,
            tensor_isometry_check,
        )

        coherent = (
            not external
            and not outputs
            and not any(v is not None or n in {"hmr", "oxor"} for n, _, v in before + after)
        )
        if coherent:
            wires = {q for _, qs, _ in before + after for q in qs}
            tensor_forms = (
                forms
                if forms is not None
                else {q: frozenset({(0, q)}) if q in inputs else frozenset() for q in wires}
            )
            # Union bound covers the equality polynomial and two assertion
            # trace-loss checks. Every random variable is chosen AFTER freeze.
            variables = set().union(
                *(v - {(2,)} for q, v in tensor_forms.items() if q not in (resources or {}))
            )
            degree = max(1, len(variables) + len(tensor_forms), 2 * len(variables))
            bits = max(32, degree.bit_length() + 16)
            rounds = max(1, math.ceil(math.log(alpha / 3) / math.log(degree / 2**bits)))
            return tensor_isometry_check(
                before,
                after,
                tensor_forms,
                resources,
                seed=seed,
                rounds=rounds,
                coefficient_bits=bits,
            ) | {"confidence_failure_budget": alpha, "claim": "complete isometry equality"}
        wires = {q for _, qs, _ in before + after for q in qs}
        if not resources and len(wires) <= 12:
            try:
                columns = affine_columns(forms, sorted(wires)) if forms is not None else None
                degree = 4 + sum(n == "assert_zero" for word in (before, after) for n, _, _ in word)
                bits = max(32, degree.bit_length() + 16)
                rounds = max(1, math.ceil(math.log(alpha) / math.log(degree / 2**bits)))
                return sample_channel_equivalence(
                    before,
                    after,
                    inputs,
                    seed=seed,
                    rounds=rounds,
                    coefficient_bits=bits,
                    external=external,
                    outputs=outputs,
                    columns=columns,
                ) | {"confidence_failure_budget": alpha, "claim": "complete channel equality"}
            except Unsupported:
                pass
        tensor_forms = (
            forms
            if forms is not None
            else {q: frozenset({(0, q)}) if q in inputs else frozenset() for q in wires}
        )
        variables = set().union(
            *(v - {(2,)} for q, v in tensor_forms.items() if q not in (resources or {}))
        )
        degree = max(1, 2 * (len(variables) + len(tensor_forms)) + len(external) + len(outputs))
        bits = max(32, degree.bit_length() + 16)
        rounds = max(1, math.ceil(math.log(alpha / 3) / math.log(degree / 2**bits)))
        return tensor_channel_fingerprint(
            before,
            after,
            tensor_forms,
            resources,
            seed=seed,
            rounds=rounds,
            coefficient_bits=bits,
            external=external,
            outputs=outputs,
        ) | {"confidence_failure_budget": alpha, "claim": "complete channel equality"}


def reserve_sampling_attempt(ledger, bundle_hash, gates_hash):
    """Atomic, persistent trainer ledger. Every frozen submission consumes an index."""
    import secrets
    import sqlite3

    path = Path(ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = secrets.token_bytes(32)
    commitment = hashlib.sha256(seed).hexdigest()
    with sqlite3.connect(path, timeout=30, isolation_level=None) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, bundle TEXT NOT NULL, gates TEXT NOT NULL, commitment TEXT NOT NULL, created TEXT NOT NULL)"
        )
        db.execute("BEGIN IMMEDIATE")
        cursor = db.execute(
            "INSERT INTO attempts (bundle,gates,commitment,created) VALUES (?,?,?,?)",
            (bundle_hash, gates_hash, commitment, datetime.now(UTC).isoformat()),
        )
        index = cursor.lastrowid
        db.commit()
    return {
        "attempt": index,
        "seed": seed.hex(),
        "seed_commitment": commitment,
        "bundle_sha256": bundle_hash,
        "h_ops": gates_hash,
        "confidence_budget": sampling_policy()["campaign_confidence_budget"]
        / (index * (index + 1)),
    }
