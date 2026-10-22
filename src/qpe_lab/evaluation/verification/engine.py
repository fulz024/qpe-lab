"""Generate goals, dispatch backends, and compose complete circuit evidence."""

import copy
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from qpe_lab.circuits.checks import classical_interface, elementary_cost, verify_phase_fusion
from qpe_lab.circuits.ops import elementary_gates
from qpe_lab.circuits.sim import Incorrect, Unsupported
from qpe_lab.contracts import VerificationGoal, VerificationResult, sampling_policy
from qpe_lab.evaluation.verification import exact, lean, sampled


@dataclass(frozen=True)
class Backend:
    name: str
    assurance: str
    verify: Callable[[VerificationGoal, object], VerificationResult]
    version: str = "1"


# Only the maintainer registers executable verifiers and their assurance level.
BACKENDS = (
    Backend("exact", "proved", exact.verify, "3"),
    Backend("lean", "proved", lean.verify, "monomial-1/lean-4.35.0"),
    Backend("sampled", "sampled", sampled.verify, "4"),
)


class ProofCache:
    """Bounded, evaluator-owned exact evidence. Never caches sampled verdicts.

    A cache lives in one trusted evaluator process. Backend identity/version,
    concrete gates, contract, input/output conditions and semantics bind each
    entry. Evidence is copied on both insertion and retrieval.
    """

    def __init__(self, capacity=512):
        self.entries, self.capacity = OrderedDict(), capacity

    def key(self, backend, goal):
        return backend.name, backend.version, backend.verify, goal.statement_sha256

    def get(self, backend, goal):
        key = self.key(backend, goal)
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return copy.deepcopy(self.entries[key])

    def put(self, backend, goal, evidence):
        if backend.assurance != "proved":
            return
        self.entries[self.key(backend, goal)] = copy.deepcopy(evidence)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)


class Session:
    def __init__(
        self,
        seconds,
        context=None,
        *,
        exact_only=False,
        cache=None,
        required_backend=None,
        enabled_backends=None,
        progress=None,
    ):
        self.remaining, self.context, self.exact_only = seconds, context, exact_only
        self.cache = cache if cache is not None else ProofCache()
        self.required_backend = required_backend
        self.enabled_backends = enabled_backends
        self.attempts = []
        self.disproved = {}
        self.progress = progress
        self.phases = []
        self.phase = None
        self.phase_deadline = float("inf")

    def notify(self, backend=None):
        if self.progress is not None and self.phase is not None:
            self.progress(self.phase | {"backend": backend})

    def run_phase(self, name, seconds, function, *, local=False):
        """Bound all phase work, including partition discovery and composition."""
        if seconds <= 0:
            return None
        started = time.monotonic()
        self.phase = {"phase": name, "phase_started": started, "phase_budget_seconds": seconds}
        self.phase_deadline = started + seconds
        self.notify()
        outcome = "completed"
        try:
            return timed_verification(function, seconds)
        except Incorrect:
            outcome = "counterexample"
            if not local:
                raise
        except Unsupported:
            outcome = "unfinished"
        finally:
            self.phases.append(
                {
                    "phase": name,
                    "budget_seconds": seconds,
                    "seconds": time.monotonic() - started,
                    "outcome": outcome,
                }
            )
            self.phase_deadline = float("inf")
        return None

    def dispatch(self, goal, *, assurance=None):
        bound = goal.sha256
        statement = goal.statement_sha256
        if statement in self.disproved:
            raise Incorrect(self.disproved[statement])
        for backend in BACKENDS:
            if self.enabled_backends is not None and backend.name not in self.enabled_backends:
                continue
            if backend.assurance not in {"proved", "sampled"}:
                raise ValueError("Invalid maintained backend assurance")
            if (
                self.exact_only
                and backend.assurance != "proved"
                or assurance is not None
                and backend.assurance != assurance
                or self.required_backend is not None
                and backend.name != self.required_backend
            ):
                continue
            evidence = self.cache.get(backend, goal) if backend.assurance == "proved" else None
            cached = evidence is not None
            started = time.monotonic()
            self.notify(backend.name)
            try:
                if cached:
                    result = VerificationResult("proved", bound, evidence)
                else:
                    allowance = self.phase_deadline - time.monotonic()
                    if backend.assurance == "proved":
                        allowance = min(allowance, self.remaining)
                    if allowance <= 0:
                        raise Unsupported("Verification reached its reserved time budget")
                    call = lambda backend=backend: backend.verify(goal, self.context)
                    result = (
                        call() if allowance == float("inf") else timed_verification(call, allowance)
                    )
            except Unsupported as exc:
                result = VerificationResult(
                    "unknown",
                    bound,
                    detail=str(exc),
                    reason="timeout" if "time budget" in str(exc) else "unsupported",
                )
            except TimeoutError as exc:
                result = VerificationResult("unknown", bound, detail=str(exc), reason="timeout")
            finally:
                if backend.assurance == "proved" and not cached:
                    self.remaining -= time.monotonic() - started
            if not isinstance(result, VerificationResult) or result.goal_sha256 != bound:
                raise ValueError("Backend result does not bind the submitted verification goal")
            if goal.sha256 != bound:
                raise ValueError("Backend mutated the verification goal")
            self.attempts.append(
                {
                    "backend": backend.name,
                    "version": backend.version,
                    "goal_sha256": bound,
                    "scope": goal.scope,
                    "status": result.status,
                    "reason": result.reason,
                    "detail": result.detail,
                    "cached": cached,
                    "phase": (self.phase or {}).get("phase"),
                    "seconds": time.monotonic() - started,
                }
            )
            if result.status == "counterexample":
                detail = result.detail or "Backend found a counterexample"
                self.disproved[statement] = detail
                raise Incorrect(detail)
            if result.status in {"unknown", "error"}:
                continue
            if result.status != backend.assurance or not isinstance(result.evidence, dict):
                raise ValueError("Invalid backend evidence or assurance level")
            if not cached:
                self.cache.put(backend, goal, result.evidence)
            return result.evidence | {
                "goal_sha256": bound,
                "statement_sha256": goal.statement_sha256,
                "backend": backend.name,
                "backend_version": backend.version,
                "sampling": result.status == "sampled",
                "cached": cached,
                "scope": goal.scope,
            }
        raise Unsupported("No backend completed verification of this goal")


class PrefixFacts:
    """Sound affine support of a trusted prefix; assertions add no assumptions."""

    def __init__(self, reference):
        self.clean = reference.get("ancilla_initial") == "zero"
        self.protected = set(reference["protected_wires"])
        self.forms, self.fresh = {}, 0

    def form(self, q):
        return self.forms.get(
            q, frozenset() if self.clean and q not in self.protected else frozenset({(0, q)})
        )

    def run(self, gates):
        for name, wires, value in gates:
            if name == "hmr":
                self.forms[wires[0]] = frozenset()
            elif name == "x" and value is None:
                self.forms[wires[0]] = self.form(wires[0]) ^ {(2,)}
            elif name == "cx" and value is None:
                self.forms[wires[1]] = self.form(wires[0]) ^ self.form(wires[1])
                if len(self.forms[wires[1]]) > 4096:
                    raise Unsupported("Affine proof expression limit exceeded")
            elif name in {"h", "x", "cx"}:
                self.fresh += 1
                self.forms[wires[-1]] = frozenset({(1, self.fresh)})


class PrefixResources:
    """Proved separable equatorial factors of the actual reference prefix.

    Only exact state propagation establishes a resource assumption at a cut.
    In particular, X-eigenstate kickback preserves its target even when the
    controls and all remaining logical inputs are arbitrary and entangled.
    """

    def __init__(self, resources):
        self.states = dict(resources)

    def phase(self, q, power):
        if q in self.states:
            bits, value = self.states[q]
            precision = max(3, bits)
            self.states[q] = (
                precision,
                (value * (1 << (precision - bits)) + power * (1 << (precision - 3)))
                % (1 << precision),
            )

    def run(self, gates):
        from qpe_lab.circuits.gadgets.synthesis import ccx_clifford_t

        i = 0
        while i < len(gates):
            name, qs, value = gates[i]
            length = 1
            if name in {"h", "cx"} and value is None:
                word = gates[i : i + 15]
                if name == "cx":
                    inverse = {"t": "tdg", "tdg": "t", "s": "sdg", "sdg": "s"}
                    word = [(inverse.get(n, n), ws, v) for n, ws, v in reversed(word)]
                if len(word) == 15 and word[0][0] == "h" and word[3][0] == word[1][0] == "cx":
                    a, b, t = word[3][1][0], word[1][1][0], word[0][1][0]
                    if len({a, b, t}) == 3 and word == list(ccx_clifford_t(a, b, t)):
                        name, qs, length = "ccx", (a, b, t), 15
            i += length
            if name in {"neg", "oxor", "assert_zero"}:
                continue
            if name in {"z", "s", "sdg", "t", "tdg"} and value is None:
                self.phase(qs[0], {"z": 4, "s": 2, "sdg": -2, "t": 1, "tdg": -1}[name])
                continue
            if name == "x" and value is None and qs[0] in self.states:
                bits, power = self.states[qs[0]]
                self.states[qs[0]] = (bits, -power % (1 << bits))
                continue
            if name in {"x", "cx", "ccx"} and qs[-1] in self.states:
                bits, power = self.states[qs[-1]]
                power %= 1 << bits
                if power in {0, 1 << (bits - 1)}:
                    if power:
                        if name == "cx" and value is None:
                            self.phase(qs[0], 4)
                        else:
                            for q in qs[:-1]:
                                self.states.pop(q, None)
                    continue
            for q in qs:
                self.states.pop(q, None)


def region_goals(parent, regions):
    """Check full coverage and derive every interface from actual gate streams."""
    before, after = parent.reference["gates"], parent.proposed
    parent_hash = parent.sha256
    facts, left, right, goals = PrefixFacts(parent.reference), 0, 0, []
    resources = PrefixResources(parent.resources)
    for a, b, c, d in regions:
        if not (left <= a <= b <= len(before) and right <= c <= d <= len(after)):
            raise Incorrect("Overlapping or out-of-range verification regions")
        if before[left:a] != after[right:c]:
            raise Incorrect("Verification regions leave an uncovered change")
        facts.run(before[left:a])
        resources.run(before[left:a])
        old, new = before[a:b], after[c:d]
        od, rd = classical_interface(old)
        nd, nr = classical_interface(new)
        external = sorted((rd - od) | (nr - nd))
        live = classical_interface(before[b:])[1] | classical_interface(after[d:])[1]
        wires = sorted({q for g in old + new for q in g[1]})
        goals.append(
            VerificationGoal(
                parent.reference | {"gates": old},
                new,
                parent.reference_sha256,
                parent.max_qubits,
                scope={
                    "kind": "region",
                    "parent_sha256": parent_hash,
                    "reference_range": [a, b],
                    "candidate_range": [c, d],
                },
                input_conditions={
                    "affine": [[q, sorted(facts.form(q))] for q in wires],
                    "external_records": external,
                },
                output_requirements={
                    "quantum_wires": wires,
                    "classical_records": sorted((od | nd) & live),
                },
                resource_inputs={q: v for q, v in resources.states.items() if q in wires},
                proof=next(
                    (
                        p
                        for p in parent.attachments or []
                        if p["reference_range"] == [a, b] and p["candidate_range"] == [c, d]
                    ),
                    None,
                ),
            )
        )
        facts.run(old)
        resources.run(old)
        left, right = b, d
    if before[left:] != after[right:]:
        raise Incorrect("Verification regions leave an uncovered tail")
    return goals


def decompositions(before, after):
    start = 0
    while start < min(len(before), len(after)) and before[start] == after[start]:
        start += 1
    a, b = len(before), len(after)
    while a > start and b > start and before[a - 1] == after[b - 1]:
        a, b = a - 1, b - 1
    if a == b == start:
        return
    plans = [[(start, a, start, b)]]
    if max(len(before), len(after)) <= 200_000:
        matcher = SequenceMatcher(None, list(map(repr, before)), list(map(repr, after)))
        intervals = [row[1:] for row in matcher.get_opcodes() if row[0] != "equal"]
        if len(intervals) <= 128:
            plans.append(intervals)
    seen = set()
    for plan in plans:
        for pad in (0, 2, 16):
            regions = []
            for a, b, c, d in plan:
                r = [
                    max(0, a - pad),
                    min(len(before), b + pad),
                    max(0, c - pad),
                    min(len(after), d + pad),
                ]
                if regions and (r[0] <= regions[-1][1] or r[2] <= regions[-1][3]):
                    regions[-1][1], regions[-1][3] = r[1], r[3]
                else:
                    regions.append(r)
            key = tuple(map(tuple, regions))
            if key not in seen:
                seen.add(key)
                yield regions


def compose(session, goal, regions, *, assurance=None):
    goals = region_goals(goal, regions)
    proofs = [session.dispatch(part, assurance=assurance) for part in goals]
    return {
        "method": "covered region composition",
        "checks": proofs,
        "coverage": [part.scope for part in goals],
        "sampling": any(p["sampling"] for p in proofs),
    }


def attachment_plan(goal):
    """Use attached proof boundaries while still covering all other changes."""
    before, after = goal.reference["gates"], goal.proposed
    anchors = sorted(
        p["reference_range"] + p["candidate_range"]
        for p in goal.attachments or []
        if p["reference_range"] is not None
    )
    if not anchors:
        return None
    result, left, right = [], 0, 0
    for a, b, c, d in anchors + [[len(before), len(before), len(after), len(after)]]:
        if not (left <= a <= b <= len(before) and right <= c <= d <= len(after)):
            return None
        if before[left:a] != after[right:c]:
            # The unannotated gap is itself a goal; other backends may prove it.
            result.append([left, a, right, c])
        if a != b or c != d:
            result.append([a, b, c, d])
        left, right = b, d
    return result


def certificate_program(reference, certificate, reference_sha256):
    if certificate["reference_sha256"] != reference_sha256:
        raise Incorrect("Submission targets a different frozen gate reference")
    current = list(reference["gates"])
    stages = []
    for stage in certificate["stages"]:
        regions, proposed, cursor = [], [], 0
        if stage["kind"] in {"reuse", "phase"}:
            proposed = elementary_gates(stage["gates"])
        else:
            for edit in stage["edits"]:
                a, b = edit["start"], edit["stop"]
                if not cursor <= a < b <= len(current):
                    raise Incorrect("Local rewrite range is outside the actual input stream")
                proposed.extend(current[cursor:a])
                c = len(proposed)
                proposed.extend(elementary_gates(edit["gates"]))
                regions.append([a, b, c, len(proposed)])
                cursor = b
            proposed.extend(current[cursor:])
        verify_phase_fusion(proposed, proposed)
        stages.append((stage, current, proposed, regions))
        current = proposed
    return current, stages


def prove_certificate(session, goal, stages, *, assurance=None):
    proofs = []
    for index, (stage, before, after, regions) in enumerate(stages):
        part = replace(
            goal,
            reference=goal.reference | {"gates": before},
            proposed=after,
            certificate=None,
            scope={"kind": "stage", "index": index},
            proof=stage if stage["kind"] == "reuse" else None,
            semantics="qpe.clean-wire-allocation.v1"
            if stage["kind"] == "reuse"
            else goal.semantics,
        )
        if stage["kind"] in {"reuse", "phase"}:
            evidence = session.dispatch(part, assurance="proved")
        else:
            evidence = compose(session, part, regions, assurance=assurance)
        proofs.append({"kind": stage["kind"], **evidence})
    return {
        "method": "verified transformation composition",
        "stages": proofs,
        "sampling": any(p["sampling"] for p in proofs),
    }


def verify_entry(
    reference,
    candidate,
    reference_sha256,
    *,
    max_qubits=1024,
    exact_seconds=60,
    context=None,
    exact_only=False,
    cache=None,
    required_backend=None,
    enabled_backends=None,
    resource_inputs=None,
    seconds=300,
    schedule=None,
    progress=None,
):
    started = time.monotonic()
    deadline = started + seconds
    schedule = schedule or sampling_policy()["schedule"]
    scale = min(1.0, seconds / 300)

    def left():
        return max(0, deadline - time.monotonic())

    goal = VerificationGoal(
        reference,
        elementary_gates(candidate["gates"]),
        reference_sha256,
        max_qubits,
        candidate.get("certificate"),
        scope={"kind": "entry"},
        attachments=candidate.get("proofs"),
        resource_inputs=resource_inputs,
    )
    goal = replace(
        goal, proof=next((p for p in goal.attachments or [] if p["reference_range"] is None), None)
    )
    cost = elementary_cost(goal.proposed, reference["protected_wires"])
    if cost["qubits"] > max_qubits:
        raise Incorrect("Complete circuit exceeds qubit budget")
    session = Session(
        (
            exact_seconds
            + sum(
                schedule[k] * scale
                for k in ("quick_seconds", "certificate_seconds", "local_seconds")
            )
        )
        if exact_seconds > 0
        else 0,
        context,
        exact_only=exact_only,
        cache=cache,
        required_backend=required_backend,
        enabled_backends=enabled_backends,
        progress=progress,
    )
    proof, stages = None, None

    def phase(name, budget, function, *, local=False):
        return session.run_phase(name, min(left(), budget), function, local=local)

    # A short whole-entry probe handles identical/structural cases. Its timeout
    # cannot consume the separately reserved local and whole-search budgets.
    proof = phase(
        "quick_exact",
        schedule["quick_seconds"] * scale,
        lambda: session.dispatch(goal, assurance="proved"),
    )
    if proof is None and goal.certificate is not None:
        built = phase(
            "certificate_binding",
            schedule["planning_seconds"] * scale,
            lambda: certificate_program(reference, goal.certificate, reference_sha256),
        )
        if built is not None:
            actual, stages = built
            if actual != goal.proposed:
                raise Incorrect("Certificate proves a different circuit than the submitted gates")
            if required_backend is None and stages:
                proof = phase(
                    "certificate_exact",
                    schedule["certificate_seconds"] * scale,
                    lambda: prove_certificate(session, goal, stages, assurance="proved"),
                    local=True,
                )
    plans = []
    if proof is None:

        def plan():
            found = list(decompositions(reference["gates"], goal.proposed))
            # Try smaller changed regions first; a whole rewrite is still a goal.
            found.sort(key=lambda p: max((max(b - a, d - c) for a, b, c, d in p), default=0))
            attached = attachment_plan(goal)
            if attached is not None:
                found.insert(0, attached)
            return found

        plans = phase("partition_planning", schedule["planning_seconds"] * scale, plan) or []

    def regions(assurance):
        for plan in plans:
            try:
                return compose(session, goal, plan, assurance=assurance)
            except (Incorrect, Unsupported):
                pass  # An unsuitable local split is not a global counterexample.
        raise Unsupported("No region plan completed verification")

    if proof is None and plans:
        proof = phase(
            "local_exact", schedule["local_seconds"] * scale, lambda: regions("proved"), local=True
        )
    if proof is None:
        # Retain the old 60-second whole-search allowance, plus unused probe time.
        # Short external deadlines reserve time for both sampled routes as well.
        allowance = min(
            session.remaining, left() if exact_only or required_backend else left() * 0.4
        )
        proof = phase("whole_exact", allowance, lambda: session.dispatch(goal, assurance="proved"))
    if proof is None and not exact_only and required_backend is None:
        proof = phase(
            "whole_sampled",
            left() * schedule["whole_sampled_fraction"],
            lambda: session.dispatch(goal, assurance="sampled"),
        )

        def mixed():
            if stages is not None:
                try:
                    return prove_certificate(session, goal, stages)
                except (Incorrect, Unsupported):
                    pass
            return regions(None)

        if proof is None:
            proof = phase("mixed_regions", left(), mixed, local=True)
    if proof is None:
        error = Unsupported(
            "No complete verification; "
            + "; ".join(f"{a['backend']}: {a['detail']}" for a in session.attempts[-4:])
        )
        error.verification_attempts = session.attempts
        error.verification_schedule = session.phases
        raise error
    return goal.proposed, {
        "correctness_status": "verified",
        "cost": cost,
        "evidence": {
            **proof,
            "proof": copy.deepcopy(proof),
            "reference_sha256": reference_sha256,
            "goal_sha256": goal.sha256,
            "backend_attempts": session.attempts,
        },
        "verification_schedule": session.phases,
    }


def verify_gate_rewrites(reference, doc, reference_sha256, *, max_qubits=1024):
    """Direct certificate audit; runtime submission uses verify_entry's fallback."""
    proposed, stages = certificate_program(reference, doc, reference_sha256)
    goal = VerificationGoal(reference, proposed, reference_sha256, max_qubits, doc)
    evidence = prove_certificate(Session(60, exact_only=True), goal, stages, assurance="proved")
    cost = elementary_cost(proposed, reference["protected_wires"])
    if cost["qubits"] > max_qubits:
        raise Incorrect("Rewritten circuit exceeds the frozen qubit budget")
    return proposed, {"correctness_status": "verified", "cost": cost, "evidence": evidence}


def timed_verification(function, seconds):
    """Nestable deadline for all verification work; never a passing verdict."""
    import signal

    from qpe_lab.circuits.sim import Unsupported

    class ProofDeadline(Exception):
        pass

    def expired(signum, frame):
        raise ProofDeadline

    if seconds <= 0:
        raise Unsupported("Verification reached its reserved time budget")
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    old_delay, old_interval = signal.getitimer(signal.ITIMER_REAL)
    # An outer phase alarm retains its own exception, so inner fallback loops
    # cannot swallow it and continue after the phase has expired.
    owns_alarm = not old_delay or seconds < old_delay
    if owns_alarm:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return function()
    except ProofDeadline as exc:
        raise Unsupported("Verification reached its reserved time budget") from exc
    finally:
        if owns_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            if old_delay:
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(0.000001, old_delay - (time.monotonic() - started)),
                    old_interval,
                )
