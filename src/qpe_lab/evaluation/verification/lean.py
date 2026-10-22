"""Lean comparator backend for small, phase-sensitive monomial circuits.

The trusted translation represents |x> -> omega**p |y> by (y,p mod 8).
Equality for every allowed basis input preserves the linear extension, including
coherence with external systems. H, measurement, assertions and feedback require
other semantics and are deliberately outside this first backend's domain.
"""

import hashlib
import json
import tempfile
from pathlib import Path

from qpe_lab.circuits.sim import Unsupported
from qpe_lab.contracts import VerificationResult, project_root
from qpe_lab.evaluation.sandbox import minimal_env, run_job

VERSION = "4.35.0"
PERMITTED_AXIOMS = ["propext", "Quot.sound", "Classical.choice"]


def toolchain():
    return project_root() / ".tools" / f"lean-{VERSION}-linux"


def problem(goal, proof="by sorry"):
    """Generate the statement AND its definitions; the solver supplies one term."""
    if goal.semantics != "qpe.phase-sensitive-instrument.v1":
        raise Unsupported("Lean monomial backend does not cover this equivalence relation")
    before, after, wires = goal.reference["gates"], goal.proposed, goal.wires
    allowed = {"x", "cx", "cz", "z", "s", "sdg", "t", "tdg", "neg"}
    if (
        len(wires) > 8
        or max(len(before), len(after)) > 256
        or goal.external_records
        or goal.live_records
        or any(n not in allowed or v is not None for n, _, v in before + after)
    ):
        raise Unsupported("Lean monomial domain: at most 8 wires / 256 unconditional gates")
    mapping = {q: i for i, q in enumerate(wires)}

    def word(gates):
        names = {"x": 0, "cx": 1, "cz": 2, "z": 3, "s": 4, "sdg": 5, "t": 6, "tdg": 7, "neg": 8}
        return (
            "["
            + ", ".join(
                f"({names[n]}, {mapping[qs[0]] if qs else 0}, {mapping[qs[1]] if len(qs) > 1 else 0})"
                for n, qs, _ in gates
            )
            + "]"
        )

    columns = goal.columns
    return f"""import Init
set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace QPE
-- The second component is the exponent of exp(i*pi/4), modulo eight.
abbrev State := Nat × Nat
def bit (x q : Nat) : Bool := (x / 2^q) % 2 == 1
def phase (s : State) (p : Nat) : State := (s.1, (s.2 + p) % 8)
def step (g : Nat × Nat × Nat) (s : State) : State :=
  let (tag, a, b) := g
  match tag with
  | 0 => (Nat.xor s.1 (2^a), s.2)
  | 1 => if bit s.1 a then (Nat.xor s.1 (2^b), s.2) else s
  | 2 => if bit s.1 a && bit s.1 b then phase s 4 else s
  | 3 => if bit s.1 a then phase s 4 else s
  | 4 => if bit s.1 a then phase s 2 else s
  | 5 => if bit s.1 a then phase s 6 else s
  | 6 => if bit s.1 a then phase s 1 else s
  | 7 => if bit s.1 a then phase s 7 else s
  | _ => phase s 4
def run (gates : List (Nat × Nat × Nat)) (x : Nat) : State :=
  gates.foldl (fun s g => step g s) (x, 0)
def reference : List (Nat × Nat × Nat) := {word(before)}
def candidate : List (Nat × Nat × Nat) := {word(after)}
def allowed (x : Nat) : Bool := {columns!r}.contains x
theorem circuit_equivalent : ∀ x : Fin {1 << len(wires)},
    allowed x.val = true → run reference x.val = run candidate x.val :=
  {proof}
end QPE
"""


def verify(goal, context=None):
    if not goal.proof or goal.proof.get("backend") != "lean":
        return VerificationResult(
            "unknown", goal.sha256, detail="No Lean proof attached", reason="unsupported"
        )
    try:
        challenge = problem(goal)
        solution = problem(goal, goal.proof["source"])
    except Unsupported as exc:
        return VerificationResult("unknown", goal.sha256, detail=str(exc), reason="unsupported")
    runtime = toolchain()
    if not (runtime / "bin/lake").is_file():
        return VerificationResult(
            "unknown", goal.sha256, detail="Lean 4.35.0 is not installed", reason="unsupported"
        )
    with tempfile.TemporaryDirectory(prefix="qpe-lean-") as tmp:
        folder = Path(tmp)
        (folder / ".lake").mkdir()
        wrapper = folder / "sandbox"
        wrapper.write_text("""#!/usr/bin/python3 -I
import os, sys
args, i = [], 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "--tmpfs":
        args += ["--size", "134217728", arg, sys.argv[i+1]]
        i += 2
    elif arg == "--dev":
        args += ["--dev-bind", "/dev", sys.argv[i+1]]
        i += 2
    elif arg == "--":
        args += ["--remount-ro", "/dev", "--setenv", "LEAN_STACK_SIZE_KB", "16384", *sys.argv[i:]]
        break
    else:
        args.append(arg)
        i += 1
os.execv("/opt/bwrap", ["/opt/bwrap", *args])
""")
        wrapper.chmod(0o755)
        (folder / "Challenge.lean").write_text(challenge)
        (folder / "Solution.lean").write_text(solution)
        (folder / "lakefile.toml").write_text(
            'name = "qpe_proof"\nmoreLeanArgs = ["-j1", "-s16384"]\n[[lean_lib]]\nname = "Challenge"\n'
            '[[lean_lib]]\nname = "Solution"\n'
        )
        (folder / "lake-manifest.json").write_text(
            json.dumps(
                {
                    "version": "1.1.0",
                    "packagesDir": ".lake/packages",
                    "packages": [],
                    "name": "qpe_proof",
                    "lakeDir": ".lake",
                }
            )
        )
        (folder / "comparator.json").write_text(
            json.dumps(
                {
                    "challenge_module": "Challenge",
                    "solution_module": "Solution",
                    "theorem_names": ["QPE.circuit_equivalent"],
                    "permitted_axioms": PERMITTED_AXIOMS,
                }
            )
        )
        # The official comparator checks statement identity, axiom dependencies
        # and replays serialized proofs in the kernel. The outer sandbox hides
        # task files, ledgers and credentials even from arbitrary Lean tactics.
        bwrap = project_root() / ".tools/bubblewrap-0.11.0/bwrap"
        command = [
            str(bwrap),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--cap-drop",
            "ALL",
        ]
        for path in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(path).exists():
                command += ["--ro-bind", path, path]
        for path in ("/home", "/root", "/run/user", "/var"):
            command += ["--dir", path]
        command += [
            "--ro-bind",
            str(runtime),
            "/opt/lean",
            "--ro-bind",
            str(bwrap),
            "/opt/bwrap",
            "--ro-bind",
            str(folder),
            "/proof",
            "--size",
            str(128 * 1024**2),
            "--tmpfs",
            "/proof/.lake",
            "--size",
            str(128 * 1024**2),
            "--tmpfs",
            "/tmp",
            "--chdir",
            "/proof",
            "--remount-ro",
            "/dev",
            "/usr/bin/prlimit",
            "--nproc=128",
            "--cpu=60",
            "--",
            "/opt/lean/bin/lake",
            "comparator",
        ]
        env = minimal_env() | {
            "PATH": "/opt/lean/bin:/usr/bin:/bin",
            "COMPARATOR_BWRAP": "/proof/sandbox",
            "LEAN_NUM_THREADS": "1",
            "LEAN_STACK_SIZE_KB": "16384",
        }
        job = run_job(command, folder, folder / "check.log", 45, env)
        output = (folder / "check.log").read_text(errors="replace")
        if job["timed_out"]:
            return VerificationResult(
                "unknown", goal.sha256, detail="Lean proof check timed out", reason="timeout"
            )
        if job["returncode"] != 0:
            # A bad proof term is not evidence that the circuit is wrong.
            return VerificationResult(
                "unknown" if job["returncode"] == 1 else "error",
                goal.sha256,
                detail=output[-4000:],
                reason="unsupported" if job["returncode"] == 1 else "tool_error",
            )
        return VerificationResult(
            "proved",
            goal.sha256,
            {
                "method": "Lean kernel and comparator",
                "toolchain": VERSION,
                "theorem": "QPE.circuit_equivalent",
                "permitted_axioms": PERMITTED_AXIOMS,
                "challenge_sha256": hashlib.sha256(challenge.encode()).hexdigest(),
                "solution_sha256": hashlib.sha256(solution.encode()).hexdigest(),
                "domain": "phase-sensitive monomial linear map",
                "process": job,
            },
        )
