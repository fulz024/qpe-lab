"""Freeze full toy and Fe2S2 SOS inputs; no new scientific dependencies needed."""

import argparse
import json
from pathlib import Path

import numpy as np

from qpe_lab.contracts import digest
from qpe_lab.tasks.data.hamiltonian import (
    centered_sos,
    make_model,
    read_fcidump,
    with_alias_preparation,
    with_paper_preparation,
    write_walk_task,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clifford-t", action="store_true")
    parser.add_argument("--walk-error-budget", type=float, default=1e-4)
    parser.add_argument("--objective", choices=("t_count", "tq_cost"), default="t_count")
    parser.add_argument(
        "--profile", choices=("reference", "alias", "paper", "paper-strict"), default="reference"
    )
    parser.add_argument("--alias-bits", type=int, default=11)
    parser.add_argument("--angle-bits", type=int, default=15)
    args = parser.parse_args()
    optimization = {
        "cost_model": "clifford_t" if args.clifford_t else "ideal",
        "walk_error_budget": args.walk_error_budget if args.clifford_t else 0.0,
        "objective": args.objective,
        "synthesis_backend": "pygridsynth-2.0.0",
        "ccx_decomposition": "7t-no-ancilla",
    }
    toy = make_model(
        np.array([[[0.6, 0.8], [0.8, -0.6]]]),
        np.array([[[0.7, -0.2], [-0.3, 0.5]]]),
        np.array([[0.25, -0.4]]),
        np.array([-0.35, 0.6]),
        np.array([[0.8, -0.6], [0.6, 0.8]]),
        energy_offset=-1.2,
        beta=5,
        coefficient_bits=12,
    )
    toy["source"] = {
        "kind": "deterministic synthetic example",
        "scope": "two spatial modes, both spins, SF plus creation/annihilation generators",
    }
    if args.profile == "paper-strict":
        toy = make_model(
            np.array([[[0.6, 0.8]]]),
            np.array([[[0.7]]]),
            np.array([[0.4]]),
            np.array([-0.35, 0.6]),
            np.array([[0.8, -0.6], [0.6, 0.8]]),
            energy_offset=-1.2,
            beta=2,
            coefficient_bits=8,
        )
        toy["source"] = {
            "kind": "small physical Appendix B verification fixture",
            "scope": "nonbinary outer range, binary inner range, both D1 and Q1",
        }
    if args.profile == "alias":
        toy = with_alias_preparation(toy, min(args.alias_bits, 3))
    elif args.profile == "paper":
        toy = with_paper_preparation(toy, min(args.alias_bits, 3))
    elif args.profile == "paper-strict":
        toy = with_paper_preparation(toy, 1, uniform_bits=3)
    subdirectory = "walk" if args.profile == "reference" else "walk-" + args.profile
    toy_path = write_walk_task(
        toy,
        ROOT / "benchmarks/dfthc/toy" / subdirectory,
        f"toy-dfthc-{args.profile}"
        if args.profile != "reference"
        else "toy-dfthc-controlled-sos-walk",
        optimization=optimization,
    )
    directory = ROOT / "benchmarks/dfthc/fe2s2"
    source = directory / "factors.json"
    factors = json.loads(source.read_text())
    fcidump = ROOT / "third_party/uncompute/tasks/chemistry-fe2s2/data/fcidump.xz"
    h, g, core, electrons = read_fcidump(fcidump)
    u, w, d, eig, vectors, diagnostics = centered_sos(factors, h, g, core, electrons)
    comparisons = {}
    for key in ("lambda_H1", "lambda_H2", "lambda_DFTHC", "H0", "error_h2", "error_h2_ord=1"):
        expected = factors["published_values"][key]
        # Archive optimized in float32; this reconstruction uses float64.
        error = abs(diagnostics[key] - expected)
        tolerance = 0.005 if key == "error_h2_ord=1" else 0.001
        comparisons[key] = {
            "computed": diagnostics[key],
            "published": expected,
            "abs_difference": error,
            "tolerance": tolerance,
        }
        if error > tolerance:
            raise ValueError(
                f"Official factor/integral reconstruction does not match {key}: {comparisons[key]}"
            )
    model = make_model(
        u,
        w,
        d,
        eig,
        vectors,
        energy_offset=diagnostics["energy_offset"],
        beta=args.angle_bits if args.profile in ("alias", "paper", "paper-strict") else 16,
    )
    model["source"] = {
        "factor_sha256": digest(source),
        "fcidump_compressed_sha256": digest(fcidump),
        "factor_source_sha256": factors["source_sha256"],
        "source_url": factors["source_url"],
        "electrons": electrons,
        "definition": "centered K=E-I; symmetry shift (Nhat-Ne)*(H1_I+Hsym:K/2)",
        "diagnostics": diagnostics,
        "published_comparisons": comparisons,
    }
    if args.profile == "alias":
        model = with_alias_preparation(model, args.alias_bits)
        model["source"]["profile"] = (
            "alias/QROAM/phase-gradient workbench target; expanded generator indexing; "
            "not a claim of identical Table V finite-precision tensors or resource counts"
        )
    elif args.profile == "paper":
        model = with_paper_preparation(model, args.alias_bits)
        model["source"]["profile"] = (
            "Appendix B compact gate circuit; deterministic rounding; "
            "not the publication's stochastic finite-precision tensors"
        )
    elif args.profile == "paper-strict":
        model = with_paper_preparation(model, args.alias_bits, uniform_bits=7)
        model["source"]["profile"] = (
            "Appendix B flagged preparation, shared QROAM and measured erasure schedule; "
            "deterministic finite target; author stochastic sample not identified"
        )
    real_path = write_walk_task(
        model,
        directory / subdirectory,
        (
            f"fe2s2-dfthc-{args.profile}-a{args.alias_bits}-b{args.angle_bits}"
            if args.profile in ("alias", "paper", "paper-strict")
            else "fe2s2-dfthc-controlled-sos-walk-b16-c24"
        ),
        optimization=optimization,
    )
    print(
        json.dumps(
            {"toy": str(toy_path), "fe2s2": str(real_path), "published_comparisons": comparisons},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
