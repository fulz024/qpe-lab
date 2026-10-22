"""Aggregate actual gate resources and calculate the training reward."""

from qpe_lab.circuits.checks import elementary_cost


def training_reward(status, cost, policy):
    """The verified artifact's TQ is the sole reward input, independent of method."""
    if status not in {"verified", "exact_verified", "bounded_verified"}:
        return 0.0
    if cost.get("status") != "complete":
        raise ValueError("Reward requires complete actual-gate resource counts")
    c0, current = policy["C0"], cost["tq_cost"]
    if type(c0) is not int or c0 <= 0 or type(current) is not int or current < 0:
        raise ValueError("Invalid TQ normalization")
    return c0 / (c0 + current)


def circuit_pair_cost(entries):
    """Two sequential calls sharing named physical lines; classical IDs are local."""
    from collections import Counter

    costs = {name: elementary_cost(e["gates"], e["protected_wires"]) for name, e in entries.items()}
    physical = sorted({q for cost in costs.values() for q in cost["physical_wires"]})
    counts = Counter()
    for cost in costs.values():
        counts.update(cost["compiled_elementary_gates"])
    result = {
        key: sum(cost[key] for cost in costs.values())
        for key in ("t_count", "operations", "clifford", "measurements", "classical_xor")
    }
    return result | {
        "status": "complete",
        "qubits": len(physical),
        "physical_wires": physical,
        "tq_cost": result["t_count"] * len(physical),
        "compiled_elementary_gates": dict(sorted(counts.items())),
        "entries": costs,
        "scope": "one forward and one adjoint core call; shared physical wire pool",
    }


def evaluation_feedback(report, baseline, best=None):
    """Explain a trainer verdict; comparisons never change acceptance or reward."""
    status = report.get("correctness_status", "error")
    cost = report.get("cost", {})
    accepted = status in {"verified", "exact_verified", "bounded_verified"}
    accepted = accepted and cost.get("status") == "complete"
    kinds = {
        "fail": ("incorrect", "产物或电路不满足合同；依据 detail 修正后再提交。"),
        "unsupported": (
            "verification_incomplete",
            "尚未完成验证；可继续整体改写，或提供可组合的局部变换证据。",
        ),
        "timeout": ("deadline", "本次生成或验证超时；查看阶段诊断，缩减计算或拆分证明目标。"),
    }
    kind, advice = kinds.get(status, ("execution_error", "查看 detail 中的程序或环境错误。"))
    methods, claims, bounds = set(), set(), {}

    def visit(node):
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            if isinstance(node.get("method"), str):
                methods.add(node["method"])
            if node.get("backend") == "sampled":
                claims.add(node.get("claim", "unspecified"))
                bounds[node["goal_sha256"]] = node.get("false_accept_bound", 0)
            # proof is a duplicate of the root evidence in runtime reports.
            for key, child in node.items():
                if key not in {"proof", "backend_attempts"}:
                    visit(child)

    visit(report.get("evidence", {}))
    metrics = (
        "t_count",
        "qubits",
        "tq_cost",
        "clifford",
        "measurements",
        "classical_xor",
        "operations",
    )
    compared = best or baseline
    result = {
        "category": "accepted" if accepted else kind,
        "scope": report.get("scope"),
        "improved_reference": accepted and cost["tq_cost"] < baseline["tq_cost"],
        "improved_best": accepted and cost["tq_cost"] < compared["tq_cost"],
        "comparison_target": "session_best" if best else "frozen_reference",
        "verification": {
            "methods": sorted(methods),
            "claims": sorted(claims),
            "sampled": bool(report.get("evidence", {}).get("sampling", False)),
            "false_accept_bound_sum": sum(bounds.values()),
            "scientific_review": report.get("scientific_review", {}).get("status", "not_run"),
        },
        "resource_changes": {
            k: {"actual": cost[k], "reference": baseline[k], "delta": cost[k] - baseline[k]}
            for k in metrics
            if k in cost and k in baseline
        },
        "reward_meaning": "T/T† × 共用物理比特池；不等同于 Toffoli 数、完整 QPE 成本或硬件运行时间。",
    }
    if accepted:
        advice = (
            "验证通过，并降低了比较目标的 TQ；可继续优化。"
            if result["improved_best"]
            else "验证通过；有奖励不等于超过已有最佳，本次未降低比较目标的 TQ。"
        )
    result["next_action"] = advice
    execution_failure = report.get("failure_stage") in {"generation", "freezing"} and status in {
        "error",
        "timeout",
    }
    resolved = accepted or status == "fail"
    result["training_signal"] = {
        "outcome": "accepted"
        if accepted
        else "rejected"
        if resolved or execution_failure
        else "unresolved",
        "correctness_target": int(accepted) if resolved else None,
        "correctness_mask": resolved,
        # Unfinished verification has no correctness label, but still has its
        # actual task reward. Dropping such rows would bias RL toward timeouts.
        "reward_target": report.get("reward", 0.0) if accepted else 0.0,
        "reward_mask": True,
        "counts_as_attempt": True,
        "deferred_eligible": bool(
            not resolved
            and not execution_failure
            and report.get("h_ops")
            and status in {"unsupported", "timeout", "error"}
            and not report.get("completion")
        ),
    }
    return result
