"""Deliver one frozen task, its gate DSL and submission context to a launcher."""

import json
import shlex
import sys
from pathlib import Path

from qpe_lab.circuits.ops import GATE_SCHEMA, read_reference
from qpe_lab.contracts import digest, project_root
from qpe_lab.tasks.bundle import load_training


def dsl_reference():
    rows = [
        "门记录为 `(name, wires, value)`；线号从零开始。",
        "",
        "| name | 线数 | 含义 |",
        "|---|---:|---|",
    ]
    rows += [
        f"| `{name}` | {arity} | {meaning} |" for name, (arity, meaning) in GATE_SCHEMA.items()
    ]
    rows += [
        "",
        "量子门的 value 为 None，或 `{'condition': id}` 表示经典记录为 1 时执行。",
        "hmr、oxor、assert_zero 使用表中的专用格式；assert_zero 的 value 必须为 None。",
        "Toffoli 等宏展开到上述基本门。测量记录在单个入口内定义，使用前必须已定义。",
    ]
    return "\n".join(rows)


def circuit_primer():
    """Small, executable identities; no chemistry prerequisite or claimed proof."""
    return (
        "把电路看成按顺序作用在复数振幅上的程序。每条线不是一个固定的0/1，"
        "它可以处于叠加态并与其他线相关。正确性要求输出振幅和相位符合合同。\n"
        "X交换|0>与|1>；Z给|1>乘-1；H把|0>变为(|0>+|1>)/sqrt(2)，"
        "把|1>变为(|0>-|1>)/sqrt(2)。T给|1>乘exp(i*pi/4)，T†乘其逆。\n"
        "例一：同一条线上的相邻T、T†可删除，任意输入上的作用都不变。\n"
        "例二：相邻T、T不能直接删除：它们虽然不改变基态的0/1值，"
        "却给|1>乘i，会改变叠加态。它们可以合并为S。\n"
        "门只在证明可交换时才可调换顺序；资源态的特殊性质要读取合同。"
        "hmr是测量并复位，assert_zero只检查是否为零。\n"
        "reward衡量通过验证后的TQ；原参考也有0.5分。比较feedback.improved_best，"
        "并保留已通过的最佳程序。unsupported表示验证未完成，不是数学反例。"
        "反馈中的training_signal区分已通过、已否决和未完成；未完成仍消耗提交预算。"
    )


def agent_task(bundle, expected_sha256, *, workspace, budget, session=False):
    """Read-only task delivery. The launcher supplies permissions and starts the model."""
    if not isinstance(budget, str) or not budget.strip():
        raise ValueError("The launcher must specify the iteration budget and stopping condition")
    folder, task, manifest = load_training(bundle, expected_sha256)
    reference = read_reference(folder / "reference.json")
    reference_path = Path("/reference.json") if session else folder / "reference.json"
    visible_root = Path("/app") if session else project_root()
    entries = reference.get("entries", {"walk": reference})
    contract = reference.get(
        "contract",
        {
            "scope": "complete controlled walk",
            "registers": {"protected": reference.get("protected_wires", [])},
        },
    )
    command = [
        sys.executable,
        "-m",
        "qpe_lab",
        "training-worker",
        "--bundle",
        str(folder),
        "--bundle-sha256",
        expected_sha256,
    ]
    context = {
        "schema": "qpe.agent-task.v1",
        "task": task.name,
        "scope": manifest["scope"],
        "bundle": "trainer-managed" if session else str(folder),
        "bundle_sha256": expected_sha256,
        "reference": str(reference_path),
        "reference_sha256": manifest["files"]["reference.json"],
        "contract": contract,
        "entries": {
            name: {k: entry[k] for k in ("ancilla_initial", "protected_wires")}
            for name, entry in entries.items()
        },
        "baseline_cost": {
            k: manifest["baseline_cost"][k] for k in ("t_count", "qubits", "tq_cost")
        },
        "reward": manifest["reward"],
        "limits": manifest["limits"],
        "verification": manifest["verification"],
        "workspace": "/workspace" if session else str(Path(workspace).resolve()),
        "budget": budget,
        "resources": {
            name: str(visible_root / path)
            for name, path in {
                "generator_interface": "docs/CIRCUIT_GENERATORS.md",
                "architecture": "docs/DFTHC_ARCHITECTURE.md",
                "example_generator": "candidates/dfthc_walk/reuse.py",
                "gadgets": "src/qpe_lab/circuits/gadgets",
            }.items()
        },
        "submission": {
            "command": command,
            "cwd": str(project_root()),
            "request": {"source": "Python 电路生成器源码"},
        },
    }
    if session:
        context["submission"] = {"request": {"action": "submit", "source": "Python 电路生成器源码"}}
    general = project_root() / "AGENTS.md"
    scope = (
        "核心 K 与 K†，分别提交 forward 和 adjoint"
        if manifest["scope"] == "core"
        else "完整受控 walk，提交 walk"
    )
    specific = (
        f"本轮任务：{task.name}。范围：{scope}。\n"
        f"参考：{reference_path}；SHA-256：{context['reference_sha256']}。\n"
        f"可写候选目录：{context['workspace']}。预算与停止条件：{budget}。\n"
        f"参考成本：{json.dumps(context['baseline_cost'])}。通过后 reward=C0/(C0+TQ)，C0={manifest['reward']['C0']}。\n"
        f"本轮启用验证后端：{', '.join(manifest['verification']['backends'])}。\n"
        f"本地阅读入口（绝对路径）：{json.dumps(context['resources'], ensure_ascii=False)}。\n"
        "任务合同与入口输入条件见下面的 JSON；本轮信息优先于通用说明中的默认值和历史快照。\n"
        + json.dumps(
            {"contract": contract, "entries": context["entries"]}, ensure_ascii=False, indent=2
        )
        + "\n\nDSL：\n"
        + dsl_reference()
        + "\n\n最小电路入门：\n"
        + circuit_primer()
        + "\n\n提交自包含 Python 程序。程序使用评估器注入的参考和输出路径：\n"
        "```python\nfrom qpe_lab import CircuitBuilder\nb = CircuitBuilder()\nb.emit()\n```\n"
        "独立重写时，对每个入口调用 b.select(name).submit(gates)，最后 b.emit()。\n"
        + (
            "本轮要求 Lean 证明；使用 b.attach_lean(proof_term) 附加证明，其他后端不能替代。\n"
            if manifest["verification"]["acceptance"] == "lean_required"
            else "证据可选。第二级验证通过即可获得奖励；第三级科学复核由维护方另行完成。\n"
        )
        + (
            "本轮所有文件和执行操作通过会话工具：task、read、write、experiment、submit、best。\n"
            '例如 {"action":"read","path":"/app/docs/CIRCUIT_GENERATORS.md"}；'
            '{"action":"submit","path":"/workspace/generator.py"}。\n'
            "experiment执行Python并返回输出，不产生正式reward；submit才作验收。"
            "会话自动保存最佳产物；通过best取得程序和报告。预算由服务执行。\n"
            "等待期间的progress事件说明当前阶段、已用时间和剩余预算，不是验收结果。"
            "超时或验证未完成的冻结产物可由维护方追加一次有上限的验证；"
            "agent不能修改预算或通过重启获得免费重试。\n"
            if session
            else '评估服务每行接收 {"source":"Python 源码"}，返回判定、级别、成本、reward 和运行记录。\n'
            f"服务启动命令：{shlex.join(command)}\n"
            f"服务工作目录：{project_root()}。由维护方启动后，通过其标准输入／输出提交；\n"
            "生成器失败时 detail 包含末尾错误信息。预算由启动器执行，任务 JSON 不创建目录或授予宿主权限。\n"
        )
    )
    context["instructions_sha256"] = digest(general)
    context["system_prompt"] = (
        general.read_text(encoding="utf-8") + "\n\n## 本轮冻结任务\n\n" + specific
    )
    return context
