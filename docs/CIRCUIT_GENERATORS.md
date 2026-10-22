# Python 电路生成器接口

主入口和安装步骤见 [README](../README.md)，优化 agent 的任务说明见 [AGENTS.md](../AGENTS.md)。本文集中说明提交协议、可选证据与组件任务；物理合同和验证容量见 [验证架构](DFTHC_ARCHITECTURE.md)。

## 主任务：完整门流

生成器读取评估环境注入的冻结参考，输出 `QPEGEN01` 产物。核心任务必须同时包含 `forward` 和 `adjoint`；完整 walk 任务只有 `walk` 入口。最小程序提交原参考：

```python
from qpe_lab import CircuitBuilder

b = CircuitBuilder()
b.emit()
```

独立构造时，分别调用 `b.select("forward").submit(forward_gates)` 和 `b.select("adjoint").submit(adjoint_gates)`，最后 `b.emit()`。门记录为 `(name, wires, value)`，例如 `("t", (3,), None)`。`b.contract` 提供寄存器与借用态约定，`load_task()` 读取固定 Hamiltonian。也可直接输出符合协议的 JSON，不必使用构造器。

`submit()` 整体替换所选入口，不要求沿用参考结构或提供改写历史。含测量程序的伴随单独构造。参考路径由 `QPE_REFERENCE_PATH` 注入，产物写入 `UNCOMPUTE_OPS_PATH`，均由评估环境设置。

可以提交大型相干整体改写，以及利用冻结梯度态的改写。例如梯度最高位 `g` 为 `|−⟩` 时，`CX(a,g)` 的逻辑作用是 `Z(a)`；若前缀保持该资源态，展开的 `CCX(a,b,g)` 可换成 `CZ(a,b)`。框架从合同和参考前缀证明输入条件，不需要候选附加资源态声明。所有工作位和梯度输出都参与验证。整体张量路径按中间收缩规模设限，不要求按参考组件分块；其具体容量与统计含义见架构文档。

## 可选证明证据

| 方法 | 证据内容 |
|---|---|
| `b.replace(edits)` | 无条件 Clifford+T 小块替换 |
| `b.replace(edits, proof="zx")` | 较大无条件量子块的 ZX 等价 |
| `b.replace(edits, proof="dynamic")` | 测量与反馈块的逻辑信道 |
| `b.replace(edits, proof="boolean")` | 宽可逆逻辑、相位与测量清理 |
| `b.phases(gates)` | 完整门流的相位搬移／合并 |
| `b.reuse(moves, gates)` | 工作位分配、释放和实际线号复用 |
| `b.attach_lean(proof_term, ...)` | 整个入口或指定区间的 Lean 证明项 |

`replace` 的范围是当前门流的 `[start, stop)`，同阶段范围不重叠。`edits` 中每项包含 `start`、`stop`、`gates`。证据由裁判独立核验，不是授予通过状态的声明；不提供证据时仍可走完整精确／抽样验证。

`replace` 覆盖一个非空原区间。插入门时可连同邻近原门一起替换，或用 `submit` 直接交付新门流；删除区间则令 `gates=[]`。这不限制整体提交中的插入、删除或跨组件改写。

现有最佳生成器采用：

```python
from qpe_lab import CircuitBuilder
from qpe_lab.circuits.gadgets.synthesis import constant_control_edits, reuse_clean_qubits

b = CircuitBuilder()
for name in b.entries:
    b.select(name)
    b.replace(constant_control_edits(b.gates))
    b.reuse(*reuse_clean_qubits(b.gates, b.protected_wires))
b.emit()
```

`moves` 每项为 `[position, logical_wire, physical_wire]`，物理位 `-1` 表示释放。Agent 可以自行实现分配器。释放时必须已清零，`assert_zero` 不建立零状态，也不执行复位。两入口共用物理线分配池，经典测量编号则在各入口内独立。

### Lean 附件

维护方用 `scripts/setup-lab.ps1 -Lean` 安装固定的 Lean 4.35.0 工具链。冻结新 bundle 时，`freeze-training --enable-lean` 将 Lean 加入可选后端；`--acceptance lean_required` 会自动启用 Lean 并要求证明。默认 bundle 使用现有精确／抽样后端。启用列表与工具链内容一同冻结，生成器不能自行启用工具。

生成器先提交实际门流，再附加证明项：

```python
b.submit(gates).attach_lean("by decide")
# 或给出冻结参考与实际候选中的一对区间：
b.attach_lean("by decide", reference_range=[a, b], candidate_range=[c, d])
b.emit()
```

范围都是半开区间，参考坐标相对于冻结参考，候选坐标相对于最终提交门流。可附多个不重复的目标；每个入口至多 128 个附件，每个证明项至多 64 KiB。框架根据实际门流及前缀推导条件生成定理，证明项不能额外声明输入假设。可用维护方的 `evaluation.verification.lean.problem(goal)` 查看命题源码。当前 Lean 支持 8 线／256 门以内的无条件可逆和相位门，包含整体负号；具体范围见架构文档。

产物仍只有一个 `QPEGEN01` 文件。每个入口可含 `proofs` 列表，每项为 `{"backend":"lean","source":"by decide","reference_range":null,"candidate_range":null}`；非空范围使用两个整数的列表。无效或未完成的证明会保留诊断，默认任务仍可尝试其他后端。

## 运行与反馈

启动器用 `agent-task` 取得本轮完整任务 JSON，参数和示例见 [接入 agent](../README.md#接入-agent)。`system_prompt` 合并通用工作约定与冻结任务；`contract`、`entries`、`reference`、`baseline_cost`、`submission` 可供启动器直接读取。`resources` 给出接口文档、架构、示例和构件目录的绝对路径，从独立候选工作区也能找到。支持门表由 `circuits/ops.py` 的 `GATE_SCHEMA` 生成，与产物解析器共用定义。

命令行使用 `evaluate-training --bundle ... --bundle-sha256 ... --candidate ...`；训练服务每行只接收 `{"source":"Python 源码"}`。生成器不接收额外命令行参数，候选参数固定在源码内。

命令行入口会保存同目录顶层 `.py` 辅助文件，最多 32 个、每个 1 MiB；不自动收集子包或附带数据。服务入口只接收一个源码字段，采用自包含脚本。依赖限于已安装环境，固定数据来自任务。

训练生成器在无网络的隔离环境中运行，只能写 `circuit.ops` 和标准输出／错误日志；临时目录只读，可在内存中构造电路。生成与验证共享 600 秒期限。裁判独立读取冻结门流并统计成本；stdout 不能声明判定或成绩。生成器退出失败时，服务的 `detail` 返回日志末尾至多 4,000 字节的错误信息，完整日志仍在运行目录。

第二级 **验证通过**即可获得奖励、保留候选并继续优化。第三级 **科学复核通过**由维护方追加，奖励不变。每次运行在 `runs/` 保存源码、门流、日志、证据和 `report.json`；报告包含判定、级别、T/Q/TQ、reward、摘要及诊断。详细验证路径、统计含义和上限集中在 [验证架构](DFTHC_ARCHITECTURE.md)。

`feedback` 给出 `category`、`improved_reference`、`improved_best`、`comparison_target`、验证方法和完整映射/信道误收界，以及实际 T/Q/TQ、Clifford、测量和经典 XOR 相对参考的变化。`incorrect` 是产物或电路不满足合同；`verification_incomplete` 是验证未完成；`deadline` 是超时；`execution_error` 是程序或环境错误。没有发现反例但验证未完成时，可以继续整体重写或提供变换证据，不能据此断言电路错误。

## 会话工具接口

`agent-session` 是维护方启动的 JSONL 工具服务。模型启动器取首行 `task.system_prompt` 作为系统提示，注册一个转发 JSON 请求的工具即可；不需要了解 DFTHC。这里的 Python 子进程是工具服务，不是模型：

```python
import json
import subprocess

service = subprocess.Popen(
    [".venv-lab/bin/python", "-m", "qpe_lab", "agent-session",
     "--session", "runs/sessions/demo", "--max-submissions", "20", "--seconds", "1800",
     "--progress-events"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
)

def receive():
    while True:
        line = service.stdout.readline()
        if not line:
            raise RuntimeError("QPE service exited before its final response")
        message = json.loads(line)
        if message.get("event") == "progress":
            # Display or forward to the UI; this is not the tool's final verdict.
            print("QPE progress:", message)
            continue
        return message

ready = receive()
system_prompt = ready["task"]["system_prompt"]

def qpe_tool(request):
    # Serialize calls to this function; one reply corresponds to one request.
    service.stdin.write(json.dumps(request) + "\n")
    service.stdin.flush()
    return receive()

# The external model uses system_prompt and this tool, without host file/shell tools.
print(qpe_tool({"action": "best"}))
```

| action | 参数与结果 |
|---|---|
| `task` | 返回当前任务、起点、最小电路语义说明和工具格式 |
| `read` | `path`，可选 `offset`/`limit`；读取文件或目录，支持分页 |
| `write` | `path`、`text`；写入 `/workspace` 下的文本文件 |
| `experiment` | `source`、可选 `timeout`（默认 30 秒，最多 60 秒）；隔离执行 Python，返回末尾 32 KiB 输出；`reward=null` |
| `submit` | `source` 或 `/workspace` 中的 `.py` 文件 `path`，二者选一；返回正式评估及与最佳的比较 |
| `best` | 最佳已通过生成器的文件快照、入口文件名、绑定摘要及报告 |

每个响应还返回剩余时间、提交和实验次数。源码/请求最多 1 MiB；工作区最多 128 个文件、8 MiB。文件提交遵循原有同目录最多 32 个 `.py` 辅助文件的规则；`source` 提交为自包含程序。每次会话允许至多四倍提交数的实验，以及至少 100 次、通常为提交数 20 倍的写入/执行工具调用；读取、查看任务和取回最佳不消耗这些计数。实验不消耗正式提交次数，两者共享会话时间。

响应的 `ok` 只表示工具请求完成；电路是否通过看 `result.correctness_status`，是否改善看 `result.feedback.improved_best`。实验中的打印文本不作为任何判定。

`--progress-events` 也适用于 `training-worker`、`evaluate-training` 和 `complete-verification`。事件可能出现在启动 `ready` 之前及提交完成之前，启动器读取到最终响应才结束调用；不必发起新的提交来查询进度。省略该选项时，事件写到 stderr，stdout 保持原有最终响应格式；若启动器将 stderr 接入管道，也要持续消费该管道。事件字段包括 `stage`、可用时的 `entry/phase/backend`、`elapsed_seconds`、`remaining_seconds`、阶段预算及阶段已用时间。剩余时间是上限，不是预计完成时间。生成、实验和验证进程等待期间约每两秒更新一次，短阶段可能只出现在最终 `verification_schedule` 中。

最终 `timing` 给出端到端时间、生成与验证进程耗时、工作预算和最后观察到的阶段；`verification_schedule` 给出各入口每个阶段的预算、耗时和完成情况。采用的后端及尝试耗时保存在运行报告的证据中。启动参考检查不占搜索会话预算；开始搜索后，生成、实验和验证均占会话时间。

`feedback.training_signal` 用于把验证结论接入训练：`outcome` 为 `accepted`、`rejected` 或 `unresolved`；`correctness_target` 在通过时为 1、明确否决时为 0、未完成时为 null，`correctness_mask` 指明是否有此监督标签。生成程序失败也有零奖励，但不被伪装成量子电路的数学反例。`counts_as_attempt` 始终为 true，`deferred_eligible` 指明维护方是否可以补验。所有未通过提交的当前 reward 仍为零，最佳产物只从验证通过的记录中选取。

`reward_target` 保留当前实际任务得分，`reward_mask` 为 true，包括未完成时的零分；它与用于数学正确性监督的 `correctness_mask` 分开。训练器可以等补验完成后再更新样本，但应保留未完成样本及其已消耗预算，不把它们从 RL 批次、组内比较或成功率分母中直接删除。补验通过后更新原样本得分；仍未完成时保留零任务奖励与空正确性标签。会话执行预算，具体训练损失和延迟更新由外部训练器实现。

维护方用 `complete-verification --bundle ... --bundle-sha256 ... --run ...` 对符合条件的冻结产物追加一次最长 1,800 秒的验收。源码和门流必须与原提交摘要一致，生成器不再执行；新挑战占用连续账本中的下一次统计预算，不能重复使用旧抽样结论。补验预约持久化，重启或重复命令不会增加次数。补验结果更新原 `run/report.json`，原报告保存在 `completion/original-report.json`；训练器按同一 run 更新原样本，不把补验算成一个新的成功候选。它可以通过精确或完整映射／信道抽样得到第二级“验证通过”，不自动获得第三级“科学复核通过”。

实验可以读取 `/workspace`、`/reference.json`、`/task`、`/app/src`、`/app/docs` 和已有构造器；使用已安装的 Python 库，在内存中进行搜索和模拟。实验使用单进程隔离、4 GiB 地址空间上限和输出上限，与生成器一致。实验里的工作区只读；修改文件通过 `write`，由服务统一控制容量。打印的“得分”只作为实验输出文本，不影响正式结果。

服务的预算、候选快照和最佳记录保存在工作区外。启动时自动验收参考和初始候选，准备时间不占本轮预算。重启同一路径保持原限额和截止时间，失败提交也计次；超时或次数耗尽后仍可取回成果。每次 `best` 从实际验收报告选择最低 TQ，并核对源码、门流摘要；科学复核已否决的结果不再作为最佳。连续训练的统计账本跨会话保留。

## 不懂量子计算时的第一个实验

门按程序顺序作用在复数振幅上，不能仅比较输出的 0/1。下面的独立小实验可直接放进 `experiment.source`；三个结果依次为 `proved`、`counterexample`、`proved`：

```python
from qpe_lab.contracts import VerificationGoal
from qpe_lab.evaluation.verification.exact import verify

t, tdg, s = ("t", (0,), None), ("tdg", (0,), None), ("s", (0,), None)
for before, after in [([t, tdg], []), ([t, t], []), ([t, t], [s])]:
    ref = {"gates": before, "protected_wires": [0], "ancilla_initial": "zero"}
    print(verify(VerificationGoal(ref, after, "local-example", 8)).status)
```

T 与 T† 相消，所以可以删除；两个 T 给 `|1⟩` 乘以 i，删除会损坏相位，但合并为 S 正确。局部实验帮助寻找变换，正式收益仍由整个生成器的提交决定。`assert_zero` 不等于复位，含测量的程序也不能简单倒放作为伴随。

## 候选目录

`candidates/` 保留可复用的交付程序，日常搜索候选放在指定工作区或 `runs/candidates/`。

| 文件（相对于 `candidates/`） | 用途 |
|---|---|
| `dfthc_walk/optimized.py` | 构造当前主任务的冻结参考 |
| `dfthc_walk/reuse.py` | 当前最佳完整门流生成器，常量控制＋辅助位复用 |
| `dfthc_walk/constant_control.py` | 只做常量控制简化，用于对照 |
| `dfthc_walk/paper.py` | 论文路线的结构化门级构造 |
| `dfthc_walk/baseline.py`、`grouped.py` | 理想旋转 walk 的未分组／分组组件基线 |
| `dfthc_walk/alias_qroam.py` | 原 uncompute 查表／测量擦除程序嵌入 walk 的示例，使用 alias profile |
| `fe2s2_angles/{unary,selectswap}.py` | 独立角度查表的两种构造 |
| `fe2s2_rotation/{unary,selectswap}.py` | 加载、固定旋转和卸载的两种构造 |

## 组件任务

主训练使用上述完整门流接口。以下组件接口保留用于局部算法实验、参考构造及回归，通过 `qpe_lab evaluate` 运行。

公开导入保持 `from qpe_lab import build_lookup, build_rotation, build_walk, load_task`。直接使用 `qpe_lab.circuits.builder` 中的底层构造函数时显式传入任务对象；任务加载使用 `qpe_lab.tasks.task.read_task(path)`。可复用组件均位于 `qpe_lab.circuits.gadgets`。

| 任务 | 构造接口／产物 | 验收与成本 |
|---|---|---|
| 查表 | `build_lookup` / `OPSTRM01` | 全地址 XOR、任意目标字、符号相位及工作位清理；实际离散门计数 |
| 固定旋转消费者 | `build_rotation` / `QPECIR01` | 加载与卸载各自通过查表验收，中间旋转按整数角模板组合；理想旋转的总 TQ 为 partial |
| 结构化 SOS walk | `build_walk` / `QPEWLK01` | 独立数据与组件数学检查；Clifford+T 合同下展开并统计实际 T |

查表例子：

```python
from qpe_lab import build_lookup

b, address, target = build_lookup(method="unary")
b.emit()
```

旋转消费者执行 `|a,e,ψ,0⟩ → |a,e,U(table[a])^e ψ,0⟩`，地址、使能和系统可相干输入。Fe₂S₂ 角度微核使用 256 个地址、每项 304 位，其中 210–255 为零；两个自旋块共享 20 个空间模式的旋转网络。单粒子 Givens 约定为 `[[cos θ,sin θ],[-sin θ,cos θ]]`，消费者逆序、反号执行消元网络。加载和卸载须分别生成合法 XOR 电路。

```python
from qpe_lab import build_rotation

build_rotation(method="unary").emit()
```

该组件允许同一连续模式对内的角度重排、拆分／合并和整数整周抵消。其理想 Givens 尚未按实际 T 计费，因此报告 `partial`；主训练的完整门流已展开为基本门。微核的 16 位角度配置与主任务有限精度合同分别固定。

```bash
.venv-lab/bin/python -m qpe_lab evaluate \
  --task benchmarks/dfthc/fe2s2/lookup.task.json \
  --candidate candidates/fe2s2_angles/unary.py
.venv-lab/bin/python -m qpe_lab evaluate \
  --task benchmarks/dfthc/toy/rotation/rotation.task.json \
  --candidate candidates/fe2s2_rotation/unary.py
```

## 固定任务与数据准备

| 路径（相对于 `benchmarks/dfthc/`） | 用途 |
|---|---|
| `fe2s2/factors.json`、`manifest.json` | 官方因子转换与来源校验值 |
| `fe2s2/angle-table.json`、`lookup.task.json`、`rotation.task.json` | 角度查表和固定旋转组件任务 |
| `fe2s2/walk-paper-strict/` | 当前主任务及论文路线的固定输入 |
| `fe2s2/walk/` | 独立 Hamiltonian 恢复检查和理想旋转组件基线 |
| `toy/rotation/`、`toy/walk/`、`toy/walk-paper-strict/` | 小实例组件验证和运行测试 |

任务 JSON 与引用数据按摘要绑定。原始数值来源和许可见 [THIRD_PARTY](../THIRD_PARTY.md)。日常运行使用已提交数据；维护方需要重新准备时执行：

```bash
.venv-lab/bin/python scripts/fetch_fe2s2_dfthc.py
.venv-lab/bin/python scripts/prepare_rotation_tasks.py
.venv-lab/bin/python scripts/prepare_walk_tasks.py --profile reference
.venv-lab/bin/python scripts/prepare_walk_tasks.py --profile paper-strict --clifford-t
```

`fetch` 按固定哈希读取官方归档成员；数据准备脚本会重写对应任务，修改后需重新冻结训练 bundle。`--profile alias` 和 `--profile paper` 可生成中间方案供研究，这两类生成快照不提交。

`run_baselines.py` 复现上游原始任务，`run_dfthc.py` 批量检查组件及结构化 walk，并提供固定参数搜索。主训练的自由电路优化始终通过 `evaluate-training` 或 `training-worker`。
