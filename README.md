# QPE Lab

固定 DFTHC Hamiltonian 的量子电路优化工作台。Agent 交付 **Python 电路生成器**；评估器隔离运行程序、检查实际门流，按 **T/T† 数 × 逻辑比特数** 发放奖励。

默认任务覆盖查表、alias 选择、轨道旋转、算符执行和清理，一次提交核心 K 与 K† 各一个完整入口。均匀准备、反射和 QPE 调度固定在外围。

候选可以整体重写电路、跨组件化简、重新分配辅助位，也可以利用冻结合同中的梯度资源态性质。具体接口、输入条件与示例见 [生成器接口](docs/CIRCUIT_GENERATORS.md)。

| Fe₂S₂ 范围 | T/T† | Q | TQ |
|---|---:|---:|---:|
| 核心参考：K 与 K† 各一次 | 15,246 | 447 | 6,814,962 |
| 当前最佳核心生成器 `reuse.py` | **15,230** | **429** | **6,533,670** |
| 共享调度完整受控 walk 候选 | 16,800 | 431 | 7,240,800 |

第二级 **验证通过**：精确或抽样路径通过即可获得奖励和继续优化。第三级 **科学复核通过**：维护方额外核对参考数学检查与候选精确等价，奖励不变。科学范围及论文对照见 [资源复现](docs/SOS_WALK.md)。

## 文件放在哪里

| 位置 | 内容与保留原因 |
|---|---|
| `src/qpe_lab/tasks/` | 任务数据、合同加载、冻结 bundle、agent 任务说明 |
| `src/qpe_lab/circuits/` | DSL、构造器、可复用组件、纯电路检查与模拟器 |
| `src/qpe_lab/evaluation/` | 隔离执行、验证后端、成本、奖励与报告 |
| `third_party/uncompute/` | 固定版本的上游代码、原始任务和数据，使用 submodule |
| `benchmarks/` | 可直接运行的固定任务数据与小实例；保留摘要，避免每次联网重建 |
| `candidates/` | 维护方选定的参考、示例和最佳 Python 生成器 |
| `tests/` | 数学、门语义、错误电路和执行隔离回归 |
| `scripts/` | 安装、上游复现、批量运行和数据生成入口 |
| `config/upstream.json` | 上游代码、Python、uv 和依赖锁的固定版本 |
| `docs/` | [生成器接口](docs/CIRCUIT_GENERATORS.md)、[验证架构](docs/DFTHC_ARCHITECTURE.md)、[科学问题与资源](docs/SOS_WALK.md) |
| `AGENTS.md` | 交给优化 agent 的统一任务说明 |
| `pyproject.toml`、`uv.lock` | 包定义及可复现依赖环境 |
| `THIRD_PARTY.md` | 第三方代码、数据及许可来源 |

以下仅保存在本地，由 `.gitignore` 排除：

- `runs/`：实验候选、冻结 bundle、运行报告和训练账本。连续训练保留 `training-ledger.sqlite3`。
- `output/`：论文、汇报、导出文件和历史归档；`tmp/`：一次性调试脚本。
- `.venv-*`、`.tools/`、`.bootstrap/`、`.cache/`：本机环境、工具和下载缓存。
- 日志、生成门流、结果快照及审计运行目录不提交。需要发布的结果摘要写入现有文档，可复用反例写入测试。

三层共用 `contracts.py` 中的任务规格和验证目标。电路层不依赖任务层或评估层；任务层可以调用电路构造，评估层读取冻结任务并验证实际电路。`cli.py` 串联这些入口，`from qpe_lab import CircuitBuilder` 等公开接口保持一致。具体文件与扩展方式见 [架构](docs/DFTHC_ARCHITECTURE.md#三层结构与文件职责)。

## 安装

```bash
git -c core.autocrlf=false clone https://github.com/fulz024/qpe-lab.git
cd qpe-lab
sudo apt-get install python3 python3-venv python3-pip git gcc make util-linux libseccomp2
bash scripts/setup_lab_wsl.sh
```

运行环境为 Linux / WSL，需要允许用户命名空间。安装入口固定依赖、初始化上游、编译 C 内核并检查隔离运行能力。`.venv-lab` 用于本项目，`.venv-wsl` 用于上游原始任务复现。

Windows 的唯一安装入口是 `scripts/setup-lab.ps1 -Distribution Ubuntu-20.04`；后续命令在 WSL 的项目目录执行。`setup_wsl.sh` 是主安装脚本调用的上游环境准备步骤。

## 开始优化

维护方先用参考构造程序生成门流：

```bash
.venv-lab/bin/python -m qpe_lab evaluate \
  --task benchmarks/dfthc/fe2s2/walk-paper-strict/walk.task.json \
  --candidate candidates/dfthc_walk/optimized.py --timeout 300
```

将返回目录记为 `<run>`，冻结任务：

```bash
.venv-lab/bin/python -m qpe_lab freeze-training \
  --task benchmarks/dfthc/fe2s2/walk-paper-strict/walk.task.json \
  --circuit <run>/circuit.ops --scope core --output runs/training-fe2s2
```

保存返回的 `<bundle_sha256>`，提交当前最佳生成器：

```bash
.venv-lab/bin/python -m qpe_lab evaluate-training \
  --bundle runs/training-fe2s2 --bundle-sha256 <bundle_sha256> \
  --candidate candidates/dfthc_walk/reuse.py
```

参考 reward 为 0.5；通过候选的 reward 为 `C0/(C0+TQ)`，未通过为 0。每次运行保存源码快照、门流、证据和成本。冻结时指定 `--scope walk` 可评估完整 walk；它有独立的成本基线。

维护方可追加科学复核：

```bash
.venv-lab/bin/python -m qpe_lab scientific-review \
  --bundle runs/training-fe2s2 --bundle-sha256 <bundle_sha256> --run <candidate_run>
```

## 接入 agent

推荐使用包含权限、预算和最佳结果管理的会话入口。安装后由维护方运行：

```bash
.venv-lab/bin/python -m qpe_lab agent-session \
  --session runs/sessions/experiment-01 --max-submissions 20 --seconds 1800 --progress-events
```

首次启动自动构造并冻结 Fe₂S₂ 核心任务，复现参考和当前最佳生成器，再开始计时。参考数学检查可能需要数分钟；这些检查和起点复现不占本轮预算，服务准备好后输出 `ready`。也可传入已有的 `--bundle`、`--bundle-sha256` 和可选 `--initial-candidate`。复现失败会报告运行目录，不会带着未验证的起点开始搜索。

准备完成后服务输出 `event=ready`，其中 `task.system_prompt` 是交给模型的完整任务，`task.best` 是已复现的起点。启用 `--progress-events` 时，启动和执行期间会穿插 `event=progress`，包含当前入口、阶段、已用时间和剩余预算；启动器持续读取到 `ready` 或最终响应。未启用时，进度写到 stderr，stdout 保持每个请求一行最终反馈：

```json
{"action":"task"}
{"action":"read","path":"/app/docs/CIRCUIT_GENERATORS.md"}
{"action":"experiment","source":"from qpe_lab import CircuitBuilder\nb=CircuitBuilder()\nprint(b.contract)","timeout":30}
{"action":"submit","source":"from qpe_lab import CircuitBuilder\nb=CircuitBuilder()\nb.emit()"}
{"action":"best"}
```

模型启动器只需注入这份提示词、把 `task/read/write/experiment/submit/best` 请求转发到服务。模型和模型凭据由启动器管理；此模式下文件、实验和提交工具都走该服务，启动器不再给模型另一个宿主终端或宿主文件工具。AI 仍能读框架、写候选、运行 Python 搜索和独立重写完整电路。`experiment` 在隔离环境中执行、返回打印输出；正式判定和奖励来自 `submit`。

虚拟 `/workspace` 是候选目录，参考、代码和文档只读；裁判运行记录、种子、账本及会话状态不可通过这些工具访问。会话执行提交次数和时间预算，错误提交也计次；实验另有次数上限并共享时间预算。`best` 返回最佳已通过程序及其报告，失败或退步不会覆盖它。反馈直接说明是否超过参考/最佳、验证方法、错误类别和其他资源变化。

有 reward 表示候选已通过验证，不代表它优于当前最佳；参考自身也有 0.5 分。查看 `feedback.improved_best` 判断是否推进了本轮最佳结果。`unsupported` 表示验证器在当前能力或计算预算内未完成验证，本次零奖励；它不是电路错误的数学反例。可以继续改写、附加可组合证据，或把候选交给维护方分析所需验证能力。

验证依次安排短时整体探测、可选证据、局部精确、完整精确、完整映射／信道抽样和混合分块，各阶段有保留预算。生成与验证主体默认共用 600 秒；两个入口分配剩余验证时间，前一个入口提前完成的时间留给后一个。整体重写始终有独立验证路径，所有通过条件保持一致。最终响应的 `timing` 和 `verification_schedule` 记录实际耗时；剩余预算表示时间上限，并非完成时间预测。

继续同一会话时使用 `agent-session --session runs/sessions/experiment-01`，预算不重置；预算结束后仍可读取和导出结果。不同会话也沿用同一训练统计账本。协议、实验容量和接入示例见 [生成器接口](docs/CIRCUIT_GENERATORS.md#会话工具接口)。

合作者通过 `best` 取回最佳 Python 生成器及运行报告，交付时附上 bundle 摘要、任务范围、验证级别和 T/Q/TQ/reward。统一比较结果时，维护方用同一冻结任务重新提交这些源码，以复验结果登记成绩。使用不同裁判版本或冻结任务得到的报告应分别注明，核心、完整 walk 和整个 QPE 的成本也分别报告。

后训练接入读取 `feedback.training_signal`，分别处理通过、已否决和未完成。验证未完成时，正确性标签为空，原提交仍消耗预算且当前 reward 为零；训练器保留该样本与完整尝试记录。维护方可以对已冻结门流追加一次最多 1,800 秒的验证：

```bash
.venv-lab/bin/python -m qpe_lab complete-verification \
  --bundle runs/training-fe2s2 --bundle-sha256 <bundle_sha256> --run <candidate_run>
```

补验不再运行生成器，核对源码和门流摘要，使用新挑战并继续扣除原统计账本预算；完成后更新原运行报告。它使用相同的精确／抽样验收标准，奖励公式不变，仍可能返回未完成。补验权限属于维护方，会话工具不能调整验证预算。详细训练字段与进度接入见 [生成器接口](docs/CIRCUIT_GENERATORS.md#会话工具接口)。

已有独立启动器的维护方也可只导出任务，继续使用底层评估服务：

```bash
.venv-lab/bin/python -m qpe_lab agent-task \
  --bundle runs/training-fe2s2 --bundle-sha256 <bundle_sha256> \
  --workspace runs/candidates/session-01 --budget "30 分钟或 20 次提交" \
  --output runs/session-01/task.json
```

这个 JSON 包含 `system_prompt`、参考位置与摘要、入口合同、DSL、参考成本、reward、提交命令、预算和文档的绝对路径。通用说明来自 [AGENTS.md](AGENTS.md)，本轮信息来自冻结 bundle，DSL 来自实际门解析器。此底层接口由外部启动器负责工作区、权限和预算。维护方启动评估服务：

```bash
.venv-lab/bin/python -m qpe_lab training-worker \
  --bundle runs/training-fe2s2 --bundle-sha256 <bundle_sha256>
```

每行输入 `{"source":"Python 源码"}`，返回判定、验证级别、成本、reward 和运行记录。`agent-task` 提供任务，`training-worker` 评估提交；模型由外部启动器启动。学习 agent 只写候选工作区，裁判、固定数据和结果由维护方管理。

训练 manifest 为 v3；升级裁判后重新冻结 bundle，连续训练保留原账本。接口细节见 [生成器接口](docs/CIRCUIT_GENERATORS.md)，验证顺序和统计预算见 [验证架构](docs/DFTHC_ARCHITECTURE.md)。

## 开发与复现

```bash
# 本项目完整回归
.venv-lab/bin/python -m unittest discover -s tests -v

# 原始 Pauli-LCU：Coulomb 与 Fe₂S₂，使用原 grader 和完整测试集
.venv-wsl/bin/python scripts/run_baselines.py --suite all --jobs 8

# 组件与结构化 walk 的批量回归
.venv-lab/bin/python scripts/run_dfthc.py --include-microkernels
```

回归测试覆盖门语义、数学目标、任务派发、混合验证、证明范围与缓存、奖励计数和执行隔离；验收结果保存在每次运行记录中。可选 Lean 示例与安装见 [证明附件](docs/CIRCUIT_GENERATORS.md#lean-附件)。原始任务的已复现指标见 [资源复现](docs/SOS_WALK.md#原始-pauli-lcu-任务复现)。

日常使用直接读取已提交的 `benchmarks/`。需要重新生成数据时，维护方依次使用 `fetch_fe2s2_dfthc.py`、`prepare_rotation_tasks.py` 和 `prepare_walk_tasks.py`；参数与示例见 [生成器接口](docs/CIRCUIT_GENERATORS.md#固定任务与数据准备)。
