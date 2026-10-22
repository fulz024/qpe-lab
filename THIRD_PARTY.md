# 第三方代码与数据

## uncompute

- 仓库：https://github.com/andigu/uncompute
- 固定提交：`64b5a13aa249f1360846657392e6c66817b26d0f`
- 作者：Andi Gu、Alek H. Kemeny、Susanne F. Yelin、Mikhail D. Lukin。
- 论文：*Discovering state-of-the-art quantum circuits with large language models*，arXiv:2610.22367。
- 代码许可：Apache-2.0，完整原文位于 `third_party/uncompute/LICENSE`。

本项目直接复用该仓库，以 submodule 保留来源与版本。当前环境适配发生在项目自己的启动和记录脚本中，上游电路、数据和验证逻辑没有补丁。各任务对文献算法、数据和其他代码的归属说明，保留在上游相应 README 和 Attribution 中。

`src/qpe_lab/circuits/gadgets/qrom.py` 的 `plane_phase_table` 改编自上述固定提交中 `tasks/chemistry-fe2s2/circuits/discovered/circuit.py` 的 `b2_fixup`：沿用低两位 Reed–Muller 相位展开、共享单项控制和测量卸载方法，改为通用 DFTHC 测量表达式、完整地址域与零填充语义。相关来源仍适用上游 Apache-2.0 许可。

Fe₂S₂ 原始积分来自 Li–Chan 铁硫簇活性空间数据；上游仓库包含压缩数据、来源说明和固定校验值。本轮运行使用上游已保存的数据，不重新拟合或改变 Hamiltonian。

## DFTHC 数值数据

- 来源：[Low et al. 论文的官方数值归档，Zenodo 17066718](https://zenodo.org/records/17066718)，归档提交者 Nicholas Rubin。
- 使用 `spinfree.zip` 内 `best_solutions` 中的 Fe₂S₂ `(14,15,5)` 参数；确切成员路径和 SHA-256 保存在 `benchmarks/dfthc/fe2s2/manifest.json`。
- `factors.json` 是这些数值参数的 JSON 转换；`angle-table.json` 是本项目按明确 Givens/量化约定生成的派生数据。没有将官方 JAX 恢复脚本复制为本项目源码。
- 这些输入已扩展用于一体恢复和完整中心化 SOS walk。`tasks/data/hamiltonian.py` 独立实现论文及归档 `df_thc_sga.py` 的数学定义；代码未从官方 JAX 脚本复制。新派生数据在 `benchmarks/dfthc/fe2s2/walk/`，保留源哈希、归一化对照和精度范围。
- 官方 `integrals.zip` 的 Fe₂S₂ 成员与上游解压 FCIDUMP 逐字节相同。当前未声称达到论文全部化学精度或重现其优化资源成本；数据的权利声明以来源记录为准，不继承 uncompute 的代码许可。

## 本地执行依赖

Clifford+T 后端使用 [quantum-programming/pygridsynth](https://github.com/quantum-programming/pygridsynth)，固定 PyPI `2.0.0`，通过依赖安装，不复制其源码。项目采用其 Ross–Selinger 旋转综合接口，另行复核生成字的相位敏感矩阵残差；出处和原算法引用见该项目 README。版本、下载哈希及传递依赖在根 `uv.lock`，运行报告另记录已安装后端及数值库的文件哈希。它仅加入 `.venv-lab`，原始复现环境 `.venv-wsl` 和上游依赖锁不改动。

Bubblewrap 从本机 Ubuntu 软件源下载并提取到被 Git 忽略的 `.tools/bubblewrap/`，用于候选生成器的命名空间和挂载隔离。系统包自带的许可说明保留在提取目录中；其二进制不进入项目源码。运行报告记录其实际文件哈希。

Lean 证明环境另用 [Bubblewrap 0.11.0](https://github.com/containers/bubblewrap/tree/v0.11.0) 的有容量上限临时文件系统，从官方源码构建在 `.tools/bubblewrap-0.11.0/`。固定源码包 SHA-256 为 `cfeeb15fcc47d177d195f06fdf0847e93ee3aa6bf46f6ac0a141fa142759e2c3`，许可随源码包保留；生成器原有隔离工具保持独立。

## 电路等价验证与辅助位复用

可选形式化后端使用 [Lean 4.35.0](https://github.com/leanprover/lean4/releases/tag/v4.35.0) 官方二进制发行包及其中的 `lake comparator`，许可见发行包内的 LICENSE。下载包固定 SHA-256 为 `7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d`，安装在被 Git 忽略的 `.tools/lean-4.35.0-linux/`；运行库和可执行文件内容纳入裁判摘要。本项目自行实现门语义转换及接口，不复制独立 comparator 项目的源码。

较大 Clifford+T 电路块调用 [PyZX](https://github.com/zxcalc/pyzx)，固定 PyPI `0.10.7`，Apache-2.0 许可，通过锁定依赖安装，不复制其源码。使用其 ZX 图构造与化简；本项目独立检查最终边界连线以及完整精确标量，不使用默认忽略全局相位的便捷验收接口。包文件哈希纳入裁判指纹。

调研参考了 [MQT QCEC 官方文档](https://mqt.readthedocs.io/projects/qcec/en/stable/equivalence_checking.html)、[Hong et al., Equivalence Checking of Dynamic Quantum Circuits](https://arxiv.org/abs/2106.01658)、[DeCross et al., Qubit-reuse compilation with mid-circuit measurement and reset](https://arxiv.org/abs/2210.08039) 和 [Qiskit qubit-reuse 插件](https://github.com/qiskit-community/qiskit-qubit-reuse)。这些工具未安装或复制到本项目；小块全记录 Kraus 比较、保持相干 I/O 的清洁辅助位分配及独立核验由本项目实现。实现范围与方法选择见 [架构与验收](docs/DFTHC_ARCHITECTURE.md)。
