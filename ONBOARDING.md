# Latent Dirac 交接文档（Handoff）

写于 2026-07-05。给接力开发者：这份文档告诉你项目是什么、现在处于什么
状态、哪些决策已经定死（以及为什么）、下一步做什么、哪里有坑。
读完本文后，请通读 `AGENTS.md`（规则唯一真源）和
`docs/superpowers/specs/2026-07-05-geant4-engine-positioning-design.md`
（引擎战略的完整决策记录）。

## 一、项目是什么

Latent Dirac 是反物质工厂（正电子/反质子设施）的开放仿真平台：
声明式 YAML 场景描述"源 → 输运 → 俘获"束线，批量并行仿真与参数扫描，
3D 可视化。三大支柱：

1. **平台**：pydantic 场景 schema + 可插拔求解器 + 可选查看器
2. **吞吐**：JAX jit+vmap 批处理后端（一次发射 n_configs × n_particles），
   可微软接受目标（autodiff 穿过每个输运步）
3. **账本**：粒子永不删除，`alive` 掩码 + `lost_at_element` int32 账本，
   每个反粒子的死亡地点可寻址

**新方向（本次交接的核心）**：项目定位已升级为"内置 Geant4 引擎"。
完整的香草 Geant4 v11.4.2 源码树已 vendored 在 `geant4-v11.4.2/`
（17,645 个文件，只读基线）。战略是三轨并行：

- **轨道 A**：香草引擎基线 + 最小物理构建配方（不改 Geant4 一行）
- **轨道 B**：伴生加速库（经 `G4VFastSimulationModel` 快速仿真挂点
  接入，AdePT/Celeritas 模式）——算法优化都在这里做
- **轨道 C**：JAX 校准环（Geant4 离线产出产额表/训练数据 →
  table_based / externally calibrated 源）

## 二、当前仓库状态（交接时刻的精确快照）

本地分支 master，**全部未推送到远端**：

| 提交 | 内容 | 状态 |
|---|---|---|
| `0178fe85` `vendor:` | Geant4 v11.4.2 整树进仓（零修改，`.gitattributes -text` 保证字节级等同上游） | 已提交，未推送 |
| `64a11bc4` `chore:` | 工具链排除（ruff/MANIFEST）、NOTICE 归属、README vendored 小节、AGENTS/CLAUDE 规则 | 已提交，未推送 |
| （待提交） | M0 定位改写：安全范围四处同步重写 + 守门测试 + 引擎 spec + roadmap + 审查修复 | **工作区未提交** |

未提交部分的验证状态：**188 个测试全绿 + ruff 干净 + sdist 已确认不含
vendored 树**，并且过了一轮多 agent 代码审查（3 视角 + 12 个对抗验证，
7 项发现全部确认并已修复——主要是"定位改写只改了一半"的文档不一致）。
建议的提交信息：`docs!: reposition around the vendored Geant4 engine
track`（BREAKING：安全范围重写）。**是否提交/推送由项目所有者决定，
接手后第一件事请先 `git status` + `git log --oneline -5` 核对现状。**

另：本机（Windows）新建了 `.venv`，用 `.venv/Scripts/python.exe`
（CLAUDE.md 里写的 `.venv/bin/python` 是 Linux 路径习惯）。

## 三、必读文件（按优先级）

1. `AGENTS.md` — 物理规则、工程规则、范围、当前阶段的**唯一真源**
2. `docs/superpowers/specs/2026-07-05-geant4-engine-positioning-design.md`
   — 引擎战略完整决策记录（含被否决方案与理由）
3. `docs/roadmap.md` — 阶段计划 + 引擎轨道里程碑 M0–M4
4. `docs/safety_scope.md` — 规范安全范围条目（被测试逐字锁定）
5. `CLAUDE.md` — Claude Code 操作纪律（如果你也用 Claude 开发）

## 四、核心架构速览

```
YAML 场景 (pydantic, fail-fast) → 源采样 (NumPy RNG, seed 决定)
  → ParticleState (pytree dataclass, SoA, SI 单位)
  → A. NumPy float64 参考管线 (PipelineRunner/Stage, 真值基准)
    B. JAX 批处理 (BatchedSceneProgram: jit+vmap, lax.scan)
  → 诊断 (loss_ledger / scene_report) → 可视化 (Plotly HTML / WebP)
```

Load-bearing 的设计决策（改代码前必须懂）：

- **无量纲动量 u = p/(mc)**：3 MeV 正电子的 SI 动量² ≈ 3.5e-42，低于
  float32 最小正规数——SI 只存在于 State 边界，内核全无量纲。
  这是 float32 GPU 路线的命根子。
- **一份物理，两个后端**：`solvers/kernels.py` 的 `boris_step(..., xp)`
  是纯函数，`xp=np|jnp` 切换后端。物理只写一次。
- **粒子永不删除**：形状静态（vmap 前提）+ 账本可寻址（产品卖点）。
- **粒子抽象**：物种 = (质量, 带符号电荷) 两个数 + 纯标签
  （pdg_id/is_antimatter 零计算引用）；species 挂在云上不是逐粒子；
  已决策**不加夸克**（色禁闭，自由夸克不存在）；μ/π/K 是合法扩展
  方向（`parent_id` 字段是衰变链预留挂点）。
- **粒子间零相互作用**：demo 里正负电子"穿过"不湮灭是对的——飞行中
  湮灭截面 ~10⁻¹⁴ 量级；真实湮灭发生在打进固体后（annihilation_plate
  建模的正是这个）。

## 五、Geant4 引擎决策记录（为什么是现在这个形态）

结论：**香草 Geant4 vendored 进仓 + 伴生加速库 + 受控补丁协议**。
完整论证在 spec 里，速记版：

- **不 fork 改内核**：GeantV（CERN，数十人年）试过重写内核换性能，
  实测 1.5–2x，项目终止。Geant4 的价值是三十年实验验证血统，fork
  的瞬间血统清零。成功模式是 AdePT/Celeritas——伴生库 + 官方挂点。
- **不提取子集**：依赖网收敛（FTF → 弦碎裂 → 核模型 → 预平衡/去激发
  → particles/materials/global），真正可丢弃的只有 ~11%（可视化/UI），
  构建期选择性链接能拿到同样效果且不用维护减法 fork。
- **vendored monorepo 是所有者的明确决定**（评审建议独立 engine 仓库
  被否）。风险由护栏控制：树只读、字节级等同上游、工具链三重排除、
  补丁预算（≤5 个、单个 ≤500 行、每个附回归验证）。
- **JAX 不可被 Geant4 替代**：执行模型相反（vmap 批处理 vs 逐轨迹
  分支 MC）、Geant4 零梯度、真空场输运恰是 Geant4 无优势场景。
  正确关系 = JAX 快环 + Geant4 真值锚（sim-to-real 里 Geant4 站在
  "real" 一侧）。
- **License 红线**：归属声明（NOTICE）不可移除；"Geant4" 名称不得
  背书式使用（需协作组书面许可）；改动版必须可区分描述；公开的
  补丁视同授权回协作组，且不得写入专利申请。
- 反质子物理落点：FTFP_BERT 物理列表（FTF 弦模型 ≥3 GeV +
  Bertini 级联；`G4FTFAnnihilation` 处理 p̄ 湮灭）。
- Python 绑定 11.1 起移出 Geant4 主仓库 → 适配器走 GDML + 子进程/宏。

## 六、四大陷阱（学费已交，别再交一遍）

1. **安全范围四处同步**：`tests/test_project_positioning.py` 的
   `EXPECTED_EXCLUSIONS`（元组精确相等）+ `docs/safety_scope.md` +
   `AGENTS.md` + README（子串断言）。改一处必须改四处，且范围变更
   要先写定位 spec。
2. **README 性能措辞**：无 benchmark 引用的比较性性能词
   （faster than / fastest 等）会挂 CI。改 README 前先读
   `test_project_positioning.py`。
3. **双循环镜像**：`backends/differentiable.py` 镜像
   `jax_scene._make_simulator` 的元素循环——新元素类型必须两处同步加，
   否则梯度静默错误。
4. **vendored 树只读**：任何情况下不得编辑 `geant4-v11.4.2/` 内文件；
   不得对它跑 lint/format/pytest/批量搜索（17k 文件，`du` 都会超时）；
   涉及它的提交用 `vendor:` 前缀。

## 七、下一步工作（按 roadmap 顺序）

- **M1' 引擎构建配方**：根目录建 `recipes/`——最小物理构建
  （CMake：不链 visualization/interfaces/analysis 类别库；数据集按
  所选物理列表裁剪，不用 HP 中子物理就不装 G4NDL）。CI 放 Python
  矩阵之外（容器化）。第一个可交付物：能在 Linux 容器里构建出
  FTFP_BERT 可用的引擎二进制。
- **M2 适配器真实化**：场景 → GDML 导出 + 子进程/宏驱动 + 粒子云
  交换。**同一变更内**翻转 `test_only_placeholder_adapters_are_present`
  守门测试。
- **M3 产额表管线**：离线 FTFP_BERT 跑靶产额 → `table_based` 反质子
  源 → Chain 2 demo 的"画出来的靶"变成诚实的引擎背书源。
- **M4 伴生加速库**：根目录 `engine/`（一等公民 C++），经
  fast-sim 挂点接入，EM 域先行；性能数字必须带 vs 香草 Geant4 的
  开放基准。
- 与引擎轨道并行的 Python 侧方向：Phase 3 收尾（JAX 后端场图支持、
  交互式查看器）、平滑有限长螺线管场模型（demo 硬边场的改进，
  设计讨论已有，未立项）。

## 八、开发环境与命令

```bash
# 环境（Windows 本机已建好 .venv；新机器照此）
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,viz,jax]"

# 测试与 lint（每次提交前必须全绿）
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .

# CLI
.venv/Scripts/latent-dirac run examples/scenes/hello_beamline.yaml
.venv/Scripts/latent-dirac render examples/scenes/hello_beamline.yaml -o hello.html
```

注意：本机 CPU-only——**不得产出任何性能数字**（诚实纪律要求硬件
标注的 Linux CUDA 数字）。

## 九、工作流（每个特性）

1. 设计 spec 写到 `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
2. TDD：先写失败测试，确认因正确原因失败
3. `pytest -q` + `ruff check .` 全绿
4. 提交前跑 code review（历史上 7/7 次抓到真实物理 bug，包括一次
   反质子符号错误——不要跳过）
5. Conventional commit（`feat:`/`chore:`/`docs!:`/`vendor:`）+
   Co-Authored-By trailer，然后推送
