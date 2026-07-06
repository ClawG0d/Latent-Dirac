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
（git 记录 17,644 个文件，只读基线）。战略是三轨并行：

- **轨道 A**：香草引擎基线 + 最小物理构建配方（不改 Geant4 一行）
- **轨道 B**：伴生加速库（经 `G4VFastSimulationModel` 快速仿真挂点
  接入，AdePT/Celeritas 模式）——算法优化都在这里做
- **轨道 C**：JAX 校准环（Geant4 离线产出产额表/训练数据 →
  table_based / externally calibrated 源）

## 二、当前仓库状态（2026-07-06 更新）

master 全部推送，CI 绿（M2 又新增 13 个 Matter 适配器测试；精确总数随
并行开发漂移，以 `pytest -q` 为准）。已完成的大块（详见 CHANGELOG 与
`git log`，各自的设计记录在 docs/superpowers/specs/ 同日 spec 里）：

- **引擎轨道 M0 + M1'-lite + M2 + M3-lite**：Geant4 v11.4.2 整树 vendored
  （与上游 tag 字节级比对一致；唯一已知偏差：上游 `CHANGELOG` symlink
  在仓库里是普通文件——因保留 Windows 原生 checkout **有意维持，
  勿"修复"**，见引擎 spec 附录）；`engine/README.md` 构建配方 +
  `engine/yieldgen` 产额表 + `antiproton_yield_table` 源 + Chain 2b demo；
  **M2 Geant4 Matter 适配器真实化**（`adapters/geant4/adapter.py` +
  `engine/transformer`，材料板相空间变换，子进程+文件、决不进程内，
  详见 §七）
- **solver 动物园定位**：JAX/NumPy 是基底，CERN 工具链是引擎组件；
  README 已重构为 Genesis 风格四层架构
- **闭环 v1 全部完成（勿重做）**：openPMD 输出（`io/openpmd_io.py`，
  `[openpmd]` extra）→ uproot ROOT I/O（`io/root_io.py`，`[root]`，
  显式写 TTree 非 RNTuple）→ Xsuite adapter
  （`adapters/xsuite/adapter.py`，`[xsuite]`，placeholder 守门测试已
  翻转为 adapter-status 测试）→ mean-field 空间电荷
  （`fields/space_charge.py`，`space_charge: uniform_sphere`，NumPy
  管线专属，JAX 侧显式拒绝）
- **安全范围钉死降为三处**（所有者决定）：`EXPECTED_EXCLUSIONS` 元组 +
  `docs/safety_scope.md` + `AGENTS.md`；README 只留链接，
  **不要把整节加回**

接手后第一件事仍是 `git status` + `git log --oneline -10` 核对现状；
多会话并行开发是常态（昨天 rebase 调和了五次）——**push 前必 fetch**。

## 三、必读文件（按优先级）

1. `AGENTS.md` — 物理规则、工程规则、范围、当前阶段的**唯一真源**
2. `docs/superpowers/specs/2026-07-05-geant4-engine-positioning-design.md`
   — 引擎战略完整决策记录（含被否决方案与理由）
3. `docs/superpowers/specs/2026-07-05-solver-zoo-composition-design.md`
   — solver 动物园组合架构（组件切割、接口契约、实施顺序）
4. `docs/roadmap.md` — 阶段计划 + 引擎轨道里程碑 M0–M4 + 闭环 v1 顺序
5. `docs/safety_scope.md` — 规范安全范围条目（被测试逐字锁定）
6. `CLAUDE.md` — Claude Code 操作纪律（如果你也用 Claude 开发）

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

## 六、五大陷阱（学费已交，别再交一遍）

1. **安全范围三处同步**：`tests/test_project_positioning.py` 的
   `EXPECTED_EXCLUSIONS`（元组精确相等）+ `docs/safety_scope.md` +
   `AGENTS.md`。改一处必须改三处，且范围变更要先写定位 spec。
   README **有意不再镜像**该清单（所有者决定，2026-07-05），只保留
   到 docs/safety_scope.md 的链接（测试强制）——不要把整节加回去。
2. **README 性能措辞**：无 benchmark 引用的比较性性能词
   （faster than / fastest 等）会挂 CI。改 README 前先读
   `test_project_positioning.py`。
3. **双循环镜像**：`backends/differentiable.py` 镜像
   `jax_scene._make_simulator` 的元素循环——新元素类型必须两处同步加，
   否则梯度静默错误。
4. **vendored 树只读**：任何情况下不得编辑 `geant4-v11.4.2/` 内文件；
   不得对它跑 lint/format/pytest/批量搜索（17k 文件，`du` 都会超时）；
   涉及它的提交用 `vendor:` 前缀。
5. **WSL 文件系统**：在 WSL 里开发时 repo 必须放 ext4（`~/` 下），
   不要放 `/mnt/c`——vendored 树 17k 文件跨 9P 慢一个数量级；
   WSL 与 Windows 原生各自独立 clone，不要共用一个工作树。

## 七、下一步工作（按 roadmap 顺序；闭环 v1 已完成，见 §二）

- **M1' 引擎构建配方（续）**：首个配方已交付
  （`engine/README.md`：WSL/Linux 最小物理构建，不链
  visualization/UI，数据集构建期获取）。剩余：容器化 CI
  （放 Python 矩阵之外）与按物理列表裁剪数据集的配方变体。
- **M2 适配器真实化（已完成，勿重做）**：Geant4 Matter 适配器
  （`adapters/geant4/adapter.py` 的 `Geant4MatterAdapter` 驱动
  `engine/transformer`）——粒子云穿过 NIST 材料板，FTFP_BERT 算能损/
  散射/反质子湮灭，幸存者带引擎相空间回来、被吸收者进损失账本；
  交换走子进程 + 相空间 CSV（id 键、完成标记守卫、Windows 走 WSL
  桥），**决不进程内**。守门测试已翻转为 adapter-status 断言
  （geant4 真实、root 仍占位）。剩余：`matter_slab` 场景元素（M2b）、
  全场景 GDML 翻译。**注意实际形态是"材料板相空间变换"，不是最初
  设想的 GDML 全场景导出**——那留作后续。
- **M3 产额表管线**：首个交付已落地（`engine/yieldgen` 离线
  FTFP_BERT 产额 → `antiproton_yield_table` table_based 源 →
  引擎背书的 Chain 2b demo）。剩余：正电子/慢化产额表、surrogate
  源向 `externally calibrated` 毕业。
- **M4 伴生加速库**：根目录 `engine/`（一等公民 C++），经
  fast-sim 挂点接入，EM 域先行；性能数字必须带 vs 香草 Geant4 的
  开放基准。
- **GPU 车道**（本机 5070 Ti 正是为它准备的）：WSL2 装
  `jax[cuda12]`（Blackwell 需较新 CUDA 12.8+/jaxlib，以官方支持矩阵
  为准），先做 float32 GPU 后端对 float64 CPU 参考的容差分层校验，
  再做诚实 benchmark 套件。性能数字标签必须打全：GPU 型号 + WSL2 +
  CUDA/驱动版本 + 积分器/步长/粒子数/batch/保真层级。
- 其余 Python 侧方向（排在 GPU 车道之后）：交互式查看器、
  JAX 后端场图支持；物理填充方向：降能片物理（走 M3 产额表路线）、
  Surko 阱 buffer-gas 碰撞（需截面数据研究）。
  已完成的原列项：平滑有限长螺线管场模型（2026-07-06 落地为
  `solenoid` 元素的 `thin_sheet` profile，双后端，见同日 spec）；
  残余气体湮灭寿命模型（2026-07-06 落地为 `residual_gas_loss` 元素
  及其可微期望存活项）。

## 八、开发环境与命令

三个环境分工：macOS 机做规划与文档；测试与运行在 Windows 机的
WSL2 里（repo 放 ext4，见陷阱 5）；同机另保留 Windows 原生
checkout（独立 clone）。

```bash
# macOS / WSL2（Linux 路径习惯）
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,viz,jax,openpmd,root,xsuite]"
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/latent-dirac run examples/scenes/hello_beamline.yaml

# Windows 原生 checkout（路径差异仅此而已）
.venv/Scripts/python.exe -m pytest -q
```

GPU：WSL2 直通 RTX 5070 Ti（Blackwell）。消费卡 FP64 吞吐被砍——
float32 无量纲 kernel 车道正好对口；float64 真值锚留在 CPU NumPy
参考后端。**性能数字只能出自这台机器**，且标签打全：GPU 型号 +
WSL2 + CUDA/驱动版本 + 积分器/步长/粒子数/batch/保真层级
（macOS 规划机 CPU-only，不得产出任何性能数字）。

## 九、工作流（每个特性）

1. 设计 spec 写到 `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
2. TDD：先写失败测试，确认因正确原因失败
3. `pytest -q` + `ruff check .` 全绿
4. 提交前跑 code review（历史上 7/7 次抓到真实物理 bug，包括一次
   反质子符号错误——不要跳过）
5. Conventional commit（`feat:`/`chore:`/`docs!:`/`vendor:`）+
   Co-Authored-By trailer，然后推送
