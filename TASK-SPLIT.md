# 双机分工计划（Task Split）

写于 2026-07-06，由 Windows 台式机（WSL2 + RTX 5070 Ti）侧会话规划；
**同日晚更新**：T1（matter_slab）、T2（xsuite_lattice）、T4（buffer-gas
spec 及 buffer_gas_cooling 参数化第一档）均已落地——速度惊人。新的
任务分配见 §三；整体执行计划（GPU 车道 → M3/M4 → Phase 4）见
`docs/superpowers/specs/2026-07-06-execution-plan-gpu-to-phase4-design.md`。

本文写给 **Mac Air 上的协作者及其 Claude Code 会话**：哪些任务划给
Mac、哪些留在 Windows 机、为什么，以及接手前读什么。

前置阅读（本文不重复其内容）：`ONBOARDING.md`（项目现状与五大陷阱）
和 `AGENTS.md`（物理/工程规则唯一真源）。状态基线：闭环 v1 已完成，
引擎轨道 M0 / M1'首个配方 / M2 / M3 首张产额表已落地。

## 一、分工原则

- **Mac Air（CPU-only）承担**：自包含、纯 Python、CPU 上可完整测试的
  特性——场景 schema 元素、场图导入器、参数化物理模型、文档、demo。
- **Windows 机（WSL2 + RTX 5070 Ti）保留**：需要 GPU、需要在 WSL/Linux
  里构建或运行 Geant4 引擎、或设计上牵动 load-bearing 镜像对
  （`backends/differentiable.py` ↔ `backends/jax_scene.py`）的任务。
- 硬规则（AGENTS.md 诚实纪律）：**Mac 上绝不产出任何性能数字**。
  官方性能数字只能出自 Windows 机并带全标签（GPU 型号 + WSL2 +
  CUDA/驱动版本 + 积分器/步长/粒子数/batch/保真层级）。Mac 上的 JAX
  只做 CPU 正确性验证，不做计时、不写吞吐对比。
- `geant4-v11.4.2/` 只读冻结，任何人任何情况不编辑、不 lint、
  不批量搜索（17k+ 文件）。

## 二、Mac 环境搭建

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,viz,jax]"   # 核心，必须成功
# 可选 extras 逐个装，装不上不阻塞（CI 也是这么做的，见 ci.yml；
# 注意用 -e "." 形式——PyPI 上尚无此包，写包名会装错）：
.venv/bin/python -m pip install -e ".[openpmd]" || true
.venv/bin/python -m pip install -e ".[root]"    || true
.venv/bin/python -m pip install -e ".[xsuite]"  || true
.venv/bin/python -m pytest -q          # 全绿再开工（缺的 extra 自动跳过）
.venv/bin/python -m ruff check .
```

可选依赖全部惰性导入：模块随时可 import，缺哪个 extra 哪个测试
自动 skip。注意区别：T1（Geant4）的合同测试走 Python stub，
**无需任何引擎**即可全跑；T2（Xsuite）是进程内适配器、没有 stub 缝，
`xtrack` 装不上则 T2 的测试在本地永远 skip、只剩 CI 兜底——所以
**领 T2 之前先确认 xsuite 在你的 Apple Silicon 上装得上**，装不上
就跳过 T2 先做 T3/T4，并告知 owner（T2 改由 Windows 机做）。

## 三、划给 Mac 的任务（按此顺序做）

**2026-07-06 晚状态**：T1/T2/T4 已完成（见 CHANGELOG），当前分配：

### T5 阻性气体截面表第一档（owner 已拍板，最高优先）

按你自己写的 buffer-gas spec
（`docs/superpowers/specs/2026-07-06-buffer-gas-collisions-design.md`）
的下一梯级：为正电子–N₂/CF₄ 策展能量依赖截面表（逐文献溯源，
per-source provenance 纪律——正电子数据没有 LXCat 式统一开放库），
把 `buffer_gas_cooling` 从常数速率参数化档升级到 table-based 档。
纯 CPU、研究向。注意：trap_storage_lifecycle demo（demo 14）用的
是常数速率档，升级后按需刷新其场景参数与 README 报告块。

### 已完成（勿重做）：T1 / T2 / T4

- **T1 `matter_slab`**：已落地（`10ad5fb2`，spec
  `2026-07-06-matter-slab-scene-element-design.md`），并已被
  ELENA 交接 demo（真引擎降能片）实战使用。
- **T2 `xsuite_lattice`**：已落地（`d06a515b`，spec
  `2026-07-06-xsuite-lattice-scene-element-design.md`），并已被
  ELENA 环 demo（60 圈 xtrack 追踪）实战使用。
- **T4 buffer-gas**：spec 已交付并超额完成第一档代码
  （`buffer_gas_cooling` 常数速率参数化元素 + 可微期望存活项），
  已被阱存储生命周期 demo 使用。

### T3 CST / SIMION 场图导入格式（次优先）

扩展 `latent_dirac/fields/field_map.py`：现有 COMSOL 规则网格 CSV
导入是模板（trilinear 插值、界外返回零）。新增 CST 和 SIMION 的
导出格式解析，table-based 层级，测试用小型合成 fixture 文件。
纯解析工作，完全自包含。spec 里先做格式调研并记录来源。

### 持续任务（穿插做）

- mkdocs-material 文档站（roadmap "Continuous" 轨道）。
- demo / 场景 YAML 打磨，`docs/scene_schema.md` 与实现同步。
- 注意：README 性能措辞红线（`tests/test_project_positioning.py` 的
  正则扫 README + `docs/*.md`——比较级性能词无 benchmark 引用即挂 CI）。

## 四、留在 Windows 机的任务（Mac 请勿动这些）

- **GPU 车道**：`jax[cuda12]` float32 后端对 float64 CPU 参考的
  分层容差校验 + 带全标签的 benchmark 套件（需要 5070 Ti 本体）。
- **M1' 容器化 CI**：Geant4 引擎 Docker 构建（数小时级编译，
  x86_64/WSL 专属）。
- **M3 后续产额表**（正电子/慢化）与 surrogate 源向
  `externally calibrated` 毕业：都要跑已构建的引擎。
- **M4 伴生加速库**：C++，链接 WSL 里的 Geant4 构建。
- **镜像对设计工作**：平滑有限长螺线管场模型、JAX 后端场图支持、
  批量 monitor 快照——全都要同步改
  `jax_scene._make_simulator` ↔ `differentiable.py` 两处元素循环
  （陷阱 #3），归 Windows 侧单人所有。
- `engine/` 与 `adapters/geant4/` 内部改动。

## 五、模块所有权（避免双会话同模块，直至另行协调）

| 归属 | 模块 |
| --- | --- |
| Mac | `scene/schema.py`、`scene/build.py`、`viz/scene_3d.py`（T1/T2 期间）、`fields/field_map.py`（T3）、`docs/` 站点（T4 是纯 spec/调研，暂无代码落点） |
| Windows | `backends/`、`engine/`、`adapters/geant4/`、benchmark/CI 基础设施 |
| 谁都不动 | `geant4-v11.4.2/`（只读）；README 大结构（owner 在 GitHub 网页端直接改）；安全范围三处钉死（改动先出 positioning spec） |

跨界需求（比如 T1 发现适配器要加参数）：先和 owner 打招呼，
不要静默跨界提交。

注意：本仓库常态是**三个会话并行推 master**——写本文当天就有第三
会话把提交落进了 `scene/schema.py`/`scene/build.py`（`9c1aa128`）。
上表所有权以实时沟通为准；动手前 fetch，认领后再开工。今天同日
建 spec 的先 glob `docs/superpowers/specs/` 防撞名。

## 六、协作纪律速查（细则在 CLAUDE.md「Collaboration」）

1. 每特性：spec →（先失败的）测试 → `pytest -q` + `ruff check .`
   全绿 → code review → conventional commit（Co-Authored-By trailer）。
2. **push 前必 `git fetch` + rebase onto `origin/master`**（不 merge），
   rebase 后重跑两个门再推；被拒就再 fetch 再 rebase。
3. spec 文件名是 `YYYY-MM-DD-<topic>-design.md`，同日两会话会撞名——
   先 glob `docs/superpowers/specs/`，topic 起得有区分度。
4. 追加型共享文件（CHANGELOG.md、docs/roadmap.md、AGENTS.md "Next:"、
   README）冲突时**保留双方条目**。
5. 组件毕业（placeholder→real、planned→shipped）时，README Solvers
   表 + `docs/roadmap.md` + `CHANGELOG.md`（如涉及适配器状态还有
   `test_adapter_status_matches_roadmap` 的断言）在**同一 commit**
   里翻转；里程碑落地后同步 `ONBOARDING.md` §二/§七。
6. push 后确认 CI 终态（矩阵测 3.10–3.14，本地 Python 版本不代表
   全矩阵）；master 红了优先修或 revert。
