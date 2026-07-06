# 双机分工计划（Task Split）

写于 2026-07-06，由 Windows 台式机（WSL2 + RTX 5070 Ti）侧会话规划。
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

### T1 `matter_slab` 场景元素（M2b，roadmap 下一项）

把已真实化的 Geant4 Matter 适配器接入声明式场景层：新增
`matter_slab` 元素类型，场景 YAML 里一块 NIST 材料板即可参与管线。

- 现有基础：`latent_dirac/adapters/geant4/adapter.py` 的
  `Geant4MatterAdapter`（pydantic 配置模型，子进程 + 相空间 CSV 契约，
  `# complete = true` 完成标记守卫）。**适配器本体不要改**——只做
  schema 接线。
- 改动面：`latent_dirac/scene/schema.py`（新元素模型，
  `extra="forbid"`）、`latent_dirac/scene/build.py`（编译成 Stage
  action 闭包）、**`latent_dirac/viz/scene_3d.py`**（roadmap 2b 规则：
  每个元素类型必须有 3D 表示——`FIDELITY_LABELS` 加条目、
  `_element_segments` 加几何，否则新元素在渲染里**静默隐形**，
  不会报错提醒你）、`docs/scene_schema.md`、README/roadmap/CHANGELOG
  状态同步（同一 commit，见 §六）。
- 测试完全无需引擎：照抄 `tests/test_geant4_matter_adapter.py` 的
  Python stub transformer 模式，Mac 上全套可跑。
- JAX 侧**零工作量**：`jax_scene._base_params` 的 `_SWEEPABLE_PARAMS`
  白名单会自动拒绝未知元素类型（"not supported by the JAX backend
  yet"）。加一个断言该 ValueError 的测试即可，**不要碰镜像对的元素
  循环**。
- spec 里要定的设计点：transformer 可执行文件路径如何进场景
  （二进制路径是机器相关的——建议 run 时注入/环境变量，**不要**硬编码
  进可提交的 YAML）；板的材料/厚度/位置参数命名；引擎四元组溯源如何
  进场景报告。先读
  `docs/superpowers/specs/2026-07-05-geant4-matter-adapter-design.md`。

### T2 Xsuite lattice 场景元素

把已真实化的 Xsuite 适配器接入场景层：场景里声明一段 `xtrack.Line`
追踪。

- 现有基础：`latent_dirac/adapters/xsuite/adapter.py`
  （`ParticleState` ↔ `xtrack.Particles`，`ReferenceFrame(p0c_ev)`
  永远显式，`xsuite_tracking_stage` 已带账本盖章）。适配器本体不改。
- 设计点：场景如何引用 Line（xtrack 的 line JSON 文件路径是惯例）、
  参考动量 p0c 放哪里声明。JAX 侧同 T1 自动拒绝。
- 先读 `docs/superpowers/specs/2026-07-05-xsuite-adapter-design.md`。

### T3 CST / SIMION 场图导入格式

扩展 `latent_dirac/fields/field_map.py`：现有 COMSOL 规则网格 CSV
导入是模板（trilinear 插值、界外返回零）。新增 CST 和 SIMION 的
导出格式解析，table-based 层级，测试用小型合成 fixture 文件。
纯解析工作，完全自包含。spec 里先做格式调研并记录来源。

### T4 残余气体湮灭寿命模型（先写 spec 再动代码）

正电子/反质子在残余气体上的湮灭损失——参数化压强-寿命模型，作为
输运中的连续损失或独立元素（设计点）。纯 NumPy，parameterized 层级，
`assumptions` 与 `physics_note` 按源模型惯例写进 metadata。物理参数
化需要文献调研（截面/寿命标度），把出处写进 spec。

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
| Mac | `scene/schema.py`、`scene/build.py`、`viz/scene_3d.py`（T1/T2 期间）、`fields/field_map.py`（T3）、T4 新模型落点以其 spec 定（`beamline/` 或 `fields/`，定了在此表补一行）、`docs/` 站点 |
| Windows | `backends/`、`engine/`、`adapters/geant4/`、benchmark/CI 基础设施 |
| 谁都不动 | `geant4-v11.4.2/`（只读）；README 大结构（owner 在 GitHub 网页端直接改）；安全范围三处钉死（改动先出 positioning spec） |

跨界需求（比如 T1 发现适配器要加参数）：先和 owner 打招呼，
不要静默跨界提交。

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
