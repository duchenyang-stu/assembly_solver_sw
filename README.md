# reconstructed_solver

轻量重构版装配求解入口，分成三段：

- `input_loader.py`: 读取 assembly JSON、解析 STEP 路径、收集零件约束对。
- `solve.py`: 调用求解器并记录每对零件的求解状态。
- `visualize.py`: 对成功结果应用 4x4 变换，导出装配 STEP 和 8 个等轴测象限 PNG。

## 运行

```powershell
conda run -n catia_assembly python -m reconstructed_solver.run `
  --assembly-json ".\（5，11）多轴机械臂（已改）1_multi axis robot arm.json" `
  --output-dir ".\reconstructed_solver\output"
```

默认会处理 JSON 中所有跨零件约束对，并输出装配 STEP 以及 `iso_xp_yp_zp` 到 `iso_xn_yn_zn` 的 8 个 PNG，分别对应三维坐标系 8 个 `(±x, ±y, ±z)` 等轴测方向。只求解单个零件对时添加：

```powershell
--fixed-part "Part1.1" --moving-part "3.1"
```

## Solver 模式

默认会先试 SolveSpace，失败后再用解析几何恢复：

```powershell
--solver solvespace-then-analytic
```

可选解析几何求解：

```powershell
--solver analytic
```

如果需要保留严格 SolveSpace 行为：

```powershell
--solver solvespace
```

By default, `solvespace-then-analytic` matches the original behavior: if
SolveSpace returns OK, that result is kept. To additionally reject a SolveSpace
OK result whose diagnostic error is too high, opt in explicitly:

```powershell
--reject-high-error `
--max-error 1e-4
```

`solve_results.json` 中会记录：

- `solver_mode`: 本次运行请求的求解策略。
- `solver_used`: 当前零件对实际使用的求解器，可能是 `solvespace` 或 `analytic`。
- `primary_error`: 仅在 `solvespace-then-analytic` 中出现，表示 SolveSpace 的原始失败原因。
- `max_constraint_error`: 输出 transform 对输入约束的最大诊断误差。
- `collision`: 默认开启的实体干涉诊断；如果存在自由同轴转角，会自动采样旋转避让穿模。

默认会拒绝仍然穿模的结果，这类零件对状态会记录为 `collision`，不会导出装配 STEP/PNG。调试时如果想保留穿模结果，可以显式添加：

```powershell
--allow-interference
```

可调碰撞阈值和自由轴采样数：

```powershell
--common-volume-tolerance 1e-3 `
--contact-tolerance 1e-3 `
--rotation-samples 24
```
