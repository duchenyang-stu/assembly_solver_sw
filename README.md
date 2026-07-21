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

## 从模型预测选择约束并求解

`predictions_test.json` 中 `predictions` 往往比 GT 多。可以用 `prediction_run.py`
先从预测约束池里选出一个小而自洽的子集，再走和 GT 批处理相同的
`load_assembly_payload -> collect_pair_jobs -> solve_jobs -> save_solution_step`
流程：

```bash
conda run -n catia_assembly python -m reconstructed_solver.prediction_run \
  --predictions-json /home/xiazhen/cad/AssemblyTry/pipline_model/reconstructed_solver/predictions_test.json \
  --json-root /home/xiazhen/cad/Assembly/data/new_sw_final_assemblies/max_faces_50/pair \
  --step-root /home/xiazhen/cad/Assembly/data/new_sw_final_assemblies/max_faces_50/pair/step \
  --output-dir /home/xiazhen/cad/AssemblyTry/pipline_model/reconstructed_solver/prediction_output \
  --workers 4
```

约束选择策略分两层：

- 先做轻量筛选：只保留当前求解器支持的 `Coincident`、`Concentric`、
  `Parallel`、`Perpendicular`，按 `(part_a, face_a, part_b, face_b, type)`
  去重，丢掉低于 `--score-threshold` 的预测，再最多保留
  `--max-predictions` 个候选。
- 再用求解器验证：在候选池中构造若干 1 到 `--max-constraints` 个约束的
  子集，优先尝试高分且包含定位关系的组合，然后逐个调用现有 `solve_jobs`。
  第一个满足求解状态和选择策略的子集会被选中；如果没有完全 OK 的子集，
  会保留最佳失败/碰撞诊断，便于调阈值。

常用调参：

```bash
conda run -n catia_assembly python -m reconstructed_solver.prediction_run \
  --limit 100 \
  --score-threshold 0.5 \
  --max-predictions 16 \
  --beam-size 24 \
  --max-constraints 3
```

默认情况下，如果候选池里存在 `Coincident` 或 `Concentric`，脚本不会把纯
`Parallel/Perpendicular` 子集当成最终成功结果，因为这类约束可能只确定方向、
不确定装配位置。确实需要接受方向约束单独求解时，可以显式添加：

```bash
--allow-direction-only
```

每个样本会输出一个结果 JSON，其中 `prediction_selection.pool` 记录进入候选池
的预测，`prediction_selection.attempts` 记录每个候选子集的求解状态，
`prediction_selection.selected_predictions` 是最终选中的预测约束。总汇总写入
`prediction_results.json`，成功或可导出的 STEP 与 GT 批处理输出格式保持一致。
