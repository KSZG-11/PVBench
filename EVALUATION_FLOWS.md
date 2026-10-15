# 三类被测模型的评测流程（T2V / I2V / CI2V）

> 配套：`PHYSICS_TAXONOMY.md`（维度定义）、`prompts/physics_taxonomy_prompts.yaml`（prompt 库）、
> `run_batch.py` + `pipeline_physics.py`（执行）、`scripts/gen_physics_manifest.py`（清单生成）。
>
> **benchmark 立场**：我们评测的是「评测方法是否被正确、合理执行」，不评价被评测模型生成内容的
> 好坏。下游报告应聚焦评测器 error / 维度区分度 / 评分噪声 / 评测器一致性，而非「某模型物理更差」。

---

## 0. 统一评分状态语义

每个 `EvalResult` 带 `status`，聚合时只统计 `ok` / `proxy` / `degraded`：

| status | 含义 | 何时产生 | 在报告中 |
|--------|------|---------|---------|
| `ok` | 有外部 GT 的定量误差 | 仅 CI2V 提供仿真真值时 | 计入，带数值误差 |
| `proxy` | 无 GT 的自洽性 / prompt 代理评测 | T2V/I2V 全部维度；CI2V 无 GT 时 | 计入，标注为代理分 |
| `degraded` | 部分模型失败但有降级路径 | 如 Any6D 缺失降级 VGGT 相机位姿 | 计入（降级） |
| `skipped` | 数据不足 | 帧数/物体数不够 | **不计入** |
| `error` | 模型权重缺失或崩溃 | 未部署模型 | **不计入**，需补齐环境 |

**关键纪律**：T2V/I2V 绝不允许把单图模型预测（VGGT/DA3/SAM）当 GT 计分——只能走 proxy。
CI2V 只有仿真导出的外部 GT 才能产出 `status=ok`。

---

## 1. T2V（文本 → 视频）

**输入**：文本 prompt（`t2v_prompt`）。
**生成**：被测 T2V 模型直接生成视频（如 Wan2.1-T2V）。
**评测模式**：全部 25 维走 proxy / 自洽性路径。

流程：
```
prompts/physics_taxonomy_prompts.yaml (dimensions.<Rxx>.prompts[].t2v_prompt)
        │  scripts/gen_physics_manifest.py --modality T2V
        ▼
data/manifest_t2v.jsonl   (每条: video, modality=T2V, prompt, dimensions, modules)
        │  run_batch.py --manifest data/manifest_t2v.jsonl
        ▼
pipeline_physics.run(video, modality="T2V", prompt=..., modules=<维度对应模块>)
        ▼
各 evaluator 以 proxy/自洽性模式打分 (status=proxy)
```

要点：
- `modules` 由维度→模块映射自动决定（见 `gen_physics_manifest.py` 的 `DIM_TO_MODULE`），
  无需手工指定。
- 7 个 △ 维度（R01/R02/R03/R04/R08/R10/R15）在 T2V 下只能验证「prompt 代理后的自洽性」，
  例如 R10 用「小球水平抛出/单摆」类 prompt，再比对轨迹是否满足该场景物理方程。
- 不传 `ref_image`，参考图保真评测（L2-RF）对 T2V 样本自动跳过。

---

## 2. I2V（首帧图 + 动作 → 视频）

**输入**：首帧构图 prompt（`t2i_prompt`）+ 动作 prompt（`i2v_prompt`）。
**生成**：两步——
  1. 用 T2I 模型由 `t2i_prompt` 生成首帧参考图（仅描述初态，不描述动作结果）；
  2. I2V 模型以首帧图 + `i2v_prompt` 生成视频。
**评测模式**：物理维度同样走 proxy；但额外启用「参考图保真度（L2-RF，仅 I2V 逐样本）」。

流程：
```
prompts/physics_taxonomy_prompts.yaml
   ├─ t2i_prompt ─▶ scripts/generate_i2v_refs.py  (T2I 生成首帧图, 写 ref_image 到 manifest)
   └─ i2v_prompt ─▶ scripts/generate_i2v_videos.py (I2V 以首帧图+动作 prompt 生成视频)
        │
        ▼  manifest 每条携带: ref_image + i2v_prompt + t2i_prompt(供审计)
run_batch.py --manifest data/manifest_i2v.jsonl
        │
        ▼
pipeline_global: 逐样本 modality=I2V → ReferenceFidelityEvaluator 仅对 I2V 样本计分
pipeline_physics: modality="I2V" → 物理维度 proxy 评测
```

要点：
- **首帧 prompt 与动作 prompt 必须正确对应**：`t2i_prompt` 只描述第一帧的物体/场景/光照；
  `i2v_prompt` 描述「之后发生什么」。二者不是同一句话，重复利用会污染 I2V 条件。
- `generate_i2v_refs.py` 已写入 `t2i_prompt`/`i2v_prompt` 双字段；`generate_i2v_videos.py`
  使用 `i2v_prompt`，不会误用首帧 prompt 当动作。
- I2V 的 7 个 △ 维度同样仅 proxy（首帧图是生成条件，不是 GT）。
- 参考图保真评测（IF-10a/c/d）按逐样本 `modality` 触发，**不会**把 I2V 误判为 CI2V 角色一致性。

---

## 3. CI2V（受控输入 → 视频 + 外部 GT）

**输入**：由仿真/引擎导出的视频，以及与之**帧对齐**的外部真值（深度/法线/位姿/轨迹/掩码/相机）。
**生成**：推荐 **Unreal Engine 自动化构建**（见 `CI2V_DATA_BUILD.md`）。
**评测模式**：提供 GT 的维度产出 `status=ok` 的定量误差；未提供 GT 的维度回退 proxy。

流程：
```
UE5 场景模板(按维度) + 自动化相机/刚体脚本
        │  (见 CI2V_DATA_BUILD.md)
        ▼
rendered video (.mp4)  +  GT frames (.npy/.npz/.json: depth/normal/pose/traj/mask/camera)
        │
        ▼  data/ci2v_gt_map.json: { "<prompt_id>": { "depth_gt": ".../depth.npy", ... } }
scripts/gen_physics_manifest.py --modality CI2V --gt-map data/ci2v_gt_map.json
        ▼
data/manifest_ci2v.jsonl  (每条带 gt 字段)
        │  run_batch.py --manifest data/manifest_ci2v.jsonl
        ▼
pipeline_physics.run(video, modality="CI2V", **gt_kwargs)
   ├─ _align_temporal_gt(): 将全帧 GT 按实际抽帧索引同步采样, 避免错帧
   └─ CI2V 外部 GT 集合进入对应 evaluator → status=ok
```

要点：
- CI2V 的 GT 来源**只能是外部仿真**，单图模型预测被明确禁止冒充 GT
  （`pipeline_physics.py` 已删除 `_compute_gt_from_ref`，改为只接受 `manifest.gt`）。
- 需要的 GT 键见 `run_batch.py` 的 `GT_KEY_MAP`：
  `depth_gt / normal_gt / pointcloud_gt / gt_deformation / gt_trajectories / gt_velocities /
   gt_angular_vel / gt_poses / gt_sizes / gt_masks / camera_intrinsics / camera_poses /
   gt_frame_indices / gt_timestamps`。
- 时间对齐：UE 导出全帧 GT，抽样后由 `_align_temporal_gt` 按 `sampler.last_sample_indices`
  同步取帧，保证预测与真值同一时间轴（否则运动学量纲与对错全错）。
- R04 多视图在 CI2V 下使用**同帧多相机原生视图**（`multiview_image_paths`），不再把单视频切面板。

### R04 / R14 时间分镜代理

新版统一数据中，R04/R14 关联样本约三分之一使用不超过3阶段的时间分镜，其余保留连续描述：

- R04 `temporal_multiview_proxy`：按三个阶段中点取顺序视角，VGGT联合重建并检查点云退化和相机视角差异；结果固定为 `proxy`，`time_synchronized=false`，不能替代CI2V同步多机位GT。
- R04 `continuous_view_sequence_proxy`：未分镜环绕样本取首/中/尾视角执行同类代理。
- R14 `storyboard_object_persistence`：用全帧按“初始可见→完整遮挡→远侧观察”三个阶段检查目标召回、合法不可见和遮挡后重检。
- R14 `continuous_object_persistence`：未分镜样本自动寻找中段持续遮挡区间和后段目标重检，不把合法临时遮挡直接计为物体数错误。

分镜结构通过 `storyboard`、`evaluation_method` 和 `evaluator_context` 从生成Manifest传到评测器；R14分镜强制 `full_frame_required=true`。

---

## 4. 一键运行对照

| 模态 | manifest 生成 | 关键参数 | 评测状态 |
|------|--------------|---------|---------|
| T2V | `gen_physics_manifest.py --modality T2V` | — | 全 proxy |
| I2V | `gen_physics_manifest.py --modality I2V [--per-dimension N]` | 需先跑 ref/videos 生成脚本 | proxy + I2V 保真 |
| CI2V | `gen_physics_manifest.py --modality CI2V --gt-map data/ci2v_gt_map.json` | UE 导出 GT | 有 GT 维度 `ok` |

`--per-dimension N` 可控制每个维度抽几条变体（prompt 库数量本就不固定，
该参数用于抽样控制与算力预算）。

---

## 5. 评测器正确性关键约束（本轮审计修复）

> 这些是「评测方法是否成立」的底线，与被测模型无关。

### 5.1 物体级 vs 网格点级语义

`SpatialTracker.track_3d()` 默认铺 20×20=400 个网格查询点。表格对
R10/R11/R15/R23/R16 的要求全部是**物体级**物理量（"不同质量物体""碰撞前后速度"
"Σmᵢvᵢ"）。因此这些维度统一改用 `evaluators/physics/_object_tracks.py`：

```
SAM3 分割首帧 → 每物体掩码内采样 24 点 → 一次性喂 SpatialTracker
→ 按物体聚合质心轨迹 [N_obj, T, 3]
```

- `metrics.track_source` = `sam3_object_centroid`（正常）或 `grid_fallback`（降级为场景单体）。
- **R17 例外**：刚体旋转估计需要稠密非共线点，故仍用 `point_tracks`，这是正确用法。
- 降级时也**绝不**把 400 个点当 400 个物体，而是聚合成1 个场景级轨迹，
  使 `N<2 → skipped` 这类保护重新生效。

### 5.2 禁止恒真检查冒充分数

以下模式已从代码中清除，因为它们让所有样本都拿满分、维度失去区分度：

| 位置 | 原实现 | 现状 |
|------|--------|------|
| R01 深度序 | `n_consistent += 1` 无条件递增，`n_inconsistent` 永不递增 → 恒 5 分 | 真实判据：重叠区深度须更接近前景物体 |
| R01 无物体时 | `valid_ratio` 有效像素率（≈1.0 恒成立） | `skipped`（表格只给了深度序一条代理方案） |
| R01 法线 | `except` 后落`unit_ratio`（单位向量率恒为 1） | 仅用法线-深度梯度一致性 |
| R15 ⑨支撑 | `checks["support"] = 1.0  # 保守给分` | 真实检验触地后是否持续下陷 |
| R11 | 取匀速/匀加速两假设的 **较高分** | 由 `scene_type` 决定用哪个假设 |
| R04尺寸 | 硬阈值 `[0.01m, 20m]`（几乎必过） | Qwen3-VL 合理区间判断 |
| R06 位姿 | 帧间跳变（静止物体恒 5 分） | prompt 位姿变化比对 |

### 5.3 scene_type 必须是物理场景枚举

`gen_physics_manifest.py` 曾把维度中文名（"力的动态效应"）填进 `scene_type`，
导致 R15 永远落到 `unspecified`，表格的 ④斜面/ ⑤弹跳 / ⑧圆周 检验从不激活。
现改为读取 prompt 库的 per-prompt `scene_type` 声明，合法值：

```
free_fall / projectile / bounce / incline / collision /
pendulum / sliding / circular / rolling / unspecified
```

R15 按 `SCENE_CHECKS` 只启用该场景可判定的检验项，
不适用的场景返回 `skipped` 而非硬打分。

### 5.4 量纲一致性

R16 的 `E = KE + g·h` 要求 KE 与 PE 同长度单位。现有两档：

- 有 DA3 → `metric_scale` 把归一化坐标锚定到米，`g=9.8`，`energy_unit=SI`
- 无 DA3 → 用实测重力方向加速度中位数作 `g`，`energy_unit=normalized`

R15 的【度量深度近似数值】档据此验证 `a ≈ g`（容差 [0.5g, 2g]）。

---

## 6. Prompt 与 GT 的构建 pipeline（质量门禁）

benchmark 的 prompt 与 GT 是**评测标尺本身**，标尺不可信则所有分数无意义。
因此两者都必须有强制门禁，而非"写好就用"。

### 6.1Prompt 门禁：`scripts/lint_prompts.py`

```bash
python3 scripts/lint_prompts.py          # error>0 时退出码 2
python3 scripts/lint_prompts.py --strict # warning 也算失败
```

检查项：

| 级别 | 代码 | 含义 |
|---|---|---|
| E | `leak` | **答案泄露**：prompt 写了该维度要评测的正确物理结果 |
| E | `dup_text` | `t2i_prompt` 与 `i2v_prompt` 相同 → I2V 条件被污染 |
| E | `bad_scene` | `scene_type` 不在评测器枚举内 |
| E | `missing` | 缺 id / prompt 文本 |
| W | `no_scene` | 动力学维度未声明 `scene_type` → 场景化检验不激活 |
| W | `vague` | 缺可观测参照（机位/尺度/标记/光源） |

**`leak` 是最关键的一项**：prompt 一旦包含正确结论，被测模型只要"照抄指令"
就能拿高分——此时评测的不再是「模型是否理解物理」而是「模型是否遵循指令」，
该维度分数系统性虚高且失去区分度。

首次上线该门禁即在库中检出**12 处泄露**（已全部改写），典型如：

| 维度 | 泄露原文 | 问题 |
|---|---|---|
| R08 稳定性 | "the block tower **remains as built**" | 塔倒不倒正是判据 |
| R24 能量守恒 | "each bounce **lower than the last**" | 逐次递减正是判据 |
| R21 物体恒存 | "then **reappears** on the right side" | 遮挡后能否重现正是判据 |
| R13 光影空间 | "keep their lit sides **consistent with the left light**" | 光影自洽正是判据 |
| R18 光影时间 | "**keeps the same** highlights and shadows" | 跨帧一致正是判据 |
| R05 相对尺度 | "**growing larger** as it approaches" | 尺寸变化规律正是判据 |
| R17 角速度 | "at a **steady rate**" | ω 恒定正是判据 |

改写原则：**只描述物体与动作的客观设置，不陈述任何应然结果**。
例：R08 从"塔保持原样"改为"镜头静止五秒，无人触碰桌子"。

### 6.2 GT 门禁：`scripts/check_ci2v_gt.py`

```bash
python3 scripts/check_ci2v_gt.py --gt-map data/ci2v_gt_map.json
```

检查键名合法性、文件可读性、形状/量纲合理性、时间轴一致性，并预测
**哪些维度真能拿 `status=ok`**。三类静默失效由它兜住（详见
`CI2V_DATA_BUILD.md` §7.2）：键名写成 pipeline 侧名、逐帧序列未打包、单位未转米。

### 6.3 建议的固定顺序

```
prompt 库改动 → lint_prompts.py 通过 → gen_physics_manifest.py → 生成视频
GT 产出       → check_ci2v_gt.py 通过 → gen_physics_manifest.py --gt-map → run_batch.py
```

任一门禁不通过就不要往下跑——评测一轮的算力成本远高于跑一次lint。


