#!/usr/bin/env python3
"""
物理 3D/4D WAM Benchmark — 批量评测 driver (面向任务式服务器)

设计目标 (任务式服务器: 提交一次 → 跑完即释放):
- 模型只初始化 **一次**, 全程复用 (懒加载: 用到哪个加载哪个, 之后缓存复用)
- 遍历数据清单 (manifest) 中的全部视频, 逐个评测
- 单个视频失败不中断整体 (记录错误, 继续下一个)
- 每个视频生成独立报告 + 全部视频汇总成一份 batch_summary

用法:
    # 全部模块 (显存充足时, 单卡建议 ≥40GB)
    python run_batch.py --manifest data/manifest.jsonl

    # 按模块分组跑 (规避单卡 OOM: 每次任务只加载该组模型)
    python run_batch.py --manifest data/manifest.jsonl --group geometry
    python run_batch.py --manifest data/manifest.jsonl --group vlm
    python run_batch.py --manifest data/manifest.jsonl --group dynamics

    # 指定任意模块子集
    python run_batch.py --manifest data/manifest.jsonl --modules kinematics dynamics

    # 离线模式 (权重已下好, 不联网)
    HF_HUB_OFFLINE=1 python run_batch.py --manifest data/manifest.jsonl

清单格式见 data/README.md。
"""
import argparse
import datetime
import json
import os
import sys
import traceback
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_physics import PhysicsPipeline
from models.loader import ModelLoader


# ── 模块分组 (按共享模型/显存占用划分, 用于 --group 规避单卡 OOM) ──
# 每组内的模块共享大部分模型, 一次任务只加载该组模型即可跑完全部视频。
MODULE_GROUPS = {
    # 几何重建类: VGGT-Omega / StableNormal / DA3 / SAM3 / Any6D
    "geometry": ["geometry", "scale_pose", "deformation_spatial",
                 "statics_lighting", "multiview"],
    # VLM 语义评判类: 仅 Qwen3-VL
    "vlm": ["material", "thermal", "electromag", "phase_change"],
    # 动力学/时序类: SpatialTracker V2 / FlowSeek / VFIMamba (+VGGT/SAM3)
    "dynamics": ["kinematics", "dynamics", "motion_persistence"],
}

ALL_MODULES = [
    "geometry", "scale_pose", "deformation_spatial", "statics_lighting",
    "material", "thermal", "electromag", "phase_change",
    "kinematics", "motion_persistence", "dynamics", "multiview",
]

# manifest gt 子字段 → pipeline.run kwargs 键名映射
GT_KEY_MAP = {
    "depth": "depth_gt",
    "normals": "normal_gt",
    "pointcloud": "pointcloud_gt",
    "deformation": "gt_deformation",
    "trajectories": "gt_trajectories",
    "velocities": "gt_velocities",
    "angular_vel": "gt_angular_vel",
    "object_poses": "gt_poses",
    "poses": "gt_poses",  # 向后兼容
    "sizes": "gt_sizes",
    "masks": "gt_masks",
    # invalid CI2V 可提供与物理GT mask分离的“视频观测目标mask”，
    # 仅用于给SpatialTracker发起query，不参与物理真值比较。
    "tracking_masks": "gt_tracking_masks",
    "camera_intrinsics": "camera_intrinsics",
    "camera_extrinsics": "camera_poses",
    "frame_indices": "gt_frame_indices",
    "timestamps": "gt_timestamps",
}


# ============================================================
# 日志落盘: 把 stdout/stderr 同时 tee 到终端 + 日志文件
# ============================================================
class _Tee:
    """把 print 输出同时写到原流和日志文件（含 SAM3/DA3 等模型的调试输出）。"""

    def __init__(self, stream, file_handle):
        self.stream = stream
        self.file = file_handle

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def fileno(self):
        return self.stream.fileno()


def setup_logging(out_dir: str):
    """在输出目录下创建 run.log，把 stdout/stderr 同时写入终端与文件。

    幂等: 重复调用不会重复包装。返回日志文件路径。
    """
    if getattr(sys.stdout, "_is_teed", False):
        return os.path.join(out_dir, "run.log")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    # 追加模式: 断点续跑(--resume)时保留历史日志
    f = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    sys.stdout._is_teed = True
    sys.stderr._is_teed = True
    print(f"[Log] 日志已记录到 {log_path}")
    return log_path


# ============================================================
# 清单加载
# ============================================================
def load_manifest(path: str, data_root: str) -> list[dict]:
    """
    加载数据清单, 支持:
    - .jsonl : 每行一个 JSON 对象 (推荐)
    - .json  : 一个 JSON 数组
    - .csv   : 表头 video,modality,scene_type,ref_image,prompt,modules

    返回规范化后的 entry 列表, 每条含:
      video(绝对路径), modality, scene_type, ref_image, prompt,
      expected_materials, modules(该视频专属或 None), gt(原始子字典)
    """
    entries: list[dict] = []

    def _abspath(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        return p if os.path.isabs(p) else os.path.join(data_root, p)

    raw_list: list[dict] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                raw_list.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
    elif path.endswith(".csv"):
        import csv
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("modules"):
                    row["modules"] = row["modules"].split("|")
                raw_list.append(row)
    else:
        raise ValueError(f"不支持的清单格式: {path} (需 .jsonl / .json / .csv)")

    for raw in raw_list:
        video = raw.get("video") or raw.get("video_path")
        if not video:
            print(f"[Manifest] 跳过缺少 video 字段的条目: {raw}")
            continue
        mods = raw.get("modules")
        if isinstance(mods, str):
            mods = [m.strip() for m in mods.split("|") if m.strip()]
        dims = raw.get("dimensions") or []
        if isinstance(dims, str):
            dims = [d.strip() for d in dims.split("|") if d.strip()]
        # 当前Manifest统一使用连续R01-R25；旧历史Manifest需要先显式迁移，
        # 不能按编号盲目自动转换，因为旧R02与新R02语义不同。
        modality = (raw.get("modality") or "T2V").upper()
        if modality not in {"T2V", "I2V", "CI2V"}:
            raise ValueError(f"未知 modality={modality}; 仅支持 T2V/I2V/CI2V")
        entries.append({
            "video": _abspath(video),
            "modality": modality,
            "scene_type": raw.get("scene_type") or "unspecified",
            "ref_image": _abspath(raw.get("ref_image")),
            "multiview_images": [_abspath(p) for p in raw.get("multiview_images", [])],
            "prompt_id": raw.get("prompt_id") or raw.get("id") or "",
            "prompt": raw.get("i2v_prompt") or raw.get("t2v_prompt") or raw.get("prompt") or "",
            # eval_prompt: 物理层判据专用的场景描述 prompt (如 I2V 条目的 T2V 原版,
            # 含物体颜色/前后关系声明)。I2V 改写 prompt 丢失这些声明时用它恢复
            # R01/R11 等维度的设计判据; L1 指令遵循仍用原 prompt (真实生成指令)。
            "eval_prompt": raw.get("eval_prompt") or "",
            "t2i_prompt": raw.get("t2i_prompt") or "",
            "expected_materials": raw.get("expected_materials"),
            "storyboard": raw.get("storyboard") or {"enabled": False, "mode": "continuous"},
            "evaluation_method": raw.get("evaluation_method", "default"),
            "panel_layout": raw.get("panel_layout"),
            "evaluator_context": raw.get("evaluator_context") or {},
            "modules": mods,
            "dimensions": dims,
            "gt": raw.get("gt") or {},
        })
    return entries


def load_gt_kwargs(gt_spec: dict, data_root: str) -> dict:
    """把 manifest 的 gt 文件路径读成 numpy/json, 转成 pipeline.run 的 kwargs"""
    kwargs = {}
    for sub_key, run_key in GT_KEY_MAP.items():
        p = gt_spec.get(sub_key)
        if not p:
            continue
        if not os.path.isabs(p):
            p = os.path.join(data_root, p)
        if not os.path.exists(p):
            print(f"    [GT] 文件不存在, 跳过: {p}")
            continue
        try:
            if p.endswith(".npy"):
                kwargs[run_key] = np.load(p)
            elif p.endswith(".json"):
                with open(p, "r", encoding="utf-8") as f:
                    kwargs[run_key] = json.load(f)
            print(f"    [GT] {run_key} ← {p}")
        except Exception as e:
            print(f"    [GT] 加载失败 {p}: {e}")
    return kwargs


# ============================================================
# 跨视频聚合
# ============================================================
def _result_has_scored_dimensions(result: dict) -> bool:
    """判断一个视频是否至少有一个可计分维度。

    BenchmarkResult.overall_score() 对全 skipped/error 层返回0.0；0不是物理分数，
    不能进入总体均值，否则数据不足的样本会被错误当成0分。
    """
    for layer in (result or {}).get("layers", []):
        for item in layer.get("results", []):
            if item.get("status") in ("ok", "proxy", "degraded"):
                return True
    return False


def aggregate(records: list[dict]) -> dict:
    """
    records: [{"entry":..., "result": BenchmarkResult.to_dict() or None, "error": str}]
    汇总: 每维度均分/分布、每层均分、总体均分、error/skipped 统计
    """
    n_total = len(records)
    n_ok = sum(1 for r in records if r.get("result"))
    n_failed = n_total - n_ok

    # 每维度收集分数 (仅计入 counts_toward_score 的)
    dim_scores: dict[str, list[int]] = {}
    dim_status_count: dict[str, dict[str, int]] = {}
    overall_scores: list[float] = []

    for rec in records:
        res = rec.get("result")
        if not res:
            continue
        if (isinstance(res.get("overall_score"), (int, float))
                and _result_has_scored_dimensions(res)):
            overall_scores.append(res["overall_score"])
        for layer in res.get("layers", []):
            for d in layer.get("results", []):
                dim = d["dimension"]
                status = d.get("status", "ok")
                dim_status_count.setdefault(dim, {})
                dim_status_count[dim][status] = dim_status_count[dim].get(status, 0) + 1
                if status in ("ok", "proxy", "degraded"):
                    dim_scores.setdefault(dim, []).append(d["score"])

    dim_summary = {}
    for dim, scores in sorted(dim_scores.items()):
        dim_summary[dim] = {
            "mean_score": round(float(np.mean(scores)), 3) if scores else None,
            "n_scored": len(scores),
            "status_count": dim_status_count.get(dim, {}),
        }
    # 也纳入从未被打分(全 error/skipped)的维度
    for dim, sc in dim_status_count.items():
        if dim not in dim_summary:
            dim_summary[dim] = {"mean_score": None, "n_scored": 0, "status_count": sc}

    n_with_scores = sum(
        1 for r in records if r.get("result") and
        _result_has_scored_dimensions(r["result"])
    )
    n_without_scores = n_ok - n_with_scores
    n_skipped = sum(
        count for sc in dim_status_count.values()
        for status, count in sc.items() if status == "skipped"
    )
    n_unobservable = sum(
        count for sc in dim_status_count.values()
        for status, count in sc.items() if status == "unobservable"
    )
    return {
        "n_videos_total": n_total,
        # 保留旧字段语义: pipeline成功处理的视频数
        "n_videos_scored": n_ok,
        "n_videos_with_scores": n_with_scores,
        "n_videos_without_scores": n_without_scores,
        "n_videos_failed": n_failed,
        "n_skipped_dimension_results": n_skipped,
        "n_unobservable_dimension_results": n_unobservable,
        "overall_mean_score": round(float(np.mean(overall_scores)), 3) if overall_scores else None,
        "per_dimension": dim_summary,
    }


def write_batch_summary(records: list[dict], summary: dict, out_dir: str,
                        meta: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    # JSON
    payload = {
        "meta": meta,
        "summary": summary,
        "per_video": [
            {
                "video": os.path.basename(rec["entry"]["video"]),
                "modality": rec["entry"]["modality"],
                "scene_type": rec["entry"]["scene_type"],
                "overall_score": (rec["result"] or {}).get("overall_score"),
                "error": rec.get("error"),
            }
            for rec in records
        ],
    }
    json_path = os.path.join(out_dir, "batch_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Markdown
    lines = [
        "# Physical World Benchmark — 批量评测汇总",
        f"**时间**: {meta['timestamp']}  |  **模块**: {', '.join(meta['modules'])}",
        "",
        "## 总览",
        f"- 视频总数: {summary['n_videos_total']}",
        f"- 视频处理完成: {summary['n_videos_scored']}",
        f"- 其中含有效维度分数: {summary['n_videos_with_scores']}",
        f"- 无有效维度分数: {summary['n_videos_without_scores']}",
        f"- 运行失败: {summary['n_videos_failed']}",
        f"- 维度级 skipped: {summary['n_skipped_dimension_results']}",
        f"- 维度级 unobservable: {summary['n_unobservable_dimension_results']}",
        f"- **有效样本总体平均分**: {summary['overall_mean_score']} / 5.00",
        "",
        "## 各物理维度平均分 (仅计入 ok/proxy/degraded)",
        "",
        "| 维度 | 平均分 | 有效样本 | 状态分布 |",
        "|------|--------|----------|----------|",
    ]
    for dim, info in summary["per_dimension"].items():
        ms = info["mean_score"]
        ms_str = f"{ms:.2f}" if isinstance(ms, (int, float)) else "-"
        sc = ", ".join(f"{k}:{v}" for k, v in info["status_count"].items())
        lines.append(f"| {dim} | {ms_str} | {info['n_scored']} | {sc} |")

    lines += ["", "## 逐视频得分", "",
              "| 视频 | 模态 | 场景 | 总分 | 备注 |",
              "|------|------|------|------|------|"]
    for rec in records:
        e = rec["entry"]
        ov = (rec["result"] or {}).get("overall_score")
        ov_str = f"{ov:.2f}" if isinstance(ov, (int, float)) else "-"
        note = rec.get("error") or ""
        lines.append(f"| {os.path.basename(e['video'])} | {e['modality']} "
                     f"| {e['scene_type']} | {ov_str} | {note[:60]} |")

    md_path = os.path.join(out_dir, "batch_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_dir


# ============================================================
# 主流程
# ============================================================
def resolve_modules(args) -> list[str]:
    if args.modules:
        return args.modules
    if args.group:
        return MODULE_GROUPS[args.group]
    return None  # None = 用 config 中 enabled 的全部


# ============================================================
# 断点续跑 (checkpoint)
# ============================================================
def _checkpoint_path(out_dir: str) -> str:
    return os.path.join(out_dir, "checkpoint.jsonl")


def load_checkpoint(out_dir: str) -> dict[str, dict]:
    """读取已完成的视频结果。返回 {video绝对路径: rec}。

    checkpoint.jsonl 每行一个 {video, result, error}。
    用于 --resume 时跳过已完成视频并保留其结果。
    """
    done: dict[str, dict] = {}
    ckpt = _checkpoint_path(out_dir)
    if not os.path.exists(ckpt):
        return done
    with open(ckpt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            video = rec.get("video")
            if video:
                done[video] = {
                    "result": rec.get("result"),
                    "error": rec.get("error"),
                }
    return done


def append_checkpoint(out_dir: str, rec: dict) -> None:
    """每处理完一个视频, 追加一行到 checkpoint (原子性靠单行 append)。"""
    os.makedirs(out_dir, exist_ok=True)
    with open(_checkpoint_path(out_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="物理层批量评测 (任务式服务器: 一次加载, 跑完全部视频)")
    parser.add_argument("--manifest", "-m", required=True,
                        help="数据清单文件 (.jsonl / .json / .csv)")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--data-root", default=None,
                        help="视频/GT 相对路径的根目录 (默认=清单所在目录)")
    parser.add_argument("--output", "-o", default=None,
                        help="批量汇总输出目录 (默认=config.output_dir/batch_<时间>)")
    parser.add_argument("--group", choices=list(MODULE_GROUPS.keys()),
                        help="按预设分组跑 (规避单卡 OOM): geometry/vlm/dynamics")
    parser.add_argument("--modules", nargs="+", choices=ALL_MODULES,
                        help="指定任意模块子集 (优先于 --group)")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评测前 N 个视频 (调试用)")
    parser.add_argument("--start", type=int, default=0,
                        help="从第 N 个视频开始 (跳过前 N-1 个; 与 --resume 互斥)")
    parser.add_argument("--resume", default=None, metavar="DIR",
                        help="断点续跑: 指定上次的批量输出目录, 跳过其中已完成的视频")
    parser.add_argument("--offline", action="store_true",
                        help="离线模式 (设 HF_HUB_OFFLINE=1, 权重须已下好)")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if not os.path.exists(args.manifest):
        print(f"[Error] 清单不存在: {args.manifest}")
        sys.exit(1)

    data_root = args.data_root or os.path.dirname(os.path.abspath(args.manifest))
    entries = load_manifest(args.manifest, data_root)
    if args.limit:
        entries = entries[:args.limit]
    if not entries:
        print("[Error] 清单为空")
        sys.exit(1)

    modules = resolve_modules(args)
    print(f"[Batch] 清单={args.manifest} | 视频数={len(entries)} | "
          f"模块={'全部(config)' if modules is None else modules}")

    if args.resume and args.start:
        print("[Error] --resume 与 --start 互斥, 请二选一")
        sys.exit(1)

    # ── 初始化管线 (关键: 直接赋 loader, 绕过 initialize() 的全量预加载,
    #    改为各 evaluator 懒加载 → 用到哪个加载哪个, 全程缓存复用) ──
    pipeline = PhysicsPipeline(config_path=args.config)
    pipeline.loader = ModelLoader(pipeline.config)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # --resume 时复用原输出目录 (checkpoint + 结果追加到同一目录);
    # 否则新建带时间戳的输出目录。
    out_dir = args.output or (args.resume or os.path.join(
        pipeline.output_dir, f"batch_{args.group or 'all'}_{ts}"))

    # ── 日志落盘: 后续所有 print 输出同时写入终端与 out_dir/run.log ──
    setup_logging(out_dir)

    # ── 断点续跑: 读取已完成视频, 合并进 records ──
    records: list[dict] = []
    done: dict[str, dict] = {}
    if args.resume:
        done = load_checkpoint(args.resume)
        if done:
            print(f"[Resume] 从 {args.resume} 恢复 {len(done)} 个已完成视频, "
                  f"本轮跳过它们")
    skip_start = args.start if args.start else 0

    for i, e in enumerate(entries, 1):
        # --start: 跳过前 N-1 个; --resume: 跳过已完成视频
        if i < skip_start:
            continue
        if e["video"] in done:
            # 恢复已完成视频的结果 (checkpoint 只存 video/result/error)
            records.append({"entry": e, "result": done[e["video"]]["result"],
                            "error": done[e["video"]]["error"]})
            continue
        print(f"\n{'#'*70}\n[Batch {i}/{len(entries)}] {e['video']} "
              f"({e['modality']}, {e['scene_type']})\n{'#'*70}")
        rec = {"entry": e, "result": None, "error": None}

        if not e["video"] or not os.path.exists(e["video"]):
            rec["error"] = f"视频不存在: {e['video']}"
            print(f"  [Skip] {rec['error']}")
            records.append(rec)
            append_checkpoint(out_dir, {
                "video": e["video"], "result": rec["result"], "error": rec["error"]})
            continue

        # CLI --modules 明确指定时必须优先于 manifest 的 per-video modules。
        # 旧逻辑使用 e["modules"] or modules，导致
        #   --modules geometry
        # 实际仍按清单逐视频跑全模块，geometry 子集报告口径失真。
        # 未指定 CLI 模块时才使用 manifest 的模块列表。
        vid_modules = modules if modules is not None else e["modules"]

        # 组装 kwargs
        run_kwargs = {
            "scene_type": e["scene_type"],
            "storyboard": e["storyboard"],
            "evaluation_method": e["evaluation_method"],
            "panel_layout": e["panel_layout"],
            "evaluator_context": e["evaluator_context"],
            "multiview_image_paths": e["multiview_images"],
        }
        if e["expected_materials"]:
            run_kwargs["expected_materials"] = e["expected_materials"]
        external_gt = load_gt_kwargs(e["gt"], data_root)
        if external_gt:
            run_kwargs.update(external_gt)
            run_kwargs["_gt_source"] = "external_simulator"

        try:
            result = pipeline.run(
                video_path=e["video"],
                modules=vid_modules,
                target_dimensions=e.get("dimensions") or None,
                ref_image_path=e["ref_image"],
                modality=e["modality"],
                # 物理层内容判据(概念筛选/关系提取/PreCheck)优先用 eval_prompt
                # (场景描述), 缺省回退真实生成指令 prompt。
                prompt=e.get("eval_prompt") or e["prompt"],
                **run_kwargs,
            )
            result_dict = result.to_dict()
            # 全 skipped/error 时的overall_score=0仅表示"无可评分结果"，
            # 不是物理维度0分；报告中用null/"-"表示，避免误导聚合。
            if not _result_has_scored_dimensions(result_dict):
                result_dict["overall_score"] = None
            rec["result"] = result_dict
        except Exception as ex:
            rec["error"] = f"{type(ex).__name__}: {ex}"
            print(f"  [Error] 视频评测失败: {rec['error']}")
            traceback.print_exc()

        records.append(rec)
        # 断点记录: 每完成一个视频即落盘, 中断后 --resume 可跳过
        append_checkpoint(out_dir, {
            "video": e["video"], "result": rec["result"], "error": rec["error"]})

    # ── 汇总 ──
    summary = aggregate(records)
    meta = {
        "timestamp": ts,
        "manifest": os.path.abspath(args.manifest),
        "modules": modules or ["<config-enabled>"],
        "group": args.group,
    }
    out = write_batch_summary(records, summary, out_dir, meta)

    print(f"\n{'='*70}")
    print("批量评测完成")
    print(f"  视频: {summary['n_videos_scored']}/{summary['n_videos_total']} 成功")
    print(f"  总体平均分: {summary['overall_mean_score']} / 5.00")
    print(f"  汇总报告: {out}/batch_summary.md")
    print(f"  逐视频报告: {os.path.join(pipeline.output_dir, 'physics')}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
