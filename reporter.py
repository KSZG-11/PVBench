"""
报告生成器 — JSON + Markdown + ASCII 雷达图
"""
import os
import json
import datetime
import numpy as np
from typing import Optional
from evaluators.base import BenchmarkResult, LayerResult


class NumpyJSONEncoder(json.JSONEncoder):
    """自动转换 numpy 类型为 JSON 兼容类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Reporter:
    """评测报告生成器"""

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, result: BenchmarkResult,
                 output_prefix: str = "benchmark") -> str:
        """
        生成完整报告

        Returns:
            报告目录路径
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, f"{output_prefix}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)

        # JSON 报告
        self._write_json(result, report_dir)

        # Markdown 报告
        self._write_markdown(result, report_dir)

        print(f"\n[Report] 报告已生成: {report_dir}")
        return report_dir

    def _write_json(self, result: BenchmarkResult, report_dir: str):
        """写 JSON 报告"""
        data = {
            "overall_score": result.overall_score(),
            "n_layers": len(result.layers),
            "layers": {}
        }

        for layer in result.layers:
            layer.compute_avg()
            layer_data = {
                "layer": layer.layer,
                "avg_score": layer.avg_score,
                "n_results": len(layer.results),
                "results": []
            }
            for r in layer.results:
                layer_data["results"].append({
                    "dimension": r.dimension,
                    "score": r.score,
                    "status": r.status,
                    "violations": r.violations,
                    "metrics": {
                        k: v for k, v in r.metrics.items()
                        if isinstance(v, (int, float, str, bool, list, dict))
                    }
                })
            layer_data["n_scored"] = len(layer.scored_results())
            layer_data["n_error"] = sum(1 for r in layer.results if r.status == "error")
            layer_data["n_skipped"] = sum(1 for r in layer.results if r.status == "skipped")

            if layer.aggregates:
                layer_data["aggregates"] = layer.aggregates
            if layer.mean_scores:
                layer_data["mean_scores"] = layer.mean_scores

            data["layers"][layer.layer] = layer_data

        path = os.path.join(report_dir, "report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)

    def _write_markdown(self, result: BenchmarkResult, report_dir: str):
        """写 Markdown 报告"""
        lines = [
            "# Physical 3D/4D WAM Benchmark Report",
            f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"## Overall Score: {result.overall_score():.2f} / 5.00",
            "**Dimension schema**: Unified-46D-v2 (L1=14, L2=7, L3=25; total=46)",
            "",
        ]

        for layer in result.layers:
            layer.compute_avg()
            lines.append(f"## {layer.layer}")
            lines.append(f"**Average Score**: {layer.avg_score:.2f}")
            lines.append(f"**Results**: {len(layer.results)}")
            lines.append("")

            # 排序: 高分在前
            sorted_results = sorted(layer.results,
                                    key=lambda r: r.score, reverse=True)

            lines.append("| Dimension | Score | Status | Violations |")
            lines.append("|-----------|-------|--------|-----------|")

            for r in sorted_results:
                v_str = "; ".join(r.violations[:2]) if r.violations else "✓"
                v_str = v_str[:80]
                score_str = f"{r.score}/5" if r.counts_toward_score else "-"
                lines.append(f"| {r.dimension} | {score_str} | {r.status} | {v_str} |")

            lines.append("")

            # 聚合统计
            if layer.aggregates:
                lines.append("### Distribution")
                for key, dist in layer.aggregates.items():
                    if isinstance(dist, dict):
                        total = sum(dist.values())
                        if total > 0:
                            lines.append(f"**{key}**:")
                            for score_bin, count in sorted(dist.items()):
                                bar = "█" * int(count / max(total, 1) * 20)
                                lines.append(f"- Score {score_bin}: {count} {bar}")
                lines.append("")

        path = os.path.join(report_dir, "report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def print_layer_summary(layer: LayerResult):
        """打印层摘要"""
        layer.compute_avg()
        violations_count = sum(len(r.violations) for r in layer.results)
        print(f"  [{layer.layer}] avg={layer.avg_score:.2f} "
              f"({len(layer.results)} results, {violations_count} violations)")
        # 打印具体分数分布
        if layer.aggregates:
            for key, dist in layer.aggregates.items():
                if isinstance(dist, dict) and "_distribution" in key:
                    dim_name = key.replace("_distribution", "")
                    print(f"    {dim_name}: {dist}")
