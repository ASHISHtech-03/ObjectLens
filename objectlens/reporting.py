"""Stage 6 - visual annotation and export (image, CSV, JSON, text summary, chart)."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .pipeline import AnalysisResult

LABEL_COLORS = {
    "circle": (0, 0, 200), "ellipse": (200, 0, 200), "triangle": (0, 140, 255),
    "square": (200, 120, 0), "rectangle": (0, 160, 0), "pentagon": (160, 0, 100),
    "hexagon": (0, 200, 200), "star": (30, 30, 30), "unknown": (120, 120, 120),
}


def annotate(result: AnalysisResult) -> np.ndarray:
    """Draw contours, ids and labels (with size when calibrated) on a copy."""
    canvas = result.image.copy()
    for o in result.objects:
        color = LABEL_COLORS.get(o.label, (90, 90, 90))
        cv2.drawContours(canvas, [o.region.contour], -1, color, 2, cv2.LINE_AA)
        text = f"#{o.id} {o.label}"
        if o.is_reference:
            text += " (ref)"
        elif o.measurement:
            text += f" d={o.measurement.equivalent_diameter_mm:.0f}mm"
        x, y, w, h = o.features.bbox
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 4, y - 6)
        cv2.rectangle(canvas, (x, ty - th - 4), (x + tw + 6, ty + 3), color, -1)
        cv2.putText(canvas, text, (x + 3, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def to_rows(result: AnalysisResult) -> List[Dict]:
    rows = []
    for o in result.objects:
        f, m = o.features, o.measurement
        rows.append({
            "id": o.id, "label": o.label, "confidence": round(o.confidence, 3),
            "color": f.color_name, "area_px": round(f.area, 1),
            "perimeter_px": round(f.perimeter, 1),
            "centroid_x": round(f.centroid[0], 1), "centroid_y": round(f.centroid[1], 1),
            "length_mm": round(m.length_mm, 2) if m else "",
            "width_mm": round(m.width_mm, 2) if m else "",
            "area_mm2": round(m.area_mm2, 2) if m else "",
            "is_reference": o.is_reference,
        })
    return rows


def summarize(result: AnalysisResult) -> Dict:
    """Aggregate statistics (reference marker excluded from the counts)."""
    objs = [o for o in result.objects if not o.is_reference]
    by_shape = Counter(o.label for o in objs)
    by_color = Counter(o.features.color_name for o in objs)
    summary: Dict = {
        "image": result.image_name, "total_objects": len(objs),
        "by_shape": dict(sorted(by_shape.items())),
        "by_color": dict(sorted(by_color.items())),
        "pixels_per_mm": round(result.pixels_per_mm, 4) if result.pixels_per_mm else None,
    }
    measured = [o.measurement for o in objs if o.measurement]
    if measured:
        areas = [m.area_mm2 for m in measured]
        summary["area_mm2"] = {"min": round(min(areas), 2), "max": round(max(areas), 2),
                               "mean": round(float(np.mean(areas)), 2),
                               "total": round(float(np.sum(areas)), 2)}
    return summary


def write_csv(result: AnalysisResult, path: Path) -> None:
    rows = to_rows(result)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(result: AnalysisResult, path: Path) -> None:
    payload = {"summary": summarize(result), "objects": to_rows(result)}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_summary(result: AnalysisResult) -> str:
    s = summarize(result)
    lines = [f"Image          : {s['image']}", f"Objects found  : {s['total_objects']}"]
    if s["by_shape"]:
        lines.append("By shape       : " + ", ".join(f"{k}={v}" for k, v in s["by_shape"].items()))
        lines.append("By colour      : " + ", ".join(f"{k}={v}" for k, v in s["by_color"].items()))
    if s["pixels_per_mm"]:
        lines.append(f"Calibration    : {s['pixels_per_mm']} px/mm")
    if "area_mm2" in s:
        a = s["area_mm2"]
        lines.append(f"Area (mm^2)    : min {a['min']}, mean {a['mean']}, max {a['max']}, "
                     f"total {a['total']}")
    return "\n".join(lines)


def save_summary_chart(result: AnalysisResult, path: Path) -> None:
    """Bar chart of the object count per shape class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summarize(result)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    names, vals = list(s["by_shape"].keys()), list(s["by_shape"].values())
    ax.bar(names, vals, color="#3b6fb6")
    ax.set_ylabel("count")
    ax.set_title(f"Objects per shape - {result.image_name}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_outputs(result: AnalysisResult, out_dir: str, stem: str) -> Dict[str, Path]:
    """Write annotated image, mask, CSV, JSON and chart; returns the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "annotated": out / f"{stem}_annotated.png", "mask": out / f"{stem}_mask.png",
        "csv": out / f"{stem}_objects.csv", "json": out / f"{stem}_report.json",
        "chart": out / f"{stem}_chart.png",
    }
    cv2.imwrite(str(paths["annotated"]), annotate(result))
    if result.mask is not None:
        cv2.imwrite(str(paths["mask"]), result.mask)
    write_csv(result, paths["csv"])
    write_json(result, paths["json"])
    if result.objects:
        save_summary_chart(result, paths["chart"])
    return paths
