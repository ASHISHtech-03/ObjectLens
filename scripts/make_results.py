"""Generate result figures and sample images used by the README and the report.

Run from the project root:  python scripts/make_results.py
(requires a trained model and docs/results/metrics.json - see README)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2                      # noqa: E402
import matplotlib               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np              # noqa: E402

from objectlens.config import Config                       # noqa: E402
from objectlens.pipeline import ObjectLensPipeline         # noqa: E402
from objectlens.preprocessing import preprocess            # noqa: E402
from objectlens.reporting import annotate, save_outputs    # noqa: E402
from objectlens.segmentation import background_distance    # noqa: E402
from objectlens.synthetic import render_scene              # noqa: E402

RES = ROOT / "docs" / "results"
SAMPLES = ROOT / "data" / "samples"
RES.mkdir(parents=True, exist_ok=True)
SAMPLES.mkdir(parents=True, exist_ok=True)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def stages_figure():
    img, _ = render_scene(3, n_objects=7, touching_pairs=1, reference=(30.0, 4.0), theme="light")
    cfg = Config(ref_color="green", ref_side_mm=30.0)
    pipe = ObjectLensPipeline(cfg)
    pre = preprocess(img, cfg)
    dist = background_distance(pre.blurred)
    result = pipe.analyze_image(img, "stages_demo.png")
    panels = [("1. Input image", rgb(img), None), ("2. CLAHE grey image", pre.enhanced, "gray"),
              ("3. Lab distance to background", dist, "magma"),
              ("4. Cleaned binary mask", result.mask, "gray"),
              ("5. Classified + measured", rgb(annotate(result)), None)]
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.9))
    for ax, (title, data, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RES / "pipeline_stages.png", dpi=130)
    plt.close(fig)
    cv2.imwrite(str(SAMPLES / "scene_light.png"), img)


def sample_scenes():
    cfg = Config(ref_color="green", ref_side_mm=30.0)
    pipe = ObjectLensPipeline(cfg)
    for name, seed, theme in (("scene_dark", 12, "dark"), ("scene_cream", 21, "cream")):
        img, _ = render_scene(seed, n_objects=8, touching_pairs=1, reference=(30.0, 4.0), theme=theme)
        cv2.imwrite(str(SAMPLES / f"{name}.png"), img)
        result = pipe.analyze_image(img, f"{name}.png")
        cv2.imwrite(str(RES / f"{name}_annotated.png"), annotate(result))
    result = pipe.analyze_file(SAMPLES / "scene_light.png")
    cv2.imwrite(str(RES / "scene_light_annotated.png"), annotate(result))
    save_outputs(result, str(RES), "scene_light")


def noisy_comparison():
    cfg = Config(ref_color="green", ref_side_mm=30.0)
    pipe = ObjectLensPipeline(cfg)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, sigma in enumerate((5, 25, 45)):
        img, _ = render_scene(31, n_objects=7, reference=(30.0, 4.0), noise=float(sigma), theme="cream")
        res = pipe.analyze_image(img, "n")
        axes[0, col].imshow(rgb(img)); axes[0, col].set_title(f"noise sigma = {sigma}")
        axes[1, col].imshow(rgb(annotate(res))); axes[1, col].set_title(f"{res.count} objects detected")
        for r in (0, 1):
            axes[r, col].axis("off")
    fig.tight_layout()
    fig.savefig(RES / "noise_robustness.png", dpi=110)
    plt.close(fig)


def metric_figures():
    m = json.loads((RES / "metrics.json").read_text())
    clf = m["classifier"]
    labels, cm = clf["labels"], np.array(clf["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (held-out, n={clf['n_test']})")
    fig.tight_layout(); fig.savefig(RES / "confusion_matrix.png", dpi=140); plt.close(fig)

    comp = clf["model_comparison"]
    names = list(comp)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(names, [comp[n]["mean"] * 100 for n in names], yerr=[comp[n]["std"] * 100 for n in names],
           color=["#3b6fb6", "#3b9b6f", "#b6873b", "#999999"], capsize=4)
    ax.set_ylim(90, 100.5); ax.set_ylabel("5-fold CV accuracy (%)")
    ax.set_title("Model comparison")
    fig.tight_layout(); fig.savefig(RES / "model_comparison.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    stages_figure()
    sample_scenes()
    noisy_comparison()
    metric_figures()
    print("Figures written to", RES)
