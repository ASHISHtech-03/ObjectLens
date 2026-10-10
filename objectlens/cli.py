"""Command-line interface for ObjectLens."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2

from . import __version__
from .classifier import ALGORITHMS, ShapeClassifier
from .config import COLOR_NAMES, Config
from .evaluation import evaluate_classifier, evaluate_robustness, evaluate_scenes
from .exceptions import ObjectLensError
from .logging_setup import setup_logging
from .pipeline import ObjectLensPipeline
from .reporting import format_summary, save_outputs
from .storage import Repository
from .synthetic import build_feature_dataset, render_scene

log = logging.getLogger("objectlens.cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="objectlens",
                                description="Detect, classify and measure objects in images.")
    p.add_argument("--version", action="version", version=f"ObjectLens {__version__}")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", default="outputs/objectlens.log")
    p.add_argument("--model", default=Config().model_path, help="path of the model file")
    p.add_argument("--db", default=Config().db_path, help="path of the SQLite database")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train the shape classifier on synthetic data")
    t.add_argument("--samples", type=int, default=150, help="training samples per class")
    t.add_argument("--algorithm", choices=ALGORITHMS, default="svm")
    t.add_argument("--seed", type=int, default=42)

    e = sub.add_parser("evaluate", help="evaluate classifier and full pipeline")
    e.add_argument("--scenes", type=int, default=40)
    e.add_argument("--output", default="docs/results/metrics.json")

    a = sub.add_parser("analyze", help="analyse an image")
    a.add_argument("image")
    a.add_argument("--out", default="outputs")
    a.add_argument("--pixels-per-mm", type=float)
    a.add_argument("--ref-color", choices=COLOR_NAMES)
    a.add_argument("--ref-side-mm", type=float)
    a.add_argument("--save-db", action="store_true", help="store the result in SQLite")
    a.add_argument("--no-watershed", action="store_true")

    d = sub.add_parser("demo", help="generate a synthetic scene and analyse it")
    d.add_argument("--seed", type=int, default=3)
    d.add_argument("--out", default="outputs")
    d.add_argument("--save-db", action="store_true")

    h = sub.add_parser("history", help="show stored analyses")
    h.add_argument("--limit", type=int, default=10)
    h.add_argument("--run", type=int, help="show detections of one run id")
    return p


def _config_from(args, **extra) -> Config:
    return Config(model_path=args.model, db_path=args.db, **extra).validate()


def cmd_train(args) -> int:
    cfg = _config_from(args)
    log.info("Generating %d samples per class ...", args.samples)
    X, y = build_feature_dataset(args.samples, args.seed, cfg)
    clf = ShapeClassifier(args.algorithm, args.seed).fit(X, y)
    clf.save(cfg.model_path)
    print(f"Trained {args.algorithm} on {len(X)} samples -> {cfg.model_path}")
    return 0


def cmd_evaluate(args) -> int:
    cfg = _config_from(args)
    clf_metrics = evaluate_classifier(cfg=cfg)
    pipeline = ObjectLensPipeline(cfg)
    scene_metrics = evaluate_scenes(pipeline, args.scenes)
    robustness = evaluate_robustness(pipeline)
    payload = {"classifier": clf_metrics, "end_to_end": scene_metrics,
               "robustness": robustness}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Classifier accuracy (held-out): {clf_metrics['accuracy']:.3f}")
    for k, v in scene_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    for row in robustness:
        print(f"  noise sigma={row['noise_sigma']:>3}: recall={row['recall']:.3f} "
              f"precision={row['precision']:.3f} accuracy={row['classification_accuracy']:.3f}")
    print(f"Metrics written to {args.output}")
    return 0


def _run_analysis(pipeline: ObjectLensPipeline, result_fn, out_dir: str, stem: str,
                  save_db: bool, db_path: str) -> int:
    result = result_fn()
    paths = save_outputs(result, out_dir, stem)
    print(format_summary(result))
    print("Outputs:", ", ".join(str(p) for p in paths.values()))
    if save_db:
        run_id = Repository(db_path).save_analysis(result, pipeline.classifier.algorithm)
        print(f"Saved to database as run #{run_id}")
    return 0


def cmd_analyze(args) -> int:
    cfg = _config_from(args, pixels_per_mm=args.pixels_per_mm, ref_color=args.ref_color,
                       ref_side_mm=args.ref_side_mm, watershed=not args.no_watershed)
    pipeline = ObjectLensPipeline(cfg)
    stem = Path(args.image).stem
    return _run_analysis(pipeline, lambda: pipeline.analyze_file(args.image), args.out, stem,
                         args.save_db, cfg.db_path)


def cmd_demo(args) -> int:
    img, truths = render_scene(args.seed, n_objects=7, touching_pairs=1, reference=(30.0, 4.0))
    sample = Path("data/samples") / f"demo_scene_{args.seed}.png"
    sample.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sample), img)
    print(f"Generated {sample} with {len(truths)} ground-truth objects")
    cfg = _config_from(args, ref_color="green", ref_side_mm=30.0)
    pipeline = ObjectLensPipeline(cfg)
    return _run_analysis(pipeline, lambda: pipeline.analyze_image(img, sample.name), args.out,
                         sample.stem, args.save_db, cfg.db_path)


def cmd_history(args) -> int:
    repo = Repository(args.db)
    if args.run is not None:
        rows = repo.get_detections(args.run)
        if not rows:
            print(f"No detections for run {args.run}")
            return 1
        for r in rows:
            print(f"#{r['obj_index']:<3} {r['label']:<10} {r['color']:<8} "
                  f"conf={r['confidence']:.2f} area_px={r['area_px']:.0f} "
                  f"area_mm2={r['area_mm2'] if r['area_mm2'] is not None else '-'}")
        return 0
    runs = repo.list_runs(args.limit)
    if not runs:
        print("No stored analyses yet.")
    for r in runs:
        print(f"run #{r['id']:<4} {r['created_at']}  {r['image_name']:<28} "
              f"objects={r['object_count']} model={r['model']}")
    return 0


COMMANDS = {"train": cmd_train, "evaluate": cmd_evaluate, "analyze": cmd_analyze,
            "demo": cmd_demo, "history": cmd_history}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, args.log_file)
    try:
        return COMMANDS[args.command](args)
    except ObjectLensError as exc:
        log.error("%s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception:  # noqa: BLE001 - last-resort handler, full trace goes to the log
        log.exception("Unexpected failure")
        print("Unexpected error - see outputs/objectlens.log for details", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
