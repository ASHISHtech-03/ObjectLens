"""Render every design diagram used in docs/ and the report.

Graphviz ('dot') draws architecture / workflow / use-case / class / ER diagrams;
the sequence diagram is drawn with matplotlib.  Run:  python scripts/make_diagrams.py
"""
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = 'graph [fontname="Helvetica", pad=0.3, dpi=170]; node [fontname="Helvetica", fontsize=11]; edge [fontname="Helvetica", fontsize=9];'


def dot(name: str, body: str):
    src = OUT / f"{name}.dot"
    src.write_text(body, encoding="utf-8")
    subprocess.run(["dot", "-Tpng", str(src), "-o", str(OUT / f"{name}.png")], check=True)
    src.unlink()


def architecture():
    dot("architecture", f'''digraph G {{ rankdir=TB; {STYLE} compound=true; nodesep=0.35; ranksep=0.5;
  user [label="User / Operator", shape=box, style="rounded,filled", fillcolor="#fff2cc"];
  subgraph cluster_pres {{ label="Presentation layer"; style="rounded,filled"; fillcolor="#eef4fb";
    cli [label="CLI  (cli.py / main.py)\\ntrain | analyze | demo | evaluate | history", shape=box, style=filled, fillcolor=white]; }}
  subgraph cluster_app {{ label="Application layer"; style="rounded,filled"; fillcolor="#eef8ee";
    pipe [label="ObjectLensPipeline  (pipeline.py)\\norchestrates all stages", shape=box, style=filled, fillcolor=white];
    cfg [label="Config + validation\\n(config.py)", shape=box, style=filled, fillcolor=white];
    eval [label="Evaluation\\n(evaluation.py)", shape=box, style=filled, fillcolor=white]; }}
  subgraph cluster_cv {{ label="Computer-vision core"; style="rounded,filled"; fillcolor="#fdf1e7";
    pre [label="1 Pre-processing\\nresize, Gaussian blur, CLAHE", shape=box, style=filled, fillcolor=white];
    seg [label="2 Segmentation\\nLab distance, Otsu, morphology,\\nwatershed, half-max refine", shape=box, style=filled, fillcolor=white];
    feat [label="3 Feature extraction\\ncontours, moments, colour (HSV)", shape=box, style=filled, fillcolor=white];
    clf [label="4 Classification\\nSVM / RF / kNN + rule baseline", shape=box, style=filled, fillcolor=white];
    meas [label="5 Calibration & measurement\\npixels -> millimetres", shape=box, style=filled, fillcolor=white];
    pre -> seg -> feat -> clf -> meas; }}
  subgraph cluster_out {{ label="Output & persistence"; style="rounded,filled"; fillcolor="#f5ecf8";
    rep [label="6 Reporting\\nannotated image, CSV, JSON, chart", shape=box, style=filled, fillcolor=white];
    db [label="Repository (storage.py)\\nSQLite: runs, detections", shape=cylinder, style=filled, fillcolor=white];
    model [label="Model store\\n(joblib file)", shape=cylinder, style=filled, fillcolor=white]; }}
  syn [label="Synthetic data generator\\n(synthetic.py) - scenes + ground truth", shape=box, style="filled", fillcolor="#e8e8e8"];
  log [label="Logging (logging_setup.py)\\nconsole + rotating file", shape=note, style=filled, fillcolor="#e8e8e8"];
  user -> cli; cli -> pipe; cfg -> pipe; pipe -> pre [lhead=cluster_cv]; meas -> rep [ltail=cluster_cv];
  rep -> db [label="save_analysis"]; syn -> clf [label="training data", style=dashed]; clf -> model [label="save/load", dir=both];
  eval -> pipe [style=dashed]; syn -> eval [style=dashed]; cli -> log [style=dotted, arrowhead=none];
}}''')


def workflow():
    dot("workflow", f'''digraph G {{ rankdir=TB; {STYLE} nodesep=0.4; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#eef4fb"];
  s [label="Start", shape=oval, fillcolor="#d9ead3"]; e [label="End", shape=oval, fillcolor="#f4cccc"];
  a [label="User runs `analyze <image>`\\n(optional calibration flags)"];
  b [label="Load image"]; c [shape=diamond, style=filled, fillcolor="#fff2cc", label="Valid file &\\nformat?"];
  err [label="Raise ImageLoadError\\nlog + friendly message", fillcolor="#f4cccc"];
  d [label="Pre-process\\n(resize, blur, CLAHE)"]; f [label="Segment foreground\\n(Lab distance + Otsu + morphology)"];
  g [shape=diamond, style=filled, fillcolor="#fff2cc", label="Touching\\nobjects?"]; h [label="Split with distance\\ntransform + watershed"];
  i [label="Refine edges (half-maximum)\\nextract contours, drop specks"]; j [label="Extract features per object"];
  k [label="Classify shape (SVM)\\nconfidence < threshold -> unknown"];
  l [shape=diamond, style=filled, fillcolor="#fff2cc", label="Calibration\\ngiven?"];
  m [label="Compute px/mm from reference\\nsquare or user value"]; n [label="Measure size in mm"];
  o [label="Annotate image, write CSV / JSON / chart"]; p [shape=diamond, style=filled, fillcolor="#fff2cc", label="--save-db ?"];
  q [label="Store run + detections in SQLite"]; r [label="Print summary"];
  s -> a -> b -> c; c -> err [label="no"]; err -> e; c -> d [label="yes"]; d -> f -> g;
  g -> h [label="yes"]; g -> i [label="no"]; h -> i; i -> j -> k -> l;
  l -> m [label="yes"]; l -> o [label="no (pixels only)"]; m -> n -> o; o -> p; p -> q [label="yes"]; p -> r [label="no"]; q -> r -> e;
}}''')


def usecase():
    dot("usecase", f'''digraph G {{ rankdir=LR; {STYLE} nodesep=0.3; ranksep=1.1;
  node [shape=box, style="filled", fillcolor="#fff2cc"];
  user [label="<<actor>>\\nOperator / Student", shape=box, style="rounded,filled"];
  dev [label="<<actor>>\\nDeveloper / Evaluator", shape=box, style="rounded,filled", fillcolor="#d9ead3"];
  subgraph cluster_sys {{ label="ObjectLens system"; style="rounded"; node [shape=ellipse, style=filled, fillcolor="#eef4fb"];
    uc1 [label="Analyse image"]; uc2 [label="Detect & count objects"]; uc3 [label="Classify shapes & colours"];
    uc4 [label="Calibrate scale (px/mm)"]; uc5 [label="Measure size in mm"]; uc6 [label="Export annotated image,\\nCSV, JSON, chart"];
    uc7 [label="Store result in database"]; uc8 [label="View analysis history"]; uc9 [label="Generate demo scene"];
    uc10 [label="Train classifier"]; uc11 [label="Evaluate system"]; uc12 [label="Compare ML models"]; }}
  user -> uc1; user -> uc6; user -> uc8; user -> uc9; user -> uc4;
  dev -> uc10; dev -> uc11; dev -> uc12;
  uc1 -> uc2 [style=dashed, label="<<include>>"]; uc1 -> uc3 [style=dashed, label="<<include>>"];
  uc1 -> uc5 [style=dashed, label="<<include>>"]; uc5 -> uc4 [style=dashed, label="<<include>>"];
  uc7 -> uc1 [style=dashed, label="<<extend>>"]; uc11 -> uc12 [style=dashed, label="<<include>>"];
}}''')


def classes():
    dot("class_diagram", f'''digraph G {{ rankdir=BT; {STYLE} nodesep=0.35; ranksep=0.6;
  node [shape=record, style=filled, fillcolor="#fdfdf5", fontsize=10]; edge [arrowhead=open];
  Config [label="{{Config|+ max_dimension: int\\l+ blur_kernel: int\\l+ min_area_px: int\\l+ watershed_ratio: float\\l+ min_confidence: float\\l+ pixels_per_mm: float?\\l+ ref_color, ref_side_mm\\l|+ validate(): Config\\l}}"];
  Pipeline [label="{{ObjectLensPipeline|- config: Config\\l- _classifier: ShapeClassifier\\l|+ analyze_image(img, name): AnalysisResult\\l+ analyze_file(path): AnalysisResult\\l}}"];
  Result [label="{{AnalysisResult|+ image_name: str\\l+ objects: List[DetectedObject]\\l+ pixels_per_mm: float?\\l+ mask: ndarray\\l|+ count: int\\l}}"];
  Det [label="{{DetectedObject|+ id: int\\l+ label: str\\l+ confidence: float\\l+ is_reference: bool\\l|}}"];
  Reg [label="{{Region|+ contour: ndarray\\l+ mask: ndarray\\l+ area: float\\l|+ bbox\\l}}"];
  Feat [label="{{ObjectFeatures|+ area, perimeter\\l+ circularity, aspect_ratio\\l+ extent, solidity\\l+ vertices, hu\\l+ color_name\\l|+ vector(): ndarray\\l}}"];
  Meas [label="{{Measurement|+ length_mm, width_mm\\l+ area_mm2, perimeter_mm\\l+ equivalent_diameter_mm\\l|}}"];
  Clf [label="{{ShapeClassifier|- pipeline: sklearn Pipeline\\l+ algorithm: str\\l|+ fit(X, y)\\l+ predict(features, min_conf)\\l+ evaluate(X, y): dict\\l+ save(path) / load(path)\\l}}"];
  Rule [label="{{RuleBasedClassifier||+ predict_one(f): str\\l+ predict_vectors(X)\\l}}"];
  Repo [label="{{Repository|- db_path: str\\l|+ save_analysis(result, model)\\l+ list_runs(limit)\\l+ get_detections(run_id)\\l+ delete_run(run_id)\\l}}"];
  Err [label="{{ObjectLensError|ImageLoadError\\lConfigError\\lModelNotTrainedError\\lCalibrationError\\lStorageError\\l|}}", fillcolor="#fde9e7"];
  Mods [label="{{«modules»|preprocessing.preprocess()\\lsegmentation.segment()\\lfeatures.extract_features()\\lmeasurement.measure()\\lreporting.annotate() / save_outputs()\\lsynthetic.render_scene()\\l|}}", fillcolor="#eef4fb"];
  Pipeline -> Config [label="uses", arrowhead=vee]; Pipeline -> Clf [label="uses", arrowhead=vee]; Pipeline -> Mods [label="calls", style=dashed];
  Pipeline -> Result [label="creates", style=dashed]; Result -> Det [label="1..*", arrowhead=diamond, dir=back]; 
  Det -> Reg [label="1", arrowhead=vee]; Det -> Feat [label="1", arrowhead=vee]; Det -> Meas [label="0..1", arrowhead=vee];
  Repo -> Result [label="persists", style=dashed]; Rule -> Feat [label="reads", style=dashed]; Clf -> Feat [label="reads", style=dashed];
  Err -> Pipeline [label="raised by", style=dotted, arrowhead=none];
}}''')


def er():
    dot("er_diagram", f'''digraph G {{ rankdir=LR; {STYLE} nodesep=0.8;
  node [shape=plaintext];
  runs [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#fdfdf5">
    <tr><td colspan="2" bgcolor="#cfe2f3"><b>runs</b></td></tr>
    <tr><td align="left"><b>PK</b> id</td><td align="left">INTEGER</td></tr>
    <tr><td align="left">created_at</td><td align="left">TEXT</td></tr>
    <tr><td align="left">image_name</td><td align="left">TEXT</td></tr>
    <tr><td align="left">model</td><td align="left">TEXT</td></tr>
    <tr><td align="left">object_count</td><td align="left">INTEGER</td></tr>
    <tr><td align="left">pixels_per_mm</td><td align="left">REAL (nullable)</td></tr></table>>];
  det [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#fdfdf5">
    <tr><td colspan="2" bgcolor="#d9ead3"><b>detections</b></td></tr>
    <tr><td align="left"><b>PK</b> id</td><td align="left">INTEGER</td></tr>
    <tr><td align="left"><b>FK</b> run_id</td><td align="left">INTEGER -&gt; runs.id</td></tr>
    <tr><td align="left">obj_index</td><td align="left">INTEGER</td></tr>
    <tr><td align="left">label</td><td align="left">TEXT</td></tr>
    <tr><td align="left">confidence</td><td align="left">REAL</td></tr>
    <tr><td align="left">color</td><td align="left">TEXT</td></tr>
    <tr><td align="left">area_px, perimeter_px</td><td align="left">REAL</td></tr>
    <tr><td align="left">centroid_x, centroid_y</td><td align="left">REAL</td></tr>
    <tr><td align="left">length_mm, width_mm, area_mm2</td><td align="left">REAL (nullable)</td></tr>
    <tr><td align="left">is_reference</td><td align="left">INTEGER (0/1)</td></tr></table>>];
  runs -> det [label="  1  :  N  (ON DELETE CASCADE)", arrowhead=crow, arrowtail=tee, dir=both, fontsize=10];
}}''')


def sequence():
    actors = ["User", "CLI", "Pipeline", "Preprocess", "Segment", "Features", "Classifier", "Measure", "Reporting", "Repository"]
    msgs = [
        (0, 1, "analyze image.png --ref-color green ...", False),
        (1, 2, "ObjectLensPipeline(config).analyze_file()", False),
        (2, 3, "load_image() / preprocess()", False), (3, 2, "PreprocessResult", True),
        (2, 4, "segment(pre, cfg)", False), (4, 2, "regions (watershed split)", True),
        (2, 5, "extract_all(regions)", False), (5, 2, "ObjectFeatures[]", True),
        (2, 6, "predict(features)", False), (6, 2, "labels + confidence", True),
        (2, 7, "find_reference() / measure()", False), (7, 2, "Measurement[] (mm)", True),
        (2, 1, "AnalysisResult", True),
        (1, 8, "save_outputs(result)", False), (8, 1, "PNG / CSV / JSON paths", True),
        (1, 9, "save_analysis(result)  [--save-db]", False), (9, 1, "run_id", True),
        (1, 0, "print summary", True),
    ]
    fig, ax = plt.subplots(figsize=(15, 9))
    n = len(actors)
    for i, name in enumerate(actors):
        ax.add_patch(plt.Rectangle((i - 0.42, n * 0 + len(msgs) + 0.6), 0.84, 0.8, fc="#cfe2f3", ec="black"))
        ax.text(i, len(msgs) + 1.0, name, ha="center", va="center", fontsize=10, weight="bold")
        ax.plot([i, i], [0, len(msgs) + 0.6], ls="--", color="gray", lw=0.8)
    for k, (a, b, text, ret) in enumerate(msgs):
        y = len(msgs) - k
        ax.annotate("", xy=(b, y), xytext=(a, y),
                    arrowprops=dict(arrowstyle="->", ls="--" if ret else "-", lw=1.2,
                                    color="#555555" if ret else "black"))
        ax.text((a + b) / 2, y + 0.12, text, ha="center", va="bottom", fontsize=8.5)
    ax.set_xlim(-0.7, n - 0.3); ax.set_ylim(-0.3, len(msgs) + 1.8); ax.axis("off")
    ax.set_title("Sequence diagram - analysing one image", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "sequence_diagram.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    architecture(); workflow(); usecase(); classes(); er(); sequence()
    print("diagrams written to", OUT)
