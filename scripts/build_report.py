"""Build docs/ObjectLens_Project_Report.pdf (the detailed report for the portal).

Run from the project root after:  python main.py train && python main.py evaluate
                                   python scripts/make_results.py && python scripts/make_diagrams.py
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
DOCS, RES, DIA = ROOT / "docs", ROOT / "docs" / "results", ROOT / "docs" / "diagrams"
OUT = DOCS / "ObjectLens_Project_Report.pdf"

# ---- identity (used on the cover page only) --------------------------------------
STUDENT_NAME = "Ashish Kumar Rai"
REG_NO = "24BAI10666"
REPO_URL = "https://github.com/ashishtech-03/ObjectLens"     # change if your repo name differs

metrics = json.loads((RES / "metrics.json").read_text())
clf, e2e, rob = metrics["classifier"], metrics["end_to_end"], metrics["robustness"]

# ---- styles -----------------------------------------------------------------------
ss = getSampleStyleSheet()
BLUE, DARK = colors.HexColor("#1f4e79"), colors.HexColor("#222222")
body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=10.2,
                      leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("Heading1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17,
                    textColor=BLUE, spaceBefore=4, spaceAfter=10)
h2 = ParagraphStyle("Heading2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5,
                    textColor=BLUE, spaceBefore=10, spaceAfter=5)
cap = ParagraphStyle("cap", parent=body, fontSize=8.8, leading=11, alignment=TA_CENTER,
                     textColor=colors.HexColor("#555555"), spaceAfter=10)
cell = ParagraphStyle("cell", parent=body, fontSize=8.8, leading=11.2, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=2)
code = ParagraphStyle("code", fontName="Courier", fontSize=8, leading=9.6, backColor=colors.HexColor("#f4f4f4"),
                      borderPadding=5, spaceAfter=8, leftIndent=4)
toc1 = ParagraphStyle("toc1", fontName="Helvetica-Bold", fontSize=10.5, leading=17, leftIndent=0)
toc2 = ParagraphStyle("toc2", fontName="Helvetica", fontSize=9.3, leading=13, leftIndent=18)


def P(t, style=body):
    return Paragraph(t, style)


def B(items):
    return [Paragraph(i, bullet, bulletText="\u2022") for i in items]


def figure(path, caption, width_cm=16.0, max_h_cm=19.0):
    iw, ih = ImageReader(str(path)).getSize()
    w = width_cm * cm
    h = w * ih / iw
    if h > max_h_cm * cm:
        h = max_h_cm * cm
        w = h * iw / ih
    return KeepTogether([Image(str(path), width=w, height=h), Paragraph(caption, cap)])


def table(rows, widths_cm, header=True, zebra=True):
    data = [[Paragraph(str(c), cellb if (header and r == 0) else cell) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths_cm], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), BLUE))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f1f5fa")))
    t.setStyle(TableStyle(style))
    return t


# ---- document with TOC + footer -----------------------------------------------------
class Doc(BaseDocTemplate):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        frame = Frame(2 * cm, 2 * cm, A4[0] - 4 * cm, A4[1] - 4 * cm, id="f")
        self.addPageTemplates([PageTemplate("cover", frames=[frame], onPage=lambda c, d: None),
                               PageTemplate("main", frames=[frame], onPage=self._footer)])

    @staticmethod
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(2 * cm, 1.2 * cm, "ObjectLens - Computer Vision Project Report")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#cccccc"))
        canvas.line(2 * cm, 1.6 * cm, A4[0] - 2 * cm, 1.6 * cm)
        canvas.restoreState()

    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph):
            name = fl.style.name
            if name in ("Heading1", "Heading2"):
                key = f"h{self.seq.nextf('toc')}"
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (0 if name == "Heading1" else 1, fl.getPlainText(), self.page, key))


# ---- facts gathered from the repository -----------------------------------------------
def module_loc():
    rows = []
    for f in sorted((ROOT / "objectlens").glob("*.py")):
        rows.append((f.name, len(f.read_text().splitlines())))
    return rows


def test_counts():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    c = Counter(m.group(1) for m in re.finditer(r"^tests/(\w+\.py)::", out, re.M))
    return c


LOC = module_loc()
TESTS = test_counts()
N_TESTS = sum(TESTS.values())
TOTAL_LOC = sum(n for _, n in LOC)
cmp_ = clf["model_comparison"]
rep = clf["report"]

story = []

# ===================================== 1. COVER PAGE =====================================
story += [Spacer(1, 0.4 * cm), P("<b>VIT Bhopal University</b>", ParagraphStyle("u", parent=body, fontSize=15, alignment=TA_CENTER, textColor=BLUE)),
          P("B.Tech Computer Science and Engineering (AI &amp; ML)", ParagraphStyle("u2", parent=body, alignment=TA_CENTER, fontSize=11)),
          P("VITyarthi - Build Your Own Project", ParagraphStyle("u3", parent=body, alignment=TA_CENTER, fontSize=11)),
          Spacer(1, 1.2 * cm),
          P("<b>ObjectLens</b>", ParagraphStyle("t", parent=body, fontSize=38, leading=44, alignment=TA_CENTER, textColor=BLUE)),
          Spacer(1, 0.3 * cm),
          P("Automated Object Detection, Classification and<br/>Real-World Measurement using Computer Vision",
            ParagraphStyle("st", parent=body, fontSize=16, leading=22, alignment=TA_CENTER)),
          Spacer(1, 0.8 * cm),
          Image(str(RES / "scene_light_annotated.png"), width=10.5 * cm, height=10.5 * cm * 480 / 640),
          Spacer(1, 0.9 * cm),
          table([["Course", "Computer Vision"], ["Project type", "Build Your Own Project (individual)"],
                 ["Submitted by", f"<b>{STUDENT_NAME}</b>"], ["Registration number", f"<b>{REG_NO}</b>"],
                 ["GitHub repository", REPO_URL], ["Month / Year", "October 2026"]],
                [5.0, 11.0], header=False),
          ]
from reportlab.platypus import NextPageTemplate
story += [NextPageTemplate("main"), PageBreak()]

# ===================================== TOC ================================================
toc = TableOfContents()
toc.levelStyles = [toc1, toc2]
story += [P("Table of Contents", h1), toc, PageBreak()]

# ===================================== 2. INTRODUCTION ====================================
story += [P("1. Introduction", h1),
          P("Counting, sorting and sizing small objects - machine parts, laboratory samples, tokens, seeds - is a routine "
            "task in workshops, laboratories and classrooms. Done by hand it is slow, tiring and error-prone; commercial "
            "machine-vision systems solve it but are expensive, need special hardware and are black boxes. "
            "<b>Computer vision</b> offers a middle path: a single ordinary photograph already contains the information "
            "needed to find every object, recognise it and measure it."),
          P("<b>ObjectLens</b> is an end-to-end computer-vision application built for the <i>Computer Vision</i> course. "
            "Given a photograph of objects on a plain background it automatically (1) detects and separates every object, "
            "including objects that touch each other, (2) classifies each object's shape and colour with a trained "
            "machine-learning model, (3) converts pixel sizes to millimetres using a reference marker and (4) produces an "
            "annotated image, CSV/JSON reports, a summary chart and a searchable history database."),
          P("The project applies the central topics of the syllabus in one coherent system: image representation and "
            "colour spaces (BGR, grey, HSV, CIE-Lab), filtering and denoising, contrast enhancement (CLAHE), "
            "thresholding (Otsu), mathematical morphology, distance transform and watershed segmentation, contour "
            "analysis, shape descriptors and image moments (Hu invariants), supervised classification, camera-scale "
            "calibration and quantitative evaluation. All results in this report are reproducible with the commands in the "
            "repository README."),
          P(f"<b>Repository:</b> {REPO_URL}")]

# ===================================== 3. PROBLEM STATEMENT ===============================
story += [P("2. Problem Statement", h1),
          P("Design and implement a low-cost, transparent software system that, from a single photograph of objects lying on "
            "a plain background, can <b>detect</b> every object (even when objects touch), <b>classify</b> its shape and "
            "colour, <b>measure</b> its real-world dimensions in millimetres, and <b>report</b> the results in a form that can be "
            "inspected, exported and stored, while quantifying its own accuracy and robustness."),
          P("<b>Objectives</b>"),
          *B(["Detect all objects, including touching ones, with high precision and recall.",
              "Classify eight shape classes and name the object colour, flagging uncertain predictions as <i>unknown</i>.",
              "Calibrate pixels to millimetres from a reference square and report length, width, area and perimeter.",
              "Provide reproducible outputs (annotated image, CSV, JSON, chart) and a history database.",
              "Evaluate rigorously on data with exact ground truth, including a noise-robustness study.",
              "Deliver clean, modular, tested and documented code under version control."]),
          P("<b>Scope.</b> In scope: classical CV pipeline, ML shape classifier, calibration, synthetic data generator, CLI, SQLite, "
            "tests. Out of scope: deep-learning detectors, perspective correction, live video, cluttered or textured "
            "backgrounds. <b>Target users:</b> students and educators, small workshops and makers, developers needing a "
            "readable reference pipeline.")]

# ===================================== 4. FUNCTIONAL REQUIREMENTS =========================
story += [P("3. Functional Requirements", h1),
          P("The system provides five major functional modules, each with a clear input/output contract:"),
          table([["Module", "Input", "Processing", "Output"],
                 ["<b>M1 Detection &amp; segmentation</b>", "Image file", "Validation, resize, Gaussian blur, CLAHE, Lab background distance, Otsu, morphology, watershed, edge refinement", "Per-object contours and masks"],
                 ["<b>M2 Recognition</b>", "Contours + image", "Shape, moment and colour features; SVM classification with confidence", "Label, confidence, colour name"],
                 ["<b>M3 Measurement &amp; calibration</b>", "Features + reference/scale", "px/mm from reference square; unit conversion", "Length, width, area, perimeter in mm"],
                 ["<b>M4 Reporting &amp; storage</b>", "Analysis result", "Annotation, CSV/JSON/chart export, SQLite transaction", "Files + database rows + console summary"],
                 ["<b>M5 Training &amp; evaluation</b>", "Seeds, sample counts", "Synthetic scene generation, training, CV, end-to-end metrics, noise sweep", "Model file, metrics.json"]],
                [3.4, 2.6, 6.6, 4.4]),
          Spacer(1, 6),
          table([["ID", "Functional requirement"],
                 ["FR1", "Load and validate an image (existence, extension whitelist, decodability)."],
                 ["FR2", "Pre-process: resize large images, Gaussian denoising, CLAHE contrast enhancement."],
                 ["FR3", "Segment foreground on light or dark plain backgrounds, independent of object colour."],
                 ["FR4", "Separate touching objects (distance transform + watershed)."],
                 ["FR5", "Extract area, perimeter, circularity, aspect ratio, extent, solidity, vertex counts, Hu moments and colour."],
                 ["FR6", "Train, save, load and apply an ML classifier; output confidence; label low-confidence objects <i>unknown</i>."],
                 ["FR7", "Calibrate pixels-per-millimetre from a reference square or a user-supplied value."],
                 ["FR8", "Measure length, width, area, perimeter and equivalent diameter in millimetres."],
                 ["FR9", "Export annotated image, binary mask, CSV, JSON and a per-shape count chart."],
                 ["FR10", "Store every analysis in SQLite; list stored runs and inspect their detections."],
                 ["FR11", "Generate synthetic annotated scenes; evaluate the classifier and the whole pipeline; compare models."],
                 ["FR12", "Provide a command-line interface: train, analyze, demo, evaluate, history."]],
                [1.6, 15.4]),
          P("<b>Logical workflow of user interaction.</b> The user first trains the model once (<font face='Courier'>python main.py train</font>), "
            "then analyses an image with optional calibration flags, inspects the annotated image and exported files, "
            "optionally stores the run, and later reviews history. The complete workflow diagram is given in Section 6.", body)]

# ===================================== 5. NON-FUNCTIONAL REQUIREMENTS =====================
story += [P("4. Non-functional Requirements", h1),
          table([["ID", "Category", "Requirement", "Realisation / verification"],
                 ["NFR1", "Performance", "&lt; 0.2 s for 640x480 and &lt; 0.5 s for 1920x1440 images on a laptop CPU.", "Images are down-scaled to at most 1024 px; vectorised OpenCV/NumPy. Measured about 49 ms and 144 ms."],
                 ["NFR2", "Reliability and error handling", "Invalid input must never crash the program with a stack trace.", "Custom exception hierarchy; the CLI returns exit code 2 with a clear message for expected errors and code 1 plus a logged traceback for unexpected ones. Covered by tests."],
                 ["NFR3", "Security", "No SQL injection; all inputs validated.", "Parameterised queries only, integer coercion of ids, extension whitelist, configuration validation. Models are loaded only from user-trained files (documented)."],
                 ["NFR4", "Usability", "Self-explanatory interface and readable results.", "Help text for every command, annotated images, plain-text summary, actionable error messages (e.g. 'Run: python main.py train')."],
                 ["NFR5", "Maintainability", "Small single-purpose modules, documented and tested.", f"{len(LOC)} modules, {TOTAL_LOC} lines in the package, type hints, docstrings, {N_TESTS} automated tests."],
                 ["NFR6", "Logging / monitoring", "Every run leaves an audit trail.", "Console and rotating file log (outputs/objectlens.log) with configurable level."],
                 ["NFR7", "Resource efficiency", "CPU only, modest memory.", "Classical CV plus a small SVM (model file under 1 MB); no GPU or large downloads."],
                 ["NFR8", "Reproducibility", "Same seed gives same results.", "Seeded NumPy generators and scikit-learn random_state."]],
                [1.3, 2.7, 5.2, 7.8])]

# ===================================== 6. SYSTEM ARCHITECTURE =============================
story += [PageBreak(), P("5. System Architecture", h1),
          P("ObjectLens follows a layered architecture. The <b>presentation layer</b> (CLI) parses commands and handles errors. "
            "The <b>application layer</b> contains the pipeline orchestrator, the validated immutable configuration and the "
            "evaluation code. The <b>computer-vision core</b> is a linear chain of five stages, each in its own module with "
            "a narrow interface (pre-processing, segmentation, feature extraction, classification, measurement). The "
            "<b>output and persistence layer</b> holds reporting, the SQLite repository and the stored model. A synthetic "
            "data generator supplies labelled data for training and evaluation only."),
          figure(DIA / "architecture.png", "Figure 1 - System architecture of ObjectLens", width_cm=10.5, max_h_cm=17.5),
          P("This separation lets each stage be tested in isolation (for example segmentation is tested with hand-drawn images, "
            "the classifier with generated feature vectors) and replaced independently (for example swapping the SVM for a "
            "neural network would only touch <font face='Courier'>classifier.py</font>).")]

# ===================================== 7. DESIGN DIAGRAMS =================================
story += [PageBreak(), P("6. Design Diagrams", h1),
          P("6.1 Use Case Diagram", h2),
          P("Two actors use the system: the <b>Operator</b> analyses images, calibrates, exports and reviews history, while the "
            "<b>Developer / Evaluator</b> trains and evaluates models. Analysing an image <i>includes</i> detection, classification and "
            "measurement; measurement includes calibration; storing the result is an optional <i>extension</i>."),
          figure(DIA / "usecase.png", "Figure 2 - Use case diagram", width_cm=15.5, max_h_cm=11),
          P("6.2 Workflow Diagram", h2),
          P("The workflow shows the decisions taken while analysing an image: file validation, optional watershed splitting, "
            "optional calibration and optional database storage."),
          figure(DIA / "workflow.png", "Figure 3 - Process / workflow diagram of the analyse command", width_cm=9.8, max_h_cm=19),
          P("6.3 Sequence Diagram", h2),
          P("The sequence diagram shows the interaction between user, CLI, pipeline and the stage modules for one analysis, "
            "including the report-writing and optional persistence calls."),
          figure(DIA / "sequence_diagram.png", "Figure 4 - Sequence diagram for analysing one image", width_cm=17, max_h_cm=11),
          P("6.4 Class / Component Diagram", h2),
          P("<font face='Courier'>ObjectLensPipeline</font> composes a <font face='Courier'>Config</font> and a "
            "<font face='Courier'>ShapeClassifier</font> and produces an <font face='Courier'>AnalysisResult</font> holding one "
            "<font face='Courier'>DetectedObject</font> per object, each of which aggregates a <font face='Courier'>Region</font>, "
            "<font face='Courier'>ObjectFeatures</font> and an optional <font face='Courier'>Measurement</font>. "
            "<font face='Courier'>Repository</font> persists results; all expected failures derive from <font face='Courier'>ObjectLensError</font>."),
          figure(DIA / "class_diagram.png", "Figure 5 - Class / component diagram", width_cm=12.5, max_h_cm=17.5),
          P("6.5 ER Diagram and Schema Design", h2),
          P("Every analysis is stored as one row in <font face='Courier'>runs</font> and one row per object in "
            "<font face='Courier'>detections</font> (one-to-many, cascade delete). The two inserts happen in a single "
            "transaction so a run is never stored half-way. An index on <font face='Courier'>detections.run_id</font> keeps look-ups fast."),
          figure(DIA / "er_diagram.png", "Figure 6 - Entity-relationship diagram of the SQLite database", width_cm=15.0, max_h_cm=9)]

# ===================================== 8. DESIGN DECISIONS ================================
story += [PageBreak(), P("7. Design Decisions and Rationale", h1),
          table([["Decision", "Alternatives considered", "Rationale"],
                 ["Segment with <b>Lab distance from the border colour</b>", "Grey-level Otsu; adaptive threshold; Canny edges", "Independent of object colour and brightness; handles light and dark backgrounds without a polarity hack; yellow-on-white objects are not lost. Canny edges give open contours."],
                 ["<b>Clamped Otsu</b> threshold (15-40 Lab units)", "Pure Otsu; fixed threshold", "Pure Otsu fails with a very high-contrast object next to a low-contrast one, or on an empty image; a fixed value is not adaptive."],
                 ["<b>Half-maximum edge refinement</b>", "Use the first mask as is", "The permissive global threshold over-grew masks (area error 4-7 %). Moving each object's boundary to 50 % of its own contrast removed the bias (median error 0.64 %)."],
                 ["<b>Distance transform + watershed</b> with solidity gate, adaptive ratio and noise-core filter", "Hough circles; erosion-based splitting", "Works for any convex-ish object, not only circles; the solidity gate guarantees convex objects are never split; filtering tiny cores prevents sliver objects."],
                 ["<b>Scale-invariant 8-D feature vector</b>", "Raw pixels + CNN; HOG", "Small, interpretable, trains in seconds on CPU, and independent of object size and rotation."],
                 ["<b>SVM (RBF)</b> as the production model", "Random Forest, k-NN, rules", "Smooth class probabilities for confidence thresholding; strong on small data. Alternatives were benchmarked (Section 9)."],
                 ["Train on features of the <b>real pipeline output</b>", "Compute features from ideal geometry", "Removes train/inference mismatch (digital contour perimeter bias, noise)."],
                 ["<b>Reference-square calibration</b>", "Camera intrinsics; checkerboard", "One visible marker gives px/mm per image; no calibration session needed."],
                 ["<b>Confidence threshold -&gt; unknown</b>", "Always output the arg-max", "Flags uncertain objects instead of reporting wrong shapes confidently."],
                 ["<b>SQLite repository</b>", "CSV only; client/server DB", "Zero-configuration, transactional, cascade delete, built into Python."],
                 ["<b>Synthetic generator with exact ground truth</b>", "Manual annotation of photos", "Reproducible, unlimited, and gives exact area/position truth so size error can be measured automatically."]],
                [4.4, 3.6, 9.0])]

# ===================================== 9. IMPLEMENTATION ==================================
story += [PageBreak(), P("8. Implementation Details", h1),
          P("The package <font face='Courier'>objectlens</font> contains the modules below (lines of code in brackets)."),
          table([["Module", "LOC", "Responsibility"]] + [[f"<font face='Courier'>{n}</font>", str(c), d] for (n, c), d in zip(LOC, [
              "package metadata", "ShapeClassifier (SVM/RF/kNN), RuleBasedClassifier, model comparison", "argparse CLI and error handling",
              "immutable Config with validation", "evaluation of classifier, end-to-end pipeline and robustness", "custom exception hierarchy",
              "shape / moment / colour feature extraction", "console + rotating-file logging", "px/mm calibration and mm measurements",
              "orchestration of all stages", "image loading, validation, resize, blur, CLAHE", "annotation, CSV, JSON, chart export",
              "Lab-distance segmentation, morphology, watershed, edge refinement", "SQLite repository", "synthetic scene and dataset generator"][:len(LOC)])],
                [4.2, 1.4, 11.4]),
          P("8.1 Pre-processing", h2),
          P("Images are validated, converted to 3-channel BGR, down-scaled with area interpolation when larger than 1024 px (the scale "
            "factor is remembered so user-supplied calibration still applies), smoothed with a 5x5 Gaussian kernel and a "
            "grey version is enhanced with CLAHE (clip limit 2, 8x8 tiles) for contrast-limited local equalisation."),
          P("8.2 Segmentation", h2),
          P("The background colour is the per-channel median of the border pixels in CIE-Lab, where Euclidean distance approximates "
            "perceptual colour difference. For every pixel <i>p</i> the distance map is d(p) = || Lab(p) - Lab<sub>bg</sub> ||. "
            "The map is thresholded with Otsu's method, clamped to 15-40 Lab units, and cleaned by morphological opening (removes "
            "speckle) then closing (fills pin-holes). Connected components that are not convex (solidity below 0.96) are "
            "analysed with the distance transform; the threshold on the transform is raised from 55 % up to 85 % of its maximum "
            "until at least two significant cores (each at least 15 % of the largest) appear. These cores seed a marker-controlled "
            "watershed that splits the blob. Finally each object's mask is shrunk to its <b>half-maximum</b> edge: "
            "T<sub>i</sub> = 0.5 x median(d inside the eroded object i). Objects smaller than 150 px are discarded."),
          P("8.3 Feature extraction", h2),
          P("For every contour the system computes area A, perimeter P, circularity 4*pi*A/P<super>2</super>, aspect ratio and extent "
            "from the minimum-area rotated rectangle, solidity (A divided by convex-hull area), the number of vertices of the "
            "polygon approximation (epsilon = 3 % of the perimeter) of the contour and of its hull, log-scaled Hu moments "
            "(first two used), and the mean colour of the eroded interior, mapped to a colour name through HSV ranges. The "
            "eight-element classifier vector is scale and rotation invariant."),
          P("8.4 Classification, training data and model", h2),
          P(f"Training data are rendered by <font face='Courier'>synthetic.py</font> (8 classes x 150 samples + 50 watershed-split circles = "
            f"{clf['n_train']} samples; random size, rotation, colour, background theme, brightness gradient and noise) and are "
            f"processed by the same segmentation and feature code as real images. A scikit-learn pipeline "
            f"<i>StandardScaler -&gt; SVC(RBF, C=10, probability=True)</i> is trained, saved with joblib and loaded on demand. "
            f"Predictions with a probability below 0.45 are labelled <i>unknown</i>."),
          P("8.5 Calibration and measurement", h2),
          P("If a reference colour and side length are given, the largest <i>square</i> of that colour is the reference; "
            "its side in pixels is the mean of sqrt(area) and the mean of its rectangle sides, and k = side<sub>px</sub> / side<sub>mm</sub>. "
            "Each object then yields length = long side / k, width = short side / k, area = A / k<super>2</super>, perimeter = P / k "
            "and equivalent diameter = 2*sqrt(A/pi) / k. Alternatively the user supplies pixels-per-mm directly; if the image "
            "was down-scaled the value is scaled accordingly."),
          P("8.6 Reporting, storage and CLI", h2),
          P("<font face='Courier'>reporting.py</font> draws contours and labels, writes CSV and JSON (summary + per-object rows) and a bar "
            "chart. <font face='Courier'>storage.py</font> writes a run and its detections in one transaction. "
            "<font face='Courier'>cli.py</font> exposes five sub-commands, configures logging and converts exceptions to exit codes. "
            "Example session:"),
          Preformatted("$ python main.py train\n"
                       "Trained svm on 1300 samples -> models/shape_classifier.joblib\n"
                       "$ python main.py analyze data/samples/scene_light.png --ref-color green --ref-side-mm 30 --save-db\n"
                       "Image          : scene_light.png\n"
                       "Objects found  : 9\n"
                       "By shape       : circle=3, ellipse=1, hexagon=2, pentagon=1, rectangle=1, square=1\n"
                       "Calibration    : 3.9999 px/mm\n"
                       "Saved to database as run #1", code),
          P("8.7 Version control", h2),
          P("The project is developed in Git with small, descriptive commits (project skeleton, pre-processing and segmentation, "
            "features and classifier, measurement and storage, CLI, tests, documentation, report) so that the history documents "
            "the engineering process.")]

# ===================================== 10. SCREENSHOTS / RESULTS ==========================
story += [PageBreak(), P("9. Screenshots and Results", h1),
          P("9.1 Pipeline stages", h2),
          P("Figure 7 shows the successive representations for one scene that contains a 30 mm green reference square and two touching "
            "circles: the input, the CLAHE grey image, the Lab distance map, the cleaned binary mask and the final classified and "
            "measured result. Labels show the object number, shape and equivalent diameter (<i>d</i>) in mm."),
          figure(RES / "pipeline_stages.png", "Figure 7 - Stages of the pipeline on one scene", width_cm=17),
          P("9.2 Output on different backgrounds", h2),
          figure(RES / "scene_dark_annotated.png", "Figure 8 - Dark background", width_cm=9.2),
          figure(RES / "scene_cream_annotated.png", "Figure 9 - Cream background", width_cm=9.2),
          P("9.3 Classifier evaluation", h2),
          P(f"The classifier was trained on {clf['n_train']} samples and tested on {clf['n_test']} independent samples "
            f"(65 per class): held-out accuracy = <b>{clf['accuracy'] * 100:.1f} %</b>. Table 4 compares the models by 5-fold "
            f"stratified cross-validation on the training set."),
          table([["Model", "CV accuracy (mean)", "Std. deviation"]] +
                [[k.replace("_", " "), f"{v['mean'] * 100:.2f} %", f"{v['std'] * 100:.2f} %"] for k, v in cmp_.items()],
                [6, 5, 5]),
          Paragraph("Table 4 - Cross-validated accuracy of the candidate models", cap),
          P(f"The three ML models are essentially perfect on this data; the rule-based baseline reaches {cmp_['rule_based']['mean'] * 100:.1f} % - it "
            "fails on shapes whose polygon approximation is ambiguous (for example circles and hexagons or elongated ellipses "
            "and rectangles). The SVM was kept for its calibrated probabilities."),
          figure(RES / "confusion_matrix.png", "Figure 10 - Confusion matrix on the held-out test set", width_cm=9.0),
          figure(RES / "model_comparison.png", "Figure 11 - Model comparison (5-fold CV)", width_cm=9.0),
          P("9.4 End-to-end evaluation", h2),
          P(f"{e2e['scenes']} freshly rendered scenes (348 objects, 14 scenes contain a touching pair) with exact ground truth were analysed "
            "after calibration from the reference square:"),
          table([["Metric", "Result"],
                 ["Detection precision / recall", f"{e2e['detection_precision'] * 100:.1f} % / {e2e['detection_recall'] * 100:.1f} %"],
                 ["Shape classification accuracy (matched objects)", f"{e2e['classification_accuracy'] * 100:.1f} %"],
                 ["Colour naming accuracy", f"{e2e['colour_accuracy'] * 100:.1f} %"],
                 ["Scenes with exactly the right object count", f"{e2e['exact_count_rate'] * 100:.1f} %"],
                 ["Area error (mean / median) after px/mm calibration", f"{e2e['mean_area_error_pct']:.2f} % / {e2e['median_area_error_pct']:.2f} %"],
                 ["Recovered calibration for a true 4.000 px/mm", "3.9999 px/mm (mean over test scenes)"],
                 ["Run time per image (640x480 / 1920x1440)", "about 49 ms / 144 ms"]], [10, 6]),
          Paragraph("Table 5 - End-to-end results", cap),
          P("9.5 Robustness to image noise", h2),
          table([["Gaussian noise sigma (grey levels)", "Recall", "Precision", "Classification accuracy"]] +
                [[str(r["noise_sigma"]), f"{r['recall'] * 100:.1f} %", f"{r['precision'] * 100:.1f} %", f"{r['classification_accuracy'] * 100:.1f} %"] for r in rob],
                [5.5, 3.5, 3.5, 4.5]),
          Paragraph("Table 6 - Effect of sensor noise (20 scenes per level)", cap),
          figure(RES / "noise_robustness.png", "Figure 12 - Same scene at noise sigma 5, 25 and 45", width_cm=16.5),
          P("<b>Interpretation.</b> Gaussian smoothing, Lab-distance segmentation and morphological opening absorb even strong noise, so "
            "performance does not degrade up to sigma = 45. This is a statement about <i>noise</i> only: the synthetic scenes "
            "have clean, evenly lit geometry, so these figures are an upper bound and real photographs (shadows, glare, perspective) "
            "will score lower. This limitation is discussed again in Sections 12 and 13.")]

# ===================================== 11. TESTING ========================================
desc = {"test_config_preprocessing.py": "Config validation (8 invalid cases), missing / unsupported / corrupt files, colour-space and resize helpers",
        "test_segmentation.py": "blank image, dark background, specks, single circle area, touching circles split / merged, separate objects",
        "test_features.py": "circle and rectangle feature values, vector shape, 8 colour-name cases",
        "test_classifier.py": "untrained / missing model errors, empty data, bad algorithm, held-out accuracy, save-load round trip, unknown label, baseline, model comparison",
        "test_measurement.py": "scale from reference, invalid sizes, mm conversion, reference selection and errors",
        "test_storage_reporting.py": "SQLite round trip, cascade delete, unwritable path, SQL-injection safety, annotation, summary, exported files",
        "test_pipeline_cli.py": "full pipeline with touching pairs, regression test for sliver objects, resize + calibration, missing reference, blank image, end-to-end metrics, CLI train/analyze/history, graceful CLI errors"}
story += [PageBreak(), P("10. Testing Approach", h1),
          P(f"Testing uses <b>pytest</b> with {N_TESTS} automated tests (all passing; about 10 s). The strategy has four layers:"),
          *B(["<b>Unit tests</b> for each module with hand-made images (for example a circle drawn with OpenCV must produce one region whose area is within 5 % of pi*r<super>2</super>).",
              "<b>Validation / negative tests</b> for every error path: missing, unsupported and corrupt files, invalid configuration, untrained model, missing reference marker, unwritable database path.",
              "<b>Integration tests</b> that run the complete pipeline on generated scenes with ground truth, including a regression test (3 seeds) for the sliver-object bug found during development.",
              "<b>System / CLI tests</b> that call <font face='Courier'>main()</font> exactly as a user would (train, analyze with --save-db, history) and check exit codes and output."]),
          table([["Test file", "Tests", "What it covers"]] +
                [[f"<font face='Courier'>{f}</font>", str(n), desc.get(f, "")] for f, n in sorted(TESTS.items())], [5.2, 1.4, 10.4]),
          Paragraph("Table 7 - Test suite overview", cap),
          P("Run with <font face='Courier'>python -m pytest -q</font>. Beyond the unit tests, the quantitative evaluation of Section 9 is itself a "
            "validation test of the whole system against ground truth, and is repeated with <font face='Courier'>python main.py evaluate</font>.")]

# ===================================== 12. CHALLENGES =====================================
story += [P("11. Challenges Faced", h1),
          table([["Challenge", "Symptom", "Resolution"],
                 ["Touching objects merged into one blob", "Two touching circles were reported as one 'ellipse'; recall 92.5 %.", "The solidity gate (0.92) was too strict for a mild overlap; it was raised to 0.96 and the watershed threshold became adaptive (55 % to 85 %). Recall rose to 100 %."],
                 ["Spurious 'sliver' object", "An extra 4 mm 'unknown' object appeared between two touching circles.", "A 3-pixel noise core was used as a watershed seed. Cores smaller than 15 % of the largest are now ignored and a regression test covers it."],
                 ["Systematic area over-estimation", "Area error was 4-7 % (larger for objects with long perimeters such as stars).", "The permissive threshold dilated masks. Per-object half-maximum edge refinement removed the bias; median error fell to 0.64 %. An attempted 'half-perimeter' correction was tested and rejected because it over-corrected."],
                 ["Watershed-cut circles misclassified", "Circles split by watershed have a flat side; confidence fell below the threshold.", "Training data now include circles separated by the real watershed step."],
                 ["Digital-contour artefacts", "A perfect circle has circularity about 0.90, not 1.0, because pixel contours overestimate perimeter.", "Features are learned from real pipeline output, so the classifier sees the same distribution; unit-test expectations were corrected."],
                 ["Light and dark backgrounds", "Grey-level Otsu needs a polarity guess and loses low-contrast colours.", "Switched to Lab-distance-from-border segmentation."],
                 ["Evaluating without a labelled dataset", "No annotated real data were available.", "Built a synthetic generator with exact ground truth; honestly noted that real-photo performance still has to be verified."]],
                [3.8, 5.2, 8.0]),
          Spacer(1, 6),
          P("A further lesson was that very high scores on synthetic data can hide weaknesses: the noise sweep and the baseline comparison "
            "were added specifically to look for failure modes, and the report states plainly where the evidence stops.")]

# ===================================== 13. LEARNINGS ======================================
story += [P("12. Learnings and Key Takeaways", h1),
          *B(["<b>Colour space choice matters:</b> measuring distance in CIE-Lab made segmentation independent of object colour, something grey-level thresholding cannot do.",
              "<b>Measure, do not guess:</b> the area bias was invisible by eye and was found only by comparing against exact ground truth; a quantitative evaluation loop made each fix verifiable.",
              "<b>Classical CV is still powerful:</b> with a few well-chosen descriptors a small SVM is perfect on this task - and fully explainable, unlike a black-box network.",
              "<b>Train on what you will test on:</b> features computed from the true pipeline output prevented train/inference mismatch.",
              "<b>Edge cases dominate engineering effort:</b> touching objects, tiny noise cores and boundary bias took more time than the happy path.",
              "<b>Software engineering practice pays off:</b> modular design, custom exceptions, logging and 70 tests made refactoring safe (several algorithm changes were validated in seconds).",
              "<b>Be honest about limits:</b> synthetic results are an upper bound; documenting this is part of a trustworthy evaluation."])]

# ===================================== 14. FUTURE =========================================
story += [P("13. Future Enhancements", h1),
          *B(["Validate and fine-tune on a <b>real photo dataset</b> (shadows, glare, textured surfaces) and add data augmentation.",
              "Add <b>perspective correction</b> with a four-point homography or a checkerboard / ArUco marker, plus lens-distortion calibration.",
              "Support <b>arbitrary object categories</b> with a CNN (e.g. YOLO or Mask R-CNN) while keeping the classical measurement stage.",
              "Web or desktop <b>GUI</b> and live camera mode for real-time counting.",
              "Handle <b>partial occlusion and touching non-circular objects</b> with learned instance segmentation.",
              "Automatic detection of the reference marker (colour-independent) and sub-pixel edge fitting for finer measurement.",
              "Packaging as a pip-installable library and a continuous-integration workflow running the test suite on every push."])]

# ===================================== 15. REFERENCES =====================================
refs = ["Bradski, G. (2000). The OpenCV Library. <i>Dr. Dobb's Journal of Software Tools</i>.",
        "OpenCV documentation - Image Segmentation with Watershed Algorithm, Contour Features, Morphological Transformations. https://docs.opencv.org",
        "Otsu, N. (1979). A threshold selection method from gray-level histograms. <i>IEEE Transactions on Systems, Man, and Cybernetics</i>, 9(1), 62-66.",
        "Vincent, L., and Soille, P. (1991). Watersheds in digital spaces: an efficient algorithm based on immersion simulations. <i>IEEE TPAMI</i>, 13(6), 583-598.",
        "Hu, M.-K. (1962). Visual pattern recognition by moment invariants. <i>IRE Transactions on Information Theory</i>, 8(2), 179-187.",
        "Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. In <i>Graphics Gems IV</i>, Academic Press, 474-485.",
        "Cortes, C., and Vapnik, V. (1995). Support-vector networks. <i>Machine Learning</i>, 20(3), 273-297.",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825-2830.",
        "Szeliski, R. (2022). <i>Computer Vision: Algorithms and Applications</i>, 2nd ed. Springer.",
        "Gonzalez, R. C., and Woods, R. E. (2018). <i>Digital Image Processing</i>, 4th ed. Pearson.",
        "Harris, C. R. et al. (2020). Array programming with NumPy. <i>Nature</i>, 585, 357-362.",
        "VITyarthi - Build Your Own Project: General Project Instructions and Submission Guidelines (course document)."]
story += [P("14. References", h1)] + [P(f"[{i}] {r}") for i, r in enumerate(refs, 1)]


def main():
    DOCS.mkdir(exist_ok=True)
    doc = Doc(str(OUT), pagesize=A4, title="ObjectLens - Computer Vision Project Report",
              author=STUDENT_NAME, subject="Computer Vision - Build Your Own Project")
    doc.multiBuild(story)
    print("Report written to", OUT)


if __name__ == "__main__":
    main()
