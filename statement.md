# Project Statement - ObjectLens

## Problem statement
Small workshops, laboratories, classrooms and hobby manufacturers frequently need to **count, sort and size small objects** (components, parts, tokens, samples) laid out on a flat surface. Doing this by hand is slow, tiring and error-prone, and commercial machine-vision systems are expensive, need special hardware and are usually closed boxes whose results cannot be inspected.

**ObjectLens** is a transparent, low-cost computer-vision tool that takes an ordinary photograph of objects on a plain background and automatically
1. **detects** every object (including objects that touch each other),
2. **classifies** its shape (circle, ellipse, triangle, square, rectangle, pentagon, hexagon, star) and dominant colour,
3. **measures** its real-world size in millimetres using a reference marker or a known scale, and
4. **reports** the results as an annotated image, CSV/JSON files, a summary chart and a searchable history database.

## Scope of the project
**In scope**
- Classical computer-vision pipeline: pre-processing, colour-space based segmentation, morphology, distance-transform + watershed separation, contour analysis, moment/shape/colour features.
- Supervised machine learning (SVM, Random Forest, k-NN) for shape classification, compared against a hand-written rule-based baseline.
- Pixel-to-millimetre calibration from a reference square of known size (or a user-supplied scale).
- A synthetic scene generator with exact ground truth, used for training, testing and quantitative evaluation.
- Command-line interface, SQLite persistence, logging, validation, automated tests.

**Out of scope (see Future Enhancements in the report)**
- Deep-learning detectors (YOLO / Mask R-CNN) and arbitrary 3-D objects.
- Perspective / lens-distortion correction, live camera streaming, GUI / web front-end.
- Textured or cluttered backgrounds (a plain, roughly uniform background is assumed).

## Target users
- Students and educators learning applied computer vision.
- Small workshops / makers / labs who need quick counting and sizing of parts.
- Developers who want a readable reference pipeline they can extend.

## High-level features
| # | Functional module | What it does |
|---|-------------------|--------------|
| 1 | **Detection & segmentation** | Loads/validates an image, enhances it, separates objects from the background and splits touching objects. |
| 2 | **Recognition** | Extracts shape + colour features and classifies each object with a trained ML model (confidence-aware, "unknown" for low confidence). |
| 3 | **Measurement & calibration** | Converts pixels to millimetres via a reference square and reports length, width, area and perimeter. |
| 4 | **Reporting & storage** | Annotated image, CSV, JSON, chart and an SQLite history of every analysis. |
| 5 | **Training & evaluation** | Synthetic data generation, model training/comparison, end-to-end metrics and noise-robustness tests. |
