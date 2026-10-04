# Spatial–Temporal Feature Fusion with Multi-Head Attention for Supervised Anomaly Localization

Custom YOLOv8n appearance + R3D-18 temporal architecture with convolutional fusion,
residual spatial attention and dense key-frame localization heads. This is the
manuscript implementation, not the official YOWOv3 baseline. The six classes are
Arrest, Assault, Burglary, Robbery, Stealing and Vandalism.

## What this release contains

Executable training, evaluation, dataset auditing, controlled component variants,
three-seed runs, 8/16/32-frame retraining, fixed-checkpoint threshold sweeps,
class-aware NMS sensitivity, false-positive attribution, video-cluster bootstrap,
shape traces, forward/offline profiling, charts, record-derived overlays, an
updated self-contained notebook and checksum/export utilities.

`reported/reference_values.json` preserves the author's confirmed manuscript
values. It is a comparison reference. Generated `runs/` outputs are produced only
by execution. This source release contains no historical checkpoints, prediction
records, training histories, GPU measurements or research dataset bytes, because
those records were not contained in the supplied notebook. Unit-test fixtures are
synthetic software checks and are not manuscript results.

## Installation

Use Python 3.10 (manuscript version 3.10.13) and a suitable NVIDIA driver. Create a
virtual environment, then install the manuscript CUDA build and pinned packages:

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The CUDA 12.1 command is from the official previous-version installation guide:
https://docs.pytorch.org/get-started/previous-versions/
Additional package pins provide a reproducible proposed environment; their exact
historical versions were not all specified in the manuscript. Every execution
records installed versions. Download access is needed for the pretrained YOLOv8n
and Torchvision R3D-18 weights on first use; preserve their cache and provenance.

## Add `ucfcrime2local`

Extract the dataset into `ucfcrime2local/` beside the repository, or use an external
absolute path. Required layout:

```text
ucfcrime2local/
  frames/<video_id>/00000.jpg
  frames/<video_id>/00001.jpg
  annotations/train.csv
  annotations/val.csv
  annotations/test.csv
```

Read `DATASET.md` before adapting native records. Original annotations are not
assumed to already satisfy this CSV contract. Original video splits, frame-index
conventions, box coordinate conventions, and negative-frame verification must
come from actual records. Do not infer negative examples from missing boxes.
Supply the exact eligible-key-frame manifests used for the reported split.
The loader does not automatically turn 31,687 listed paths into 20,000 training
key frames or choose a historical validation partition.

```bash
python configure.py --data-root /absolute/path/ucfcrime2local --output configs/local.yaml
python audit_dataset.py --config configs/local.yaml --output artifacts/dataset_audit.json
```

CUDA is required by default. For an explicit CPU smoke test use `--device cpu`
with `configure.py`; CPU timing does not represent manuscript GPU performance.
All commands are executed from the repository root.

## Reference run

```bash
python train.py --config configs/local.yaml --out_dir runs/complete_T16_seed42
python evaluate.py --config configs/local.yaml --checkpoint runs/complete_T16_seed42/best.pt --output runs/complete_T16_seed42/evaluation
python analysis.py --predictions runs/complete_T16_seed42/evaluation/predictions.json --output runs/complete_T16_seed42/analysis --bootstrap 1000
python profile.py --config configs/local.yaml --checkpoint runs/complete_T16_seed42/best.pt --output runs/complete_T16_seed42/profiling
python plot_results.py --runs runs --output runs/figures
python export_overlays.py --predictions runs/complete_T16_seed42/evaluation/predictions.json --frames-root /absolute/path/ucfcrime2local/frames --output runs/figures/Figure8.PNG
python check_reported_values.py --results runs/complete_T16_seed42/evaluation/results.json
```

Use the workflow driver below when raw console logs are needed; it captures every
subprocess automatically. The standalone commands print to the terminal.

## Full manuscript experiment matrix

```bash
python run_experiments.py --data-root /absolute/path/ucfcrime2local --all
```

This performs eight 30-epoch training runs: the reference (seed 42), seeds 21/84,
spatial-only, temporal-only, fusion without attention, and independently retrained
8/32-frame complete models. Each has its own config, best/last checkpoints,
history, logs, environment, predictions and results. The reference additionally
receives threshold/NMS/error/bootstrap analyses and profiling. No significance
claim is inferred from component differences. Seeds do not make CUDA kernels
universally bitwise deterministic; the software/hardware environment matters.

Run directories are protected against accidental training overwrite. To evaluate
existing populated directories use `--skip-training`. This still evaluates and
rewrites derived summaries; preserve historical files before rerunning analysis.

## Protocol

- 16 causal RGB frames ending at the annotated key frame; repeat frame zero near
  the beginning; resize to 224×224; shared `(RGB/255 - 0.45)/0.225` normalization.
- YOLOv8n feature routing; R3D-18 convolutional stages; 256-channel fusion; four
  attention heads of dimension 64; dense class/objectness/normalized box outputs.
- AdamW, batch eight, accumulation two, 30 epochs; learning rates 1e-4, 5e-5,
  2.5e-5 and 1.25e-5 over epochs 1–10, 11–18, 19–24 and 25–30.
- Minimum mean validation mini-batch loss selects `best.pt`; test data never
  select the checkpoint. The checkpoint `epoch` field is one-based in this release.
- P/R/F1 use confidence 0.3; AP uses floor 0.001, all-point interpolation and a
  fixed six-class denominator (absent classes get zero AP).
- Highest-IoU same-class greedy matching at IoU 0.5. If that target is occupied,
  reject the prediction without reassignment. Primary evaluation has no NMS.
- One regression target per cell; later boxes overwrite coordinates on collisions.
  Multiple class labels may remain active at that cell, matching the supplied code.
- Data-loader workers are two, matching the supplied notebook's override.

## Timing boundaries

Forward-only timing: FP32 batch one, resident synthetic device tensors, ten
warm-up iterations, 100 synchronized individual measurements; one predicted key
frame per clip. THOP reports accounted operations with two FLOPs per MAC; custom
attention matrix operations are incompletely counted.

Offline timing in this release reads/decompresses pre-extracted JPEG frames,
assembles clips, preprocesses them, transfers inputs, performs a forward pass,
and decodes/serializes predictions in memory. It does **not** measure decoding
original compressed-video containers, writing outputs to disk, camera acquisition,
frame waiting, networking, display, queueing or asynchronous overlap. Do not
rename this JPEG-frame measurement as video-container decoding. Training peak
allocated memory uses real clips, batch eight and two accumulation batches, with
an AdamW update. Allocated memory and reserved memory are different quantities.

## Export and deposit

```bash
python export_release.py --repository . --runs runs --output release
```

Upload source files/notebook to GitHub. Deposit source and actual experimental
records (including real checkpoints and predictions) on Zenodo. Upload the dataset
separately with its own original attribution and applicable redistribution terms.
`RELEASE_CHECKLIST.md` documents final metadata and evidence checks. Dataset rights
are separate from code licensing; this release does not assign a new license to
UCF-Crime2Local. The repository does not publish itself or create a DOI.

## Citation and attribution

See `CITATION.cff`, `THIRD_PARTY_NOTICES.md` and the manuscript. Fill in real
repository and DOI identifiers after the deposit is created. No DOI is invented.
