# 3D Fracture Captioning with Tangent-Plane Curvature Disks and Vision-Language Models

Class-blind captioning of fractured 3D objects directly from surface geometry. Principal curvature is estimated at farthest-point-sampled locations, encoded as tangent-plane curvature disks, compressed by an MAE-style GeometryProjector into a geometry prefix, and injected into Qwen2.5-VL-7B-Instruct (LoRA). The object class is withheld from the prompt so captions must be grounded in geometry.

Paper: *3D Fracture Captioning with Tangent-Plane Curvature Disks and Vision-Language Models* (manuscript in preparation).

 
## Overview
 
**Method pipeline**
 
![Method pipeline](img/Fig2.png)

 
 
**Results**
 
![Results](img/Fig1.png)

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── mae_v3.yml                    # run config: geometry, encoder, LoRA, train, infer, cache
│   └── eval.yml                      # evaluation: references + one entry per prediction set
├── data/
│   └── captions/
│       └── captions.csv              # human-verified fracture captions
├── scripts/
│   ├── BreakingBadDataset.py         # mesh -> curvature -> FPS -> tangent-disk tensors
│   ├── caption_fractures.py          # stage 1: multi-image VLM captioning
│   ├── run_config.py                 # shared YAML loader (--config) for all pipeline scripts
│   ├── train_mae.py                  # stage 2: MAE projector + LoRA training
│   └── infer_mae.py                  # inference on one OBJ or a CSV batch
├── tools/
│   ├── render_views/
│   │   ├── render_multiview.py       # render overview + fracture-hemisphere views (frac and base)
│   │   └── make_pair_images.py       # stitch base|frac renders, write pairs.csv
│   ├── cache_data/
│   │   ├── cache_dataset.py          # precompute geometry .pt caches (parallel)
│   │   └── check_cache.py            # validate the caches (header / full check)
│   └── verify_captions/
│       └── caption_evaluator.ipynb   # human verification / correction of captions
├── evaluation/
│   ├── eval_captions.py              # all evaluation metrics + tables (also a CLI)
│   └── caption_evaluation.ipynb      # notebook front end for eval_captions.py
├── tests/                            # pytest suite (python -m pytest -q tests)
└── legacy/
    ├── train_mae_image_only.py       # image-only baseline (training)
    └── infer_mae_image_only.py       # image-only baseline (inference)
```

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `transformers`, `peft`, `bitsandbytes`, `qwen-vl-utils`, `open3d`, `trimesh`, `pyrender`, `pillow`, `scipy`, `pandas`, `numpy`, `pyyaml`. Optional: `wandb`, `flash-attn`, `threadpoolctl`, `tqdm`.

## Data

1. Download the [Breaking Bad dataset](https://breaking-bad-dataset.github.io/) and note its root folder.
2. `data/captions/captions.csv` holds the verified captions. Each row has `path` (OBJ path relative to the dataset root), `class`, `object_description`, `fracture_type`, `confidence`, `caption`, and `parse_error`.

## Pipeline

Run all commands from the repository root. Steps 2–5 read one run config, `configs/mae_v3.yml`, so caching, training and inference always use the same geometry and encoder settings. Command-line flags still override the file, and unknown keys in it are an error. Before the first run, set `data.root` (the Breaking Bad folder) and the train/test caption CSVs in that file.

The scripts import `BreakingBadDataset.py` and `run_config.py` from `scripts/`, so keep both there.

### 0. Render multi-view images

```bash
python tools/render_views/render_multiview.py meshes.csv --views 6 --size 512
```

The CSV needs `path` (fractured piece `.obj`) and `base_path` (complete base `.obj`). A `<stem>_labels.txt` with one integer per face (0 = normal, 1 = fracture) must sit next to each `path` file. For each pair the script writes one overview image and `views-1` cameras spread over the hemisphere facing the fracture, for both the fractured piece (`*_frac_view_*.png`) and the base (`*_base_view_*.png`), next to the fractured `.obj`. Set `PYOPENGL_PLATFORM=egl` on headless servers.

Then stitch each base/fractured pair side by side (intact on the left, fractured on the right) and build the CSV that stage 1 reads:

```bash
python tools/render_views/make_pair_images.py meshes.csv --out pairs.csv
```

This writes `<stem>_pair_view_overview.png` and `<stem>_pair_view_frac00.png` onward, and `pairs.csv` with `path`, `class` (copied from `meshes.csv` if present, else `unknown`), and `pair_images_path`.

### 1. Stage 1: generate candidate captions
 
**Stage 1: multi-image VLM captioning**
 
![Stage 1 captioning](img/Fig3.png)

```bash
python scripts/caption_fractures.py --csv pairs.csv --output captions_raw.csv --n_captions 2
```

The input CSV needs `path`, `class`, and `pair_images_path`. Candidates are then checked and corrected with `tools/verify_captions/caption_evaluator.ipynb`.

### 2. Precompute geometry caches (recommended)

```bash
python tools/cache_data/cache_dataset.py --config configs/mae_v3.yml   # train + test CSVs
python tools/cache_data/check_cache.py   --config configs/mae_v3.yml   # fast header check
python tools/cache_data/check_cache.py   --config configs/mae_v3.yml --check_mode full   # also NaN/Inf
```

Caches are written next to each OBJ, with the geometry settings encoded in the filename. Parallel caching starts workers with `spawn`, caps each worker's BLAS/OpenMP threads (`cache.threads_per_worker`, default `cores // workers`), writes files atomically, and retries the meshes of a crashed worker. Set `cache.workers` in the config. `cache.max_tasks_per_child` recycles workers on Python ≥ 3.11.

### 3. Train

```bash
python scripts/train_mae.py --config configs/mae_v3.yml
python scripts/train_mae.py --config configs/mae_v3.yml --epochs 1 --wandb null   # flags override
python scripts/train_mae.py --config configs/mae_v3.yml --resume checkpoints/mae_v3/ckpt_step1000
```

The fully resolved config is written to `<out>/run_config.yml` and copied into every checkpoint. Notes on the training setup:

- **Two stages.** Stage 1 trains the CTE alone for `train.freeze_lm_steps`. Stage 2 trains LoRA and the CTE jointly. With `lr_schedule: phase2`, the LoRA learning-rate cycle (warmup + decay) starts when LoRA is unfrozen.
- **MAE loss.** The decoder reconstructs the masked cells' curvature from the *encoded* visible cells, so `mae_lam` trains the encoder.
- **Validation split.** `val_split_by: object` keeps every caption of a source mesh in the same split.
- **Point-only ablation.** Set `geometry.use_tangent_disk: false`, which also forces `disk_grid: 1`.

Passing every hyperparameter as a flag, without `--config`, still works.

### 4. Inference

```bash
python scripts/infer_mae.py --config configs/mae_v3.yml                    # test CSV -> infer.out
python scripts/infer_mae.py --ckpt checkpoints/mae_v3/ckpt_best --obj path/to/fragment.obj
```

Without `--config`, settings are read from `<ckpt>/run_config.yml`. Before generating, the geometry and encoder settings are checked against the checkpoint's saved config, and a mismatch stops the run. With `infer.num_samples: 2` and `do_sample: true`, the output has `caption_idx` 0/1 (Prediction 1/2) for each object.

### 5. Evaluation

`configs/eval.yml` lists the human-verified references once and one entry per prediction CSV. Every entry becomes one row of every table, so adding an ablation is one entry:

```yaml
predictions:
  curvature_disk: {csv: outputs/predictions_mae_v3.csv, label: Curvature disk (ours), run_config: configs/mae_v3.yml}
  point_only:     {csv: outputs/predictions_point_only.csv, label: Point-only}
  image_only:     {csv: outputs/predictions_image_only.csv, label: Image-only, enabled: false}
```

```bash
python evaluation/eval_captions.py --config configs/eval.yml
python evaluation/eval_captions.py --config configs/eval.yml --only curvature_disk --pred extra=outputs/preds.csv
```

or open `evaluation/caption_evaluation.ipynb`, which switches to the repository root itself. It reports:

- **CLIP metrics.** CLIPScore (`100·cos`), Ref-Text, RefCLIPScore and CLIP R@k.
- **Collapse.** Unique / top-1 / duplicate share per class, distinct-n, CLIP near-duplicate rate and intra-class caption similarity.
- **CIDEr-R.** The gap against the human reference-vs-reference baseline, with a paired bootstrap and Bonferroni correction.
- **Fracture type.** BOUNDARY / INTERIOR accuracy.
- **Human scores.** Optional, with Spearman correlation against each metric.
- **Summary table.** One row per prediction set, in the layout of the ablation table.

Results go to `output_dir`: `summary.txt`, CSVs, `tables.tex` (LaTeX) and a provenance table of the run settings behind each prediction set. To reproduce the earlier notebook's numbers exactly, set `primary_pred: 1`, `cider_baseline: legacy` and `r_precision_ties: index` under `metrics:`.

## Tests

```bash
python -m pytest -q tests
```

The suite runs on CPU and needs no CLIP or Qwen download. It covers the config loader (the YAML reproduces the v3 command, strict keys, the train/infer consistency check), the LR schedule (checked against `torch.optim.lr_scheduler.OneCycleLR`), the MAE gradient path, and the evaluation metrics.

## Notes

- A 4-bit base model (`model.load_4bit: true`) keeps the 7B model within a single consumer GPU. Lower `encoder.mae_chunk_size` if you run out of memory.
- Changing `n_fps`, `disk_grid`, `disk_radius_frac`, `curv_knn` or `fourier_bands` needs new caches, because they are part of the cache filename. `infer_mae.py` refuses to run on a checkpoint trained with different geometry or encoder settings.
- `max_geo_tokens` < `n_fps` average-pools neighbouring prefix tokens in FPS order. Set it equal to `n_fps` to give the LLM one token per curvature disk.
- `legacy/` holds the image-only baselines and is not needed to reproduce the main results. These scripts import `BreakingBadDataset` from `scripts/`, so run them with `scripts/` on `PYTHONPATH` or copy the dataset file next to them.

## Citation

TBD

## License

TBD.