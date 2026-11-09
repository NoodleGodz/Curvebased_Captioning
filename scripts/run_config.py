"""
run_config.py — one YAML file drives cache_dataset.py, check_cache.py, train_mae.py and infer_mae.py
=================================================================================

Precedence (highest wins):   command-line flag  >  YAML value  >  code default

    python tools/cache_data/cache_dataset.py --config configs/mae_v3.yml
    python tools/cache_data/check_cache.py   --config configs/mae_v3.yml
    python scripts/train_mae.py              --config configs/mae_v3.yml
    python scripts/infer_mae.py              --config configs/mae_v3.yml
    python scripts/train_mae.py --config configs/mae_v3.yml --epochs 1   # CLI still overrides

Every key in the YAML must belong to a known section (typos are an error, not
silently ignored). Each script reads the shared sections plus its own section;
keys meant for another script are ignored by it.

train_mae.py writes the fully resolved config to <out>/run_config.yml and into
every checkpoint dir. infer_mae.py compares the geometry/encoder settings it is
about to use against <ckpt>/run_config.yml and stops on any mismatch, and if
you run infer without --config it falls back to that saved file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Schema: section -> allowed keys. Keys are argparse `dest` names, except in
# `data`, which is mapped per script (train_csv / test_csv -> --csv).
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA: dict[str, set[str]] = {
    "data": {"train_csv", "test_csv", "root", "classes", "val_split",
             "val_split_by", "seed"},
    "geometry": {"n_fps", "use_tangent_disk", "disk_grid", "disk_radius_frac",
                 "disk_fill", "curv_knn", "fourier_bands", "curv_channels"},
    "encoder": {"proj_hidden", "mae_enc_layers", "mae_dec_layers", "mae_n_heads",
                "mae_mask_ratio", "mae_lam", "mae_chunk_size", "proj_dropout",
                "max_geo_tokens"},
    "model": {"model_id", "load_4bit", "flash_attn"},
    "lora": {"lora_r", "lora_alpha", "lora_dropout", "lora_target_modules"},
    "prompt": {"target_fields"},
    "train": {"out", "epochs", "batch", "grad_accum", "lr", "proj_lr",
              "warmup_ratio", "lr_schedule", "weight_decay", "grad_clip", "max_seq_len",
              "freeze_lm_steps", "refreeze_on_new_lora_modules", "div_lam",
              "dataloader_workers", "resume", "wandb", "log_steps",
              "save_steps", "eval_steps", "log_target_samples",
              "sanity_check_every_save", "sanity_sample_idx",
              "sanity_max_new_tokens"},
    "infer": {"ckpt", "out", "load_4bit", "max_new_tokens", "do_sample",
              "temperature", "top_p", "num_samples", "device",
              "allow_config_mismatch"},
    # check_cache.py + cache_dataset.py
    "cache": {"delete_bad", "quiet", "workers", "threads_per_worker",
              "max_tasks_per_child", "force", "skip_errors",
              "check_mode", "check_workers"},
    # caption_evaluation.ipynb / eval_captions.py (keep in sync with EVAL_DEFAULTS)
    "eval": {"true_csv", "predictions", "images_csv", "image_root", "output_dir",
             "class_col_true", "class_col_pred", "class_regex", "primary_pred",
             "clip_model", "clip_w", "clipscore_report", "device", "img_batch_size",
             "txt_batch_size", "embed_cache", "r_precision_k", "r_precision_ties", "near_dup_threshold",
             "distinct_n", "cider_n_max", "cider_sigma", "cider_baseline",
             "n_bootstrap", "bootstrap_seed", "alpha", "human_scores", "n_examples"},
}
TOP_LEVEL_KEYS = {"run_name"}

SHARED_SECTIONS = ["geometry", "encoder", "model", "lora", "prompt"]
SCRIPT_SECTION = {"train": "train", "infer": "infer", "check_cache": "cache",
                  "cache_dataset": "cache"}

# Settings that must be identical between training and inference — a mismatch
# means the checkpoint sees differently-shaped or differently-built geometry.
CONSISTENCY_KEYS = [
    "n_fps", "use_tangent_disk", "disk_grid", "disk_radius_frac", "disk_fill",
    "curv_knn", "fourier_bands", "curv_channels",
    "proj_hidden", "mae_enc_layers", "mae_dec_layers", "mae_n_heads",
    "max_geo_tokens", "model_id", "target_fields",
]

SAVED_CONFIG_NAME = "run_config.yml"


# ──────────────────────────────────────────────────────────────────────────────
# Loading + validation
# ──────────────────────────────────────────────────────────────────────────────

class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: top level must be a mapping of sections")
    validate(cfg, source=str(path))
    return cfg


def validate(cfg: dict, source: str = "config") -> None:
    errors = []
    for sec, body in cfg.items():
        if sec in TOP_LEVEL_KEYS:
            continue
        if sec not in SCHEMA:
            errors.append(f"unknown section '{sec}' (known: {sorted(SCHEMA)})")
            continue
        if body is None:
            continue
        if not isinstance(body, dict):
            errors.append(f"section '{sec}' must be a mapping")
            continue
        for key in body:
            if key not in SCHEMA[sec]:
                errors.append(f"unknown key '{sec}.{key}' "
                              f"(allowed in '{sec}': {sorted(SCHEMA[sec])})")
    if errors:
        raise ConfigError(f"{source}:\n  - " + "\n  - ".join(errors))


def values_for_script(cfg: dict, script: str) -> dict[str, Any]:
    """Flatten the sections a given script reads into {dest: value}."""
    if script not in SCRIPT_SECTION:
        raise ValueError(f"unknown script '{script}'")
    vals: dict[str, Any] = {}
    for sec in SHARED_SECTIONS:
        vals.update(cfg.get(sec) or {})

    if script == "infer":
        # Inference precision is chosen only by infer.load_4bit (default bf16);
        # model.load_4bit describes how the base was loaded for training.
        vals.pop("load_4bit", None)

    data = cfg.get("data") or {}
    if "root" in data:
        vals["root"] = data["root"]
    if script == "train":
        if "train_csv" in data:
            vals["csv"] = data["train_csv"]
        for k in ("classes", "val_split", "val_split_by", "seed"):
            if k in data:
                vals[k] = data[k]
    elif script == "infer":
        if "test_csv" in data:
            vals["csv"] = data["test_csv"]
    elif script in ("check_cache", "cache_dataset"):
        csvs = [data[k] for k in ("train_csv", "test_csv") if data.get(k)]
        if csvs:
            vals["csv"] = csvs

    # Script section last: it overrides shared keys (e.g. infer.load_4bit).
    vals.update(cfg.get(SCRIPT_SECTION[script]) or {})

    if script == "infer" and "ckpt" not in vals:
        train_out = (cfg.get("train") or {}).get("out")
        if train_out:
            vals["ckpt"] = str(Path(train_out) / "ckpt_best")
    return vals


# ──────────────────────────────────────────────────────────────────────────────
# argparse integration
# ──────────────────────────────────────────────────────────────────────────────

def parse_with_config(
    parser: argparse.ArgumentParser,
    script: str,
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    """
    Add --config to `parser`, load the YAML (if any) as parser defaults, then
    parse argv. CLI flags still win. Returns the parsed Namespace with
    `args.config` set and `args.cli_overrides` listing keys given on the CLI.
    """
    parser.add_argument(
        "--config", default=None,
        help="YAML run config (see run_config.py). CLI flags override it.")
    argv = sys.argv[1:] if argv is None else list(argv)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--ckpt", default=None)
    known, _ = pre.parse_known_args(argv)

    config_path = known.config
    # infer: no --config given -> fall back to the config saved with the ckpt
    if config_path is None and script == "infer" and known.ckpt:
        saved = Path(known.ckpt) / SAVED_CONFIG_NAME
        if saved.is_file():
            config_path = str(saved)
            log.info(f"No --config given; using config saved with checkpoint: {saved}")

    actions = {a.dest: a for a in parser._actions}
    yaml_vals: dict[str, Any] = {}
    if config_path:
        cfg = load_yaml(config_path)
        yaml_vals = {k: v for k, v in values_for_script(cfg, script).items()
                     if k in actions}
        for dest, val in yaml_vals.items():
            act = actions[dest]
            if act.choices is not None and val is not None:
                items = val if isinstance(val, list) else [val]
                for it in items:
                    if it not in act.choices:
                        raise ConfigError(
                            f"{config_path}: '{dest}: {it}' not in allowed "
                            f"choices {list(act.choices)}")
            act.required = False          # satisfied by the YAML
        parser.set_defaults(**yaml_vals)

    args = parser.parse_args(argv)
    args.config = config_path
    flags = {tok.split("=", 1)[0].lstrip("-") for tok in argv if tok.startswith("--")}
    args.cli_overrides = sorted(
        a.dest for a in parser._actions
        if any(o.lstrip("-") in flags for o in a.option_strings) and a.dest != "config"
    )
    normalize(args)
    return args


def normalize(args: argparse.Namespace) -> None:
    """Fix up settings that are only meaningful in combination."""
    if getattr(args, "use_tangent_disk", True) is False and getattr(args, "disk_grid", 1) != 1:
        log.warning(
            f"use_tangent_disk=False (point mode) but disk_grid={args.disk_grid}: "
            f"point mode always produces 1x1 disks, forcing disk_grid=1 so the "
            f"cache name and the projector's spatial grid match the data.")
        args.disk_grid = 1


def resolved_sections(args: argparse.Namespace, script: str = "train") -> dict:
    """Inverse of values_for_script: rebuild a sectioned config from args."""
    a = vars(args)
    out: dict[str, Any] = {}
    if a.get("run_name"):
        out["run_name"] = a["run_name"]
    data = {}
    if "csv" in a:
        data["train_csv" if script == "train" else "test_csv"] = a["csv"]
    for k in ("root", "classes", "val_split", "val_split_by", "seed"):
        if k in a:
            data[k] = a[k]
    out["data"] = data
    for sec in SHARED_SECTIONS + [SCRIPT_SECTION[script]]:
        body = {k: a[k] for k in sorted(SCHEMA[sec]) if k in a}
        if body:
            out[sec] = body
    return out


def dump_resolved(args: argparse.Namespace, path: str | Path, script: str = "train") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Resolved {script} config — written automatically.\n"
              f"# source config: {getattr(args, 'config', None)}\n"
              f"# CLI overrides: {getattr(args, 'cli_overrides', [])}\n")
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(resolved_sections(args, script), f, sort_keys=False)
    return path


def check_consistency(args: argparse.Namespace, saved_config: str | Path) -> list[str]:
    """Compare args against a saved training config. Returns mismatch messages."""
    saved_config = Path(saved_config)
    saved = yaml.safe_load(saved_config.read_text()) or {}
    flat: dict[str, Any] = {}
    for sec in SHARED_SECTIONS:
        flat.update(saved.get(sec) or {})
    problems = []
    for k in CONSISTENCY_KEYS:
        if k not in flat or not hasattr(args, k):
            continue
        want, got = flat[k], getattr(args, k)
        if isinstance(want, float) or isinstance(got, float):
            same = want is not None and got is not None and abs(float(want) - float(got)) < 1e-9
        else:
            same = want == got
        if not same:
            problems.append(f"{k}: checkpoint trained with {want!r}, now {got!r}")
    return problems


def log_resolved(args: argparse.Namespace, keys: Optional[list[str]] = None) -> None:
    keys = keys or sorted(k for k in vars(args) if k not in ("cli_overrides",))
    log.info(f"Config file: {args.config or '(none — code defaults + CLI)'}")
    if args.cli_overrides:
        log.info(f"CLI overrides: {args.cli_overrides}")
    for k in keys:
        log.info(f"  {k:<28} {getattr(args, k, None)!r}")
