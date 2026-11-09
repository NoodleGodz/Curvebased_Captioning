"""
eval_captions.py — caption evaluation driven by the run YAML
============================================================

Computes every table in the paper's results section from the same run config
used by cache_dataset / train_mae / infer_mae:

  * CLIP metrics       CLIPScore, Ref-Text, RefCLIPScore, CLIP R@k     (tab:clip-metrics)
  * Collapse           unique / top-1 / duplicate per class            (tab:collapse-stats)
                       + distinct-n, CLIP near-duplicate rate, intra-class text similarity
  * CIDEr-R            prediction-vs-reference gap, paired bootstrap,  (tab:cider-bootstrap)
                       Bonferroni over classes
  * Headline row       one line per prediction set                     (tab:conditioning_ablation)
  * Fracture type      accuracy vs the references (when labels exist)
  * Human scores       optional: caption_evaluator.ipynb output + Spearman vs metrics

Used by caption_evaluation.ipynb, or headless:

    python evaluation/eval_captions.py --config configs/eval.yml     # recommended
    python evaluation/eval_captions.py --config configs/eval.yml --pred extra=preds_extra.csv
    python evaluation/eval_captions.py --config configs/mae_v3.yml   # one run, from infer.out

eval.yml lists the references, one entry per prediction CSV (= one ablation
row, with an optional label / run_config / human_scores / enabled flag), and
the metric settings. Paths left out come from its `run_config` (data.test_csv,
data.root, infer.out). See eval.yml for the full format.

Metrics are computed in NumPy; only CLIPEmbedder needs torch/transformers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────

EVAL_DEFAULTS: dict[str, Any] = {
    # files
    "true_csv": None,            # default: data.test_csv
    "predictions": None,         # {name: csv}; default {run_name: infer.out}
    "images_csv": None,          # CSV with images_used, if true_csv lacks it
    "image_root": None,          # default: data.root
    "output_dir": None,          # default: eval_outputs_<run_name>
    # class resolution
    "class_col_true": "class",
    "class_col_pred": "true_class",
    "class_regex": r"everyday/([^/]+)/",
    # which prediction sample feeds single-caption tables (collapse, CIDEr bootstrap)
    "primary_pred": 0,           # caption_idx; the paper's tables used 1 (see notes)
    # CLIP
    "clip_model": "openai/clip-vit-large-patch14",
    "clip_w": 2.5,               # Hessel et al. CLIP-S weight (used inside RefCLIP)
    "clipscore_report": "cos100",  # cos100 = 100·cos (paper) | hessel = w·cos·100
    "device": "auto",
    "img_batch_size": 32,
    "txt_batch_size": 64,
    "embed_cache": True,
    "r_precision_k": [1, 5],
    "r_precision_ties": "expected",  # expected | index (old notebook, torch.topk order)
    # diversity
    "near_dup_threshold": 0.95,  # CLIP text cosine for "near-duplicate"
    "distinct_n": 2,
    # CIDEr-R + bootstrap
    "cider_n_max": 4,
    "cider_sigma": 6.0,
    "cider_baseline": "symmetric",  # symmetric | legacy (ref0 vs ref1 only, old notebook)
    "n_bootstrap": 10_000,
    "bootstrap_seed": 42,
    "alpha": 0.05,
    # optional
    "human_scores": None,        # {pred_name: scored_csv from caption_evaluator.ipynb}
    "n_examples": 3,             # qualitative examples (worst RefCLIP)
}
EVAL_KEYS = set(EVAL_DEFAULTS)


# ── eval.yml (stand-alone evaluation file) ───────────────────────────────────
#   name, run_config, output_dir,
#   references: {true_csv, images_csv, image_root, class_col_true, class_regex}
#   predictions: {<name>: <csv> | {csv, label, run_config, human_scores, enabled, notes}}
#   metrics: every other EVAL_DEFAULTS key
EVAL_FILE_KEYS = {"name", "run_config", "output_dir", "references", "predictions", "metrics"}
REFERENCE_KEYS = {"true_csv", "images_csv", "image_root", "class_col_true", "class_regex"}
PREDICTION_KEYS = {"csv", "label", "run_config", "human_scores", "enabled", "notes"}
METRIC_KEYS = EVAL_KEYS - REFERENCE_KEYS - {"predictions", "output_dir", "human_scores"}
VARIANT_FIELDS = [("geometry", "n_fps"), ("geometry", "use_tangent_disk"), ("geometry", "disk_grid"),
                  ("geometry", "curv_channels"), ("encoder", "mae_mask_ratio"), ("encoder", "mae_lam"),
                  ("encoder", "max_geo_tokens"), ("lora", "lora_r"), ("train", "epochs"),
                  ("infer", "ckpt"), ("infer", "do_sample"), ("infer", "temperature")]


def _read_yaml(path: Path) -> dict:
    import yaml
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return cfg


def _read_run_config(path: Path) -> dict:
    """Run config (mae_v3.yml) with the training scripts' strict schema check."""
    try:
        here = Path(__file__).resolve().parent
        for p in (here.parent / "scripts", here):        # repo layout, then flat layout
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from run_config import load_yaml
        return load_yaml(path)
    except ImportError:
        return _read_yaml(path)


def is_eval_file(cfg: dict) -> bool:
    return bool({"references", "predictions", "metrics"} & set(cfg)) and "data" not in cfg


def _parse_eval_file(cfg: dict, path: Path) -> tuple[dict, dict, str]:
    """eval.yml → (flat eval settings, run config dict, name). Strict keys."""
    errors = [f"unknown top-level key '{k}' (allowed: {sorted(EVAL_FILE_KEYS)})"
              for k in cfg if k not in EVAL_FILE_KEYS]
    refs, metrics = cfg.get("references") or {}, cfg.get("metrics") or {}
    errors += [f"unknown key 'references.{k}' (allowed: {sorted(REFERENCE_KEYS)})"
               for k in refs if k not in REFERENCE_KEYS]
    errors += [f"unknown key 'metrics.{k}' (allowed: {sorted(METRIC_KEYS)})"
               for k in metrics if k not in METRIC_KEYS]
    for name, entry in (cfg.get("predictions") or {}).items():
        if isinstance(entry, dict):
            errors += [f"unknown key 'predictions.{name}.{k}' (allowed: {sorted(PREDICTION_KEYS)})"
                       for k in entry if k not in PREDICTION_KEYS]
            if "csv" not in entry:
                errors.append(f"predictions.{name}: missing 'csv'")
        elif not isinstance(entry, str):
            errors.append(f"predictions.{name}: expected a CSV path or a mapping")
    if errors:
        raise ValueError(f"{path}:\n  - " + "\n  - ".join(errors))

    run_cfg = _read_run_config(Path(cfg["run_config"])) if cfg.get("run_config") else {}
    flat = {**refs, **metrics}
    if cfg.get("predictions"):
        flat["predictions"] = cfg["predictions"]
    if cfg.get("output_dir"):
        flat["output_dir"] = cfg["output_dir"]
    name = cfg.get("name") or run_cfg.get("run_name") or path.stem
    return flat, run_cfg, name


def _normalize_predictions(preds, extra_human: dict) -> tuple[dict, dict, dict, dict, dict]:
    """{name: csv | {...}} → (csv, label, run_config, human_scores, notes) maps,
    in file order, skipping `enabled: false` entries."""
    csv, label, rcfg, human, notes = {}, {}, {}, {}, {}
    for name, entry in preds.items():
        e = {"csv": entry} if isinstance(entry, (str, Path)) else dict(entry)
        if e.get("enabled", True) is False:
            continue
        name = str(name)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"prediction name '{name}' must be a file-safe identifier "
                             f"(letters, digits, _ . -); put display text in 'label'")
        csv[name] = str(e["csv"])
        label[name] = str(e.get("label") or name)
        if e.get("run_config"):
            rcfg[name] = str(e["run_config"])
        if e.get("human_scores"):
            human[name] = str(e["human_scores"])
        if e.get("notes"):
            notes[name] = str(e["notes"])
    for name, f in (extra_human or {}).items():
        human.setdefault(name, f)
    return csv, label, rcfg, human, notes


def load_settings(config: Optional[str | Path] = None,
                  overrides: Optional[dict] = None) -> dict:
    """
    Resolve eval settings from either
      * an eval file (eval.yml: references / predictions / metrics), or
      * a run config (mae_v3.yml) with an optional `eval:` section (legacy).
    Precedence: defaults < file < overrides. Unset paths fall back to the run
    config's data.test_csv / data.root / infer.out.
    """
    cfg, run_cfg, name, fmt = {}, {}, None, None
    if config:
        config = Path(config)
        cfg = _read_yaml(config)
        if is_eval_file(cfg):
            flat, run_cfg, name = _parse_eval_file(cfg, config)
            fmt = "eval file"
        else:
            run_cfg = _read_run_config(config)
            flat, fmt = dict(run_cfg.get("eval") or {}), "run config"

    s = dict(EVAL_DEFAULTS)
    s.update(flat if config else {})
    unknown = set(overrides or {}) - EVAL_KEYS
    if unknown:
        raise KeyError(f"Unknown eval setting(s): {sorted(unknown)}. Allowed: {sorted(EVAL_KEYS)}")
    s.update(overrides or {})

    data, infer = run_cfg.get("data") or {}, run_cfg.get("infer") or {}
    run_name = name or run_cfg.get("run_name") or "model"
    s["run_name"] = run_name
    s["config"] = str(config) if config else None
    s["config_format"] = fmt
    s["run_config"] = str(cfg["run_config"]) if fmt == "eval file" and cfg.get("run_config") else \
        (str(config) if fmt == "run config" else None)
    s["true_csv"] = s["true_csv"] or data.get("test_csv")
    if not s["predictions"]:
        if not infer.get("out"):
            raise ValueError("No predictions: add a `predictions:` entry (or infer.out in the run config)")
        s["predictions"] = {run_cfg.get("run_name") or run_name: infer["out"]}
    elif isinstance(s["predictions"], (str, Path)):
        s["predictions"] = {run_name: str(s["predictions"])}
    (s["predictions"], s["labels"], s["variant_run_configs"],
     s["human_scores"], s["variant_notes"]) = _normalize_predictions(s["predictions"], s["human_scores"])
    if not s["predictions"]:
        raise ValueError("Every prediction set is disabled (enabled: false)")
    if fmt == "run config" and s["run_config"]:
        for n in s["predictions"]:                 # legacy format: all sets from this run
            s["variant_run_configs"].setdefault(n, s["run_config"])
    s["image_root"] = s["image_root"] or data.get("root") or "."
    s["output_dir"] = s["output_dir"] or f"eval_outputs_{run_name}"
    s["r_precision_k"] = sorted({int(k) for k in np.atleast_1d(s["r_precision_k"])})

    if not s["true_csv"]:
        raise ValueError("No reference CSV: set eval.true_csv or data.test_csv")
    if s["cider_baseline"] not in ("symmetric", "legacy"):
        raise ValueError("cider_baseline must be 'symmetric' or 'legacy'")
    if s["r_precision_ties"] not in ("expected", "index"):
        raise ValueError("r_precision_ties must be 'expected' or 'index'")
    if s["clipscore_report"] not in ("cos100", "hessel"):
        raise ValueError("clipscore_report must be 'cos100' or 'hessel'")
    unknown_h = set(s["human_scores"]) - set(s["predictions"])
    if unknown_h:
        raise ValueError(f"human_scores for unknown/disabled prediction set(s): {sorted(unknown_h)}")
    missing = [p for p in [s["true_csv"], s["images_csv"], *s["predictions"].values(),
                           *s["human_scores"].values(), *s["variant_run_configs"].values()]
               if p and not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {missing}")
    return s


def prediction_entries(s: dict) -> dict:
    """Resolved settings → full {name: {csv, label, run_config, human_scores}} entries,
    e.g. to add or drop a prediction set and call load_settings again."""
    out = {}
    for n, f in s["predictions"].items():
        e = {"csv": f, "label": label(s, n)}
        if n in s["variant_run_configs"]:
            e["run_config"] = s["variant_run_configs"][n]
        if n in s["human_scores"]:
            e["human_scores"] = s["human_scores"][n]
        out[n] = e
    return out


def label(s: dict, name: str) -> str:
    return s.get("labels", {}).get(name, name)


def describe_settings(s: dict) -> str:
    keys = ["config", "config_format", "run_config", "true_csv", "images_csv", "image_root",
            "output_dir", "primary_pred", "clip_model", "clipscore_report", "r_precision_k",
            "r_precision_ties", "cider_baseline", "n_bootstrap"]
    lines = [f"  {k:<18} {s.get(k)!r}" for k in keys]
    lines.append("  prediction sets:")
    for n, f in s["predictions"].items():
        extra = [f"run_config={s['variant_run_configs'][n]}"] if n in s["variant_run_configs"] else []
        extra += [f"human_scores={s['human_scores'][n]}"] if n in s["human_scores"] else []
        lines.append(f"    {n:<16} {label(s, n)!r:<28} {f}" + (f"  ({', '.join(extra)})" if extra else ""))
    return "\n".join(lines)


def variants_table(s: dict) -> pd.DataFrame:
    """One row per prediction set: label, CSV, and the key settings of the run
    config that produced it (provenance for the ablation table)."""
    rows = []
    for n, f in s["predictions"].items():
        r = {"set": n, "label": label(s, n), "csv": f, "run_config": s["variant_run_configs"].get(n, "")}
        rc = s["variant_run_configs"].get(n)
        if rc:
            try:
                c = _read_yaml(Path(rc))
                for sec, key in VARIANT_FIELDS:
                    r[key] = (c.get(sec) or {}).get(key)
                test_csv = (c.get("data") or {}).get("test_csv")
                if test_csv and Path(test_csv).name != Path(s["true_csv"]).name:
                    r["warning"] = f"run config test_csv={test_csv} ≠ references {s['true_csv']}"
            except Exception as e:
                r["warning"] = f"could not read run config: {e}"
        r["notes"] = s.get("variant_notes", {}).get(n, "")
        rows.append(r)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

FT_RE = re.compile(r"\b(BOUNDARY|INTERIOR)\b", re.IGNORECASE)


def parse_list(raw) -> list:
    if isinstance(raw, list):
        return [str(p) for p in raw]
    if not isinstance(raw, str) or not raw.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            v = parser(raw)
            if isinstance(v, list):
                return [str(p) for p in v]
        except Exception:
            continue
    return []


def _clean_str(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def _json_field(raw: str, key: str) -> str:
    raw = _clean_str(raw).strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        raw = raw[4:] if raw.startswith("json") else raw
    try:
        obj = json.loads(raw.strip())
        return _clean_str(obj.get(key, "")) if isinstance(obj, dict) else ""
    except Exception:
        return ""


def fracture_type_of(row) -> tuple[str, str]:
    """→ (BOUNDARY | INTERIOR | "", source). Column, then raw JSON, then caption text."""
    for src, val in (("column", _clean_str(row.get("fracture_type"))),
                     ("raw_output", _json_field(row.get("raw_output"), "fracture_type")),
                     ("caption", _clean_str(row.get("caption")))):
        m = FT_RE.search(val)
        if m:
            return m.group(1).upper(), src
    return "", ""


def _is_true(v) -> bool:
    return _clean_str(v).strip().upper() in ("TRUE", "1", "YES")


def load_long(csv_path: str, class_col: Optional[str]) -> pd.DataFrame:
    """One row per (path, caption_idx) with resolved fracture type and flags."""
    t = pd.read_csv(csv_path)
    t.columns = [c.strip() for c in t.columns]
    if "path" not in t.columns or "caption" not in t.columns:
        raise ValueError(f"{csv_path}: needs 'path' and 'caption' columns")
    t["path"] = t["path"].astype(str).str.strip()
    if "caption_idx" not in t.columns:
        t["caption_idx"] = t.groupby("path").cumcount()
    t["caption_idx"] = t["caption_idx"].astype(int)
    dup = t.duplicated(["path", "caption_idx"])
    if dup.any():
        log.warning(f"{csv_path}: {dup.sum()} duplicate (path, caption_idx) rows — keeping first")
        t = t[~dup]
    t["caption"] = t["caption"].map(_clean_str)
    ft = t.apply(fracture_type_of, axis=1, result_type="expand")
    t["ftype"], t["ftype_src"] = ft[0], ft[1]
    t["parse_err"] = t["parse_error"].map(_is_true) if "parse_error" in t.columns else False
    t["cls"] = t[class_col] if class_col and class_col in t.columns else pd.NA
    return t


@dataclass
class EvalData:
    objects: pd.DataFrame                    # one row per evaluated object
    true_cols: list[str]                     # caption_true_0, caption_true_1, ...
    preds: dict[str, pd.DataFrame]           # name -> wide frame aligned with objects
    pred_cols: dict[str, list[str]]          # name -> caption_pred_0, ...
    coverage: pd.DataFrame                   # per prediction set: objects matched / missing
    notes: list[str] = field(default_factory=list)


def _wide(long: pd.DataFrame, prefix: str, value: str) -> pd.DataFrame:
    w = long.pivot(index="path", columns="caption_idx", values=value)
    w.columns = [f"{prefix}_{int(c)}" for c in w.columns]
    return w


def load_data(s: dict) -> EvalData:
    notes = []
    true_long = load_long(s["true_csv"], s["class_col_true"])
    objects = _wide(true_long, "caption_true", "caption")
    true_cols = list(objects.columns)
    if len(true_cols) < 2:
        raise ValueError("Need ≥ 2 reference captions per object for the human baseline")
    for c in true_cols:
        objects[c] = objects[c].fillna("")
    objects = objects.join(_wide(true_long, "ftype_true", "ftype").fillna(""))
    first = true_long.groupby("path").first()
    objects["class"] = first["cls"]
    if "images_used" in true_long.columns:
        objects["images_used"] = first["images_used"]
    if s["images_csv"]:
        img = pd.read_csv(s["images_csv"])
        img.columns = [c.strip() for c in img.columns]
        img["path"] = img["path"].astype(str).str.strip()
        lookup = img.groupby("path")["images_used"].first()
        base = objects["images_used"] if "images_used" in objects else pd.Series(pd.NA, index=objects.index)
        objects["images_used"] = base.fillna(objects.index.to_series().map(lookup))
    if "images_used" not in objects:
        objects["images_used"] = "[]"
    objects["images_used"] = objects["images_used"].fillna("[]")

    preds_long = {n: load_long(p, s["class_col_pred"]) for n, p in s["predictions"].items()}
    # class: reference column → prediction file's true_class → regex on path
    for pl in preds_long.values():
        objects["class"] = objects["class"].fillna(pl.groupby("path")["cls"].first())
    rx = objects.index.to_series().str.extract(s["class_regex"])[0]
    objects["class"] = objects["class"].fillna(rx)
    if objects["class"].isna().any():
        notes.append(f"{objects['class'].isna().sum()} objects have no resolvable class "
                     f"(grouped as 'UNKNOWN')")
        objects["class"] = objects["class"].fillna("UNKNOWN")

    # evaluate on objects that every prediction set covers (comparable variants)
    cov_rows, keep = [], set(objects.index)
    for n, pl in preds_long.items():
        have = set(pl["path"])
        extra = len(have - set(objects.index))
        cov_rows.append({"prediction_set": n, "file": s["predictions"][n],
                         "ref_objects": len(objects), "matched": len(have & set(objects.index)),
                         "missing": len(set(objects.index) - have), "not_in_refs": extra,
                         "samples_per_object": pl.groupby("path").size().median()})
        keep &= have
    coverage = pd.DataFrame(cov_rows)
    if len(keep) < len(objects):
        notes.append(f"Evaluating {len(keep)} / {len(objects)} reference objects "
                     f"(the rest lack predictions in at least one set — see coverage)")
    if not keep:
        raise ValueError("No object has predictions in every prediction set — check paths")
    objects = objects.loc[sorted(keep)]

    preds, pred_cols = {}, {}
    for n, pl in preds_long.items():
        pl = pl[pl["path"].isin(objects.index)]
        caps = _wide(pl, "caption_pred", "caption").reindex(objects.index)
        cols = list(caps.columns)
        caps = caps.fillna("")
        w = caps.join(_wide(pl, "ftype_pred", "ftype").reindex(objects.index).fillna(""))
        w = w.join(_wide(pl, "parse_err", "parse_err").reindex(objects.index)
                   .fillna(True).astype(bool))
        # samples with no row at all count as parse errors (no caption)
        preds[n], pred_cols[n] = w, cols
    objects = objects.reset_index().rename(columns={"index": "path"})
    for n in preds:
        preds[n] = preds[n].reset_index(drop=True)

    p = s["primary_pred"]
    for n, cols in pred_cols.items():
        if f"caption_pred_{p}" not in cols:
            raise ValueError(f"primary_pred={p}: prediction set '{n}' has only {cols}")
    return EvalData(objects, true_cols, preds, pred_cols, coverage, notes)


def resolve_overview(path: str, images_used, root: Path) -> Optional[Path]:
    direct = root / str(path).replace(".obj", "_frac_view_overview.png")
    if direct.is_file():
        return direct
    cands = [root / p for p in parse_list(images_used)]
    ov = [p for p in cands if "overview" in p.stem.lower()]
    for p in ov + [p for p in cands if p not in ov]:
        if p.is_file():
            return p
    return None


# ──────────────────────────────────────────────────────────────────────────────
# CLIP embeddings (the only torch dependency)
# ──────────────────────────────────────────────────────────────────────────────

class CLIPEmbedder:
    """L2-normalised CLIP image/text embeddings as float32 NumPy, with an
    optional on-disk cache keyed by model + inputs."""

    def __init__(self, model_name: str, device: str = "auto", img_batch_size: int = 32,
                 txt_batch_size: int = 64, cache_dir: Optional[Path] = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).eval().to(self.device)
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.img_bs, self.txt_bs = img_batch_size, txt_batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cached(self, kind: str, items: list[str], fn):
        if not self.cache_dir:
            return fn(items)
        key = hashlib.sha1("\x1f".join([self.model_name, kind, *items]).encode()).hexdigest()[:20]
        f = self.cache_dir / f"{kind}_{key}.npy"
        if f.is_file():
            return np.load(f)
        out = fn(items)
        np.save(f, out)
        return out

    @staticmethod
    def _unwrap(feat):
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            if hasattr(feat, attr) and not hasattr(feat, "shape"):
                return getattr(feat, attr)
        return feat

    def _norm(self, feat):
        f = self.torch.nn.functional.normalize(self._unwrap(feat).float(), dim=-1)
        return f.cpu().numpy().astype(np.float32)

    def images(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image

        def run(items):
            out = []
            with self.torch.no_grad():
                for i in _progress(range(0, len(items), self.img_bs), "CLIP images"):
                    imgs = [Image.open(p).convert("RGB") for p in items[i:i + self.img_bs]]
                    inp = self.proc(images=imgs, return_tensors="pt").to(self.device)
                    out.append(self._norm(self.model.get_image_features(**inp)))
            return np.concatenate(out) if out else np.zeros((0, 1), np.float32)
        return self._cached("img", [str(p) for p in paths], run)

    def texts(self, texts: list[str]) -> np.ndarray:
        def run(items):
            out = []
            with self.torch.no_grad():
                for i in _progress(range(0, len(items), self.txt_bs), "CLIP texts"):
                    inp = self.proc(text=items[i:i + self.txt_bs], return_tensors="pt",
                                    padding=True, truncation=True).to(self.device)
                    out.append(self._norm(self.model.get_text_features(**inp)))
            return np.concatenate(out) if out else np.zeros((0, 1), np.float32)
        return self._cached("txt", [_clean_str(t) for t in texts], run)


def _progress(it, desc):
    try:
        from tqdm.auto import tqdm
        return tqdm(it, desc=desc, leave=False)
    except ImportError:
        return it


# ──────────────────────────────────────────────────────────────────────────────
# CLIP-based metrics
# ──────────────────────────────────────────────────────────────────────────────

def rowcos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.clip(a, 0, None), np.clip(b, 0, None)
    out = np.zeros_like(a, dtype=np.float64)
    nz = (a > 0) & (b > 0)
    out[nz] = 2 * a[nz] * b[nz] / (a[nz] + b[nz])
    return out


def r_precision(img: np.ndarray, txt: np.ndarray, ks, ties: str = "expected",
                tie_tol: float = 1e-6) -> dict[int, float]:
    """Image→text top-k retrieval accuracy within the pool.

    ties="expected": tied similarities (identical captions — exactly the
    collapse case) score their expected value under random tie-breaking, so a
    collapsed caption set cannot earn retrieval credit from index order.
    ties="index": lowest index wins, i.e. the old notebook's torch.topk on CPU."""
    sim = img @ txt.T
    n = len(sim)
    if ties == "index":
        order = np.argsort(-sim, axis=1, kind="stable")
        rank = np.argmax(order == np.arange(n)[:, None], axis=1)
        return {k: float((rank < k).mean()) for k in ks}
    own = np.diag(sim)[:, None]
    greater = (sim > own + tie_tol).sum(axis=1)
    tied = (np.abs(sim - own) <= tie_tol).sum(axis=1) - 1     # exclude self
    return {k: float(np.clip((k - greater) / (tied + 1), 0, 1).mean()) for k in ks}


def clip_metrics(d: EvalData, s: dict, img_emb: np.ndarray, valid: np.ndarray,
                 txt_emb: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """→ (summary table, per-class R@k, per-caption long frame). Only objects
    with an image (`valid`) are used."""
    w, ks = float(s["clip_w"]), s["r_precision_k"]
    scale = 100.0 if s["clipscore_report"] == "cos100" else 100.0 * w
    I = img_emb
    cls = d.objects.loc[valid, "class"].to_numpy()
    ref = {c: txt_emb[c][valid] for c in d.true_cols}

    def score(cand: np.ndarray, refs: list[np.ndarray]):
        cos = np.clip(rowcos(I, cand), 0, None)
        reftext = np.max([np.clip(rowcos(cand, r), 0, None) for r in refs], axis=0)
        return cos, reftext, harmonic(w * cos, reftext)

    rows, per_class, per_cap = [], [], []
    sources = [("references", c, f"Reference {i+1}", [ref[o] for o in d.true_cols if o != c], ref[c])
               for i, c in enumerate(d.true_cols)]
    for n, cols in d.pred_cols.items():
        for c in cols:
            idx = int(c.rsplit("_", 1)[1])
            sources.append((n, c, f"Prediction {idx+1}", list(ref.values()), txt_emb[f"{n}:{c}"][valid]))
    for src, col, label, refs, cand in sources:
        cos, reftext, refclip = score(cand, refs)
        rp = r_precision(I, cand, ks, s["r_precision_ties"])
        row = {"set": src, "caption": label, "column": col, "N": int(valid.sum()),
               "CLIPScore": scale * cos.mean(), "CLIPScore_std": scale * cos.std(),
               "Ref-Text": reftext.mean(), "RefCLIP": refclip.mean()}
        row.update({f"R@{k}": v for k, v in rp.items()})
        rows.append(row)
        for c_ in sorted(set(cls)):
            m = cls == c_
            if m.sum() >= 2:
                per_class.append({"set": src, "caption": label, "class": c_, "n_pool": int(m.sum()),
                                  **{f"R@{k}": v for k, v in r_precision(I[m], cand[m], ks, s["r_precision_ties"]).items()}})
        if src != "references":
            per_cap.append(pd.DataFrame({
                "set": src, "path": d.objects.loc[valid, "path"].to_numpy(), "class": cls,
                "caption_idx": int(col.rsplit("_", 1)[1]), "clipscore": scale * cos,
                "ref_text": reftext, "refclip": refclip}))
    table = pd.DataFrame(rows)
    table.attrs["chance_R@1"] = 1.0 / max(int(valid.sum()), 1)
    return table, pd.DataFrame(per_class), (pd.concat(per_cap, ignore_index=True)
                                            if per_cap else pd.DataFrame())


# ──────────────────────────────────────────────────────────────────────────────
# Collapse + diversity
# ──────────────────────────────────────────────────────────────────────────────

def _norm_caption(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", _clean_str(t).lower())).strip()


def tokenize(text) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", _clean_str(text).lower()).split()


def _collapse_stats(caps: pd.Series, norm) -> dict:
    n = len(caps)
    if n == 0:
        return {"N": 0}
    counts = caps.map(norm).value_counts()
    return {"N": n, "unique_ratio": (counts == 1).sum() / n, "distinct_ratio": len(counts) / n,
            "top1_share": counts.iloc[0] / n, "duplicate_rate": 1 - (counts == 1).sum() / n}


def distinct_n(caps, n: int) -> float:
    grams = [tuple(t[i:i + n]) for c in caps for t in [tokenize(c)] for i in range(len(t) - n + 1)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def semantic_diversity(emb: np.ndarray, cls: np.ndarray, thr: float) -> tuple[dict, dict]:
    """Per class: mean pairwise CLIP text cosine (↓ more diverse) and the share
    of captions whose nearest same-class neighbour has cosine ≥ thr."""
    sim_c, dup_c = {}, {}
    for c in sorted(set(cls)):
        e = emb[cls == c]
        if len(e) < 2:
            continue
        m = e @ e.T
        np.fill_diagonal(m, np.nan)
        sim_c[c] = float(np.nanmean(m))
        dup_c[c] = float((np.nanmax(m, axis=1) >= thr).mean())
    return sim_c, dup_c


def collapse_table(d: EvalData, s: dict, caps: pd.Series, emb: Optional[np.ndarray],
                   valid: Optional[np.ndarray]) -> pd.DataFrame:
    """Per-class literal collapse (old definitions kept: unique = singleton share,
    case-insensitive exact match) plus distinct-n and CLIP-based near-duplicates."""
    cls = d.objects["class"].to_numpy()
    rows = []
    for c in sorted(set(cls)) + ["ALL"]:
        m = np.ones(len(cls), bool) if c == "ALL" else cls == c
        sub = caps[m]
        r = {"class": c, **_collapse_stats(sub, lambda t: _clean_str(t).strip().lower())}
        r["norm_top1_share"] = _collapse_stats(sub, _norm_caption).get("top1_share")
        r[f"distinct_{s['distinct_n']}"] = distinct_n(sub, s["distinct_n"])
        rows.append(r)
    t = pd.DataFrame(rows)
    if emb is not None:
        vcls = cls[valid]
        sim_c, dup_c = semantic_diversity(emb, vcls, s["near_dup_threshold"])
        # ALL = object-weighted mean of class values (cross-class pairs say little)
        w = {c: (vcls == c).sum() for c in sim_c}
        tot = sum(w.values()) or 1
        sim_c["ALL"] = sum(sim_c[c] * w[c] for c in w) / tot
        dup_c["ALL"] = sum(dup_c[c] * w[c] for c in w) / tot
        t["intra_class_sim"] = t["class"].map(sim_c)
        t[f"near_dup@{s['near_dup_threshold']}"] = t["class"].map(dup_c)
    order = t[t["class"] != "ALL"].sort_values("top1_share", ascending=False)
    return pd.concat([order, t[t["class"] == "ALL"]], ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# CIDEr-R
# ──────────────────────────────────────────────────────────────────────────────

class CiderR:
    """CIDEr-D (IDF-weighted n-gram cosine, count clipping, length Gaussian)
    × CIDEr-R repetition penalty (mean distinct-n-gram fraction of the
    candidate). Same mechanism as the previous notebook; TF-IDF vectors are
    memoised. IDF is computed once over the reference corpus."""

    def __init__(self, reference_lists: list[list[str]], n_max: int = 4, sigma: float = 6.0):
        self.n_max, self.sigma = n_max, sigma
        self.n_docs = len(reference_lists)
        self.df = {n: defaultdict(int) for n in range(1, n_max + 1)}
        for refs in reference_lists:
            seen = {n: set() for n in range(1, n_max + 1)}
            for r in refs:
                toks = tokenize(r)
                for n in seen:
                    seen[n].update(self._grams(toks, n))
            for n in seen:
                for g in seen[n]:
                    self.df[n][g] += 1
        self._vec: dict[str, tuple] = {}

    @staticmethod
    def _grams(toks, n):
        return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]

    def _vectors(self, text: str):
        if text not in self._vec:
            toks = tokenize(text)
            vecs, norms = [], []
            for n in range(1, self.n_max + 1):
                counts = Counter(self._grams(toks, n))
                total = sum(counts.values())
                v = {g: (c / total) * math.log(max(1.0, self.n_docs / max(1, self.df[n].get(g, 0))))
                     for g, c in counts.items()} if total else {}
                vecs.append(v)
                norms.append(math.sqrt(sum(x * x for x in v.values())))
            fr = [len(set(self._grams(toks, n))) / len(self._grams(toks, n))
                  for n in range(1, self.n_max + 1) if self._grams(toks, n)]
            rep = float(np.mean(fr)) if fr else 1.0
            self._vec[text] = (len(toks), vecs, norms, rep)
        return self._vec[text]

    def score(self, cand: str, refs: list[str]) -> float:
        refs = [r for r in refs if _clean_str(r)]
        L, cv, cn, rep = self._vectors(_clean_str(cand))
        if L == 0 or not refs:
            return 0.0
        n_scores = []
        for n in range(self.n_max):
            per_ref = []
            for r in refs:
                Lr, rv, rn, _ = self._vectors(r)
                if cn[n] == 0 or rn[n] == 0:
                    sim = 0.0
                else:
                    num = sum(min(v, rv[n].get(g, 0.0)) * rv[n].get(g, 0.0) for g, v in cv[n].items())
                    sim = num / (cn[n] * rn[n])
                per_ref.append(sim * math.exp(-((L - Lr) ** 2) / (2 * self.sigma ** 2)))
            n_scores.append(10.0 * float(np.mean(per_ref)))
        return float(np.mean(n_scores)) * rep


def cider_scores(d: EvalData, s: dict) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """→ ({'name:caption_pred_k': per-object score}, baseline, legacy_baseline)."""
    refs = d.objects[d.true_cols].to_numpy().tolist()
    cider = CiderR(refs, s["cider_n_max"], s["cider_sigma"])
    tc = d.true_cols
    legacy = np.array([cider.score(r[0], [r[1]]) for r in refs])
    loo = np.array([np.mean([cider.score(r[j], [r[k] for k in range(len(tc)) if k != j])
                             for j in range(len(tc))]) for r in refs])
    baseline = loo if s["cider_baseline"] == "symmetric" else legacy
    pred = {}
    for n, cols in d.pred_cols.items():
        for c in cols:
            caps = d.preds[n][c].tolist()
            pred[f"{n}:{c}"] = np.array([cider.score(p, r) for p, r in
                                         _progress(list(zip(caps, refs)), f"CIDEr-R {n}:{c}")])
    return pred, baseline, legacy


def bootstrap_gap(diffs: np.ndarray, n_boot: int, rng) -> np.ndarray:
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    return diffs[idx].mean(axis=1)


def cider_bootstrap(d: EvalData, s: dict, pred: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    """Paired bootstrap of mean(pred − baseline) per class and overall.
    Bonferroni over classes; p = two-sided bootstrap p-value. The RNG is
    re-seeded per group, as in the previous notebook, so CIs reproduce."""
    cls = d.objects["class"].to_numpy()
    classes = sorted(set(cls))
    a_b = s["alpha"] / max(len(classes), 1)
    rows = []
    for c in classes + ["ALL"]:
        m = np.ones(len(cls), bool) if c == "ALL" else cls == c
        diffs = pred[m] - baseline[m]
        boot = bootstrap_gap(diffs, s["n_bootstrap"], np.random.default_rng(s["bootstrap_seed"]))
        lo, hi = np.percentile(boot, [100 * s["alpha"] / 2, 100 * (1 - s["alpha"] / 2)])
        p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        rows.append({"class": c, "N": int(m.sum()), "pred": pred[m].mean(),
                     "baseline": baseline[m].mean(), "gap": diffs.mean(),
                     "ci_lo": lo, "ci_hi": hi, "p_boot": p,
                     "bonferroni_sig": (p < a_b) if c != "ALL" else None})
    t = pd.DataFrame(rows)
    body = t[t["class"] != "ALL"].sort_values("gap", ascending=False)
    t = pd.concat([body, t[t["class"] == "ALL"]], ignore_index=True)
    t.attrs["alpha_bonferroni"] = a_b
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Fracture type + human scores
# ──────────────────────────────────────────────────────────────────────────────

def fracture_type_table(d: EvalData) -> pd.DataFrame:
    """Accuracy of each prediction's BOUNDARY/INTERIOR label against the
    references, on objects where the references have a label and agree."""
    ref = d.objects[[c.replace("caption_true", "ftype_true") for c in d.true_cols]]
    ref = ref.replace("", np.nan)
    agree = ref.notna().all(axis=1) & (ref.nunique(axis=1) == 1)
    label = ref.iloc[:, 0]
    rows = []
    n_lab = int(ref.notna().all(axis=1).sum())
    for n, cols in d.pred_cols.items():
        for c in cols:
            fc = c.replace("caption_pred", "ftype_pred")
            pred = d.preds[n][fc].replace("", np.nan)
            m = agree & pred.notna()
            acc = float((pred[m] == label[m]).mean()) if m.any() else float("nan")
            boundary_share = float((pred == "BOUNDARY").mean())
            rows.append({"set": n, "caption": f"Prediction {int(c.rsplit('_', 1)[1]) + 1}",
                         "labelled_refs": n_lab, "refs_agree": int(agree.sum()),
                         "ref_agreement": float(agree.sum() / n_lab) if n_lab else float("nan"),
                         "pred_labelled": int(pred.notna().sum()), "accuracy": acc,
                         "pred_BOUNDARY_share": boundary_share,
                         "ref_BOUNDARY_share": float((label[agree] == "BOUNDARY").mean())
                         if agree.any() else float("nan")})
    return pd.DataFrame(rows)


def human_scores_table(s: dict, per_caption: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge caption_evaluator.ipynb scores with per-caption metrics.
    → (summary per set incl. Spearman ρ vs each metric, per-class means)."""
    rows, by_class = [], []
    for n, f in s["human_scores"].items():
        h = pd.read_csv(f)
        h.columns = [c.strip() for c in h.columns]
        if "human_reviewed" in h.columns:
            h = h[h["human_reviewed"].astype(str).str.upper() == "TRUE"]
        h = h.dropna(subset=["score"])
        h["path"] = h["path"].astype(str).str.strip()
        h["caption_idx"] = h["caption_idx"].astype(int)
        m = per_caption[per_caption["set"] == n].merge(
            h[["path", "caption_idx", "score"]], on=["path", "caption_idx"], how="inner")
        r = {"set": n, "file": f, "scored_captions": len(h), "matched": len(m),
             "mean_score": h["score"].mean(), "std_score": h["score"].std()}
        for metric in ("clipscore", "ref_text", "refclip", "cider_r"):
            if metric in m.columns and len(m) >= 3:
                r[f"spearman_{metric}"] = m["score"].corr(m[metric], method="spearman")
        rows.append(r)
        if len(m):
            g = m.groupby("class")["score"].agg(["mean", "count"]).reset_index()
            g.insert(0, "set", n)
            by_class.append(g)
    return pd.DataFrame(rows), (pd.concat(by_class, ignore_index=True) if by_class else pd.DataFrame())


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Results:
    settings: dict
    data: EvalData
    clip: pd.DataFrame
    r_precision_class: pd.DataFrame
    collapse: dict[str, pd.DataFrame]
    collapse_refs: pd.DataFrame
    cider: dict[str, pd.DataFrame]
    fracture: pd.DataFrame
    human: pd.DataFrame
    human_by_class: pd.DataFrame
    per_caption: pd.DataFrame
    headline: pd.DataFrame
    variants: pd.DataFrame
    n_with_image: int
    notes: list[str]


def run(s: dict, embedder=None, data: Optional[EvalData] = None) -> Results:
    """Compute everything. Pass `data` (from load_data) to skip re-loading."""
    d = data if data is not None else load_data(s)
    notes = list(d.notes)
    root = Path(s["image_root"])
    out = Path(s["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    p = s["primary_pred"]
    pcol = f"caption_pred_{p}"

    # ── images + embeddings ──────────────────────────────────────────────────
    imgs = [resolve_overview(r.path, r.images_used, root) for r in d.objects.itertuples()]
    valid = np.array([x is not None for x in imgs])
    if (~valid).any():
        notes.append(f"{(~valid).sum()} / {len(valid)} objects have no overview image — "
                     f"excluded from CLIP metrics only")
    if embedder is None:
        embedder = CLIPEmbedder(s["clip_model"], s["device"], s["img_batch_size"],
                                s["txt_batch_size"], out / ".clip_cache" if s["embed_cache"] else None)
    img_emb = embedder.images([x for x in imgs if x is not None])
    txt_emb = {c: embedder.texts(d.objects[c].tolist()) for c in d.true_cols}
    for n, cols in d.pred_cols.items():
        for c in cols:
            txt_emb[f"{n}:{c}"] = embedder.texts(d.preds[n][c].tolist())

    # ── metrics ──────────────────────────────────────────────────────────────
    clip, rp_class, per_cap = clip_metrics(d, s, img_emb, valid, txt_emb)
    collapse = {n: collapse_table(d, s, d.preds[n][pcol], txt_emb[f"{n}:{pcol}"][valid], valid)
                for n in d.preds}
    collapse_refs = collapse_table(d, s, d.objects[d.true_cols[0]], txt_emb[d.true_cols[0]][valid], valid)
    cider_pred, baseline, legacy = cider_scores(d, s)
    cider = {n: cider_bootstrap(d, s, cider_pred[f"{n}:{pcol}"], baseline) for n in d.preds}
    if s["cider_baseline"] == "symmetric":
        notes.append(f"CIDEr-R baseline = both reference directions (mean "
                     f"{baseline.mean():.4f}); legacy ref0→ref1 baseline = {legacy.mean():.4f}")
    cid_rows = []
    for key, sc in cider_pred.items():
        n, c = key.split(":")
        cid_rows.append(pd.DataFrame({"set": n, "path": d.objects["path"],
                                      "caption_idx": int(c.rsplit("_", 1)[1]), "cider_r": sc}))
    cid_long = pd.concat(cid_rows, ignore_index=True)
    per_cap = cid_long.merge(per_cap, on=["set", "path", "caption_idx"], how="left") \
        if len(per_cap) else cid_long
    per_cap = per_cap.merge(d.objects[["path", "class"]], on="path", how="left",
                            suffixes=("_x", "")).drop(columns=["class_x"], errors="ignore")
    fracture = fracture_type_table(d)
    human, human_cls = human_scores_table(s, per_cap) if s["human_scores"] else (pd.DataFrame(), pd.DataFrame())

    # ── headline (ablation-table row per prediction set) ─────────────────────
    k1 = s["r_precision_k"][0]
    rows = []
    ref_clip = clip[(clip["set"] == "references")].iloc[0]
    ca = collapse_refs.iloc[-1]
    rows.append({"set": "references (Ref 1)", "Unique": ca["unique_ratio"], "Top-1": ca["top1_share"],
                 "NearDup": ca.get(f"near_dup@{s['near_dup_threshold']}"),
                 "IntraSim": ca.get("intra_class_sim"), "ΔCIDEr-R": 0.0, "CIDEr-R": baseline.mean(),
                 "CLIPScore": ref_clip["CLIPScore"], "Ref-Text": ref_clip["Ref-Text"],
                 "RefCLIP": ref_clip["RefCLIP"], f"R@{k1}": ref_clip[f"R@{k1}"],
                 "FracAcc": np.nan, "ParseErr": 0.0, "ΔCIDEr-R CI": "–"})
    for n in d.preds:
        cl = collapse[n].iloc[-1]
        cr = cider[n].iloc[-1]
        cp = clip[(clip["set"] == n) & (clip["column"] == pcol)].iloc[0]
        fr = fracture[(fracture["set"] == n) & (fracture["caption"] == f"Prediction {p + 1}")]
        rows.append({"set": n, "Unique": cl["unique_ratio"], "Top-1": cl["top1_share"],
                     "NearDup": cl.get(f"near_dup@{s['near_dup_threshold']}"),
                     "IntraSim": cl.get("intra_class_sim"), "ΔCIDEr-R": cr["gap"],
                     "ΔCIDEr-R CI": f"[{cr['ci_lo']:.3f}, {cr['ci_hi']:.3f}]", "CIDEr-R": cr["pred"],
                     "CLIPScore": cp["CLIPScore"], "Ref-Text": cp["Ref-Text"],
                     "RefCLIP": cp["RefCLIP"], f"R@{k1}": cp[f"R@{k1}"],
                     "FracAcc": fr["accuracy"].iloc[0] if len(fr) else np.nan,
                     "ParseErr": float(d.preds[n][f"parse_err_{p}"].mean())})
    headline = pd.DataFrame(rows)
    order = ["set", "Unique", "Top-1", "NearDup", "IntraSim", "ΔCIDEr-R", "ΔCIDEr-R CI", "CIDEr-R",
             "CLIPScore", "Ref-Text", "RefCLIP", f"R@{k1}", "FracAcc", "ParseErr"]
    headline = headline[order]

    variants = variants_table(s)
    for w_ in variants.get("warning", pd.Series(dtype=object)).dropna():
        notes.append(f"Provenance: {w_}")
    return Results(s, d, clip, rp_class, collapse, collapse_refs, cider, fracture, human,
                   human_cls, per_cap, headline, variants, int(valid.sum()), notes)


# ──────────────────────────────────────────────────────────────────────────────
# Presentation + export
# ──────────────────────────────────────────────────────────────────────────────

def paper_clip_table(r: Results) -> pd.DataFrame:
    """tab:clip-metrics layout. Ref-Text/RefCLIP for references = baseline row
    (shown once, on Reference 1)."""
    k1 = r.settings["r_precision_k"][0]
    out = []
    for i, row in r.clip.iterrows():
        is_ref = row["set"] == "references"
        name = row["caption"] if is_ref else (row["caption"] if len(r.data.preds) == 1
                                              else f"{label(r.settings, row['set'])} · {row['caption']}")
        first_ref = is_ref and row["column"] == r.data.true_cols[0]
        out.append({"Caption": name,
                    "CLIPScore": f"{row['CLIPScore']:.3f} ± {row['CLIPScore_std']:.3f}",
                    "Ref-Text": f"{row['Ref-Text']:.3f}" if (first_ref or not is_ref) else "–",
                    "RefCLIP": f"{row['RefCLIP']:.3f}" if (first_ref or not is_ref) else "–",
                    f"R@{k1}": f"{row[f'R@{k1}']:.4f}",
                    **{f"R@{k}": f"{row[f'R@{k}']:.4f}" for k in r.settings["r_precision_k"][1:]}})
    return pd.DataFrame(out)


def _fmt(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    f = df.copy()
    for c in f.columns:
        if pd.api.types.is_float_dtype(f[c]):
            f[c] = f[c].map(lambda v: "–" if pd.isna(v) else f"{v:.{digits}f}")
    return f


def to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Minimal booktabs table (no jinja2 needed)."""
    def esc(v):
        return str(v).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_") \
            .replace("±", r"$\pm$").replace("Δ", r"$\Delta$").replace("#", r"\#")
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{esc(caption)}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines += [r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
              " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        if str(row.iloc[0]) == "ALL":
            lines.append(r"\midrule")
        lines.append(" & ".join(esc(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def summary_tables(r: Results) -> dict[str, pd.DataFrame]:
    """Ordered dict of display-ready tables."""
    s = r.settings
    p = s["primary_pred"] + 1
    head = r.headline.copy()
    head["set"] = head["set"].map(lambda n: label(s, n))
    t = {"headline": _fmt(head), "clip": paper_clip_table(r)}
    keep = ["class", "N", "unique_ratio", "top1_share", "duplicate_rate", "distinct_ratio",
            f"distinct_{s['distinct_n']}", "intra_class_sim", f"near_dup@{s['near_dup_threshold']}"]
    for n, c in r.collapse.items():
        t[f"collapse:{n}"] = _fmt(c[[k for k in keep if k in c.columns]])
    for n, c in r.cider.items():
        c = c.copy()
        c["bonferroni_sig"] = c["bonferroni_sig"].map({True: "yes", False: "no", None: "–"})
        t[f"cider:{n}"] = _fmt(c)
    if len(r.fracture):
        t["fracture"] = _fmt(r.fracture.assign(set=r.fracture["set"].map(lambda n: label(s, n))))
    if len(r.human):
        t["human"] = _fmt(r.human.drop(columns=["file"]).assign(set=r.human["set"].map(lambda n: label(s, n))))
    t.update({"_meta": pd.DataFrame({"primary": [f"Prediction {p}"]})})
    return t


TITLES = {
    "headline": "Headline — one row per prediction set (tab:conditioning_ablation layout)",
    "clip": "CLIP-based metrics (tab:clip-metrics)",
    "collapse": "Caption collapse per class",
    "cider": "CIDEr-R gap vs human baseline, paired bootstrap",
    "fracture": "Fracture-type accuracy vs references",
    "human": "Human scores (caption_evaluator.ipynb)",
}


def summary_text(r: Results) -> str:
    s, d = r.settings, r.data
    lines = ["=" * 78, f"CAPTION EVALUATION — {s['run_name']}", "=" * 78,
             f"References : {s['true_csv']}  ({len(d.objects)} objects evaluated, "
             f"{len(d.true_cols)} refs each, {r.n_with_image} with image)",
             *[f"Prediction : {c.prediction_set} [{label(s, c.prediction_set)}] = {c.file}  "
               f"(matched {c.matched}/{c.ref_objects}, "
               f"missing {c.missing}, {c.samples_per_object:g} samples/object)"
               for c in d.coverage.itertuples()],
             f"Primary    : Prediction {s['primary_pred'] + 1} (caption_idx {s['primary_pred']}) "
             f"for collapse / CIDEr-R",
             f"CLIPScore  : {'100·cos' if s['clipscore_report'] == 'cos100' else '100·w·cos'}"
             f"   chance R@1 = {r.clip.attrs['chance_R@1']:.5f}"]
    for n in r.notes:
        lines.append(f"Note       : {n}")
    tabs = summary_tables(r)
    for key, df in tabs.items():
        if key.startswith("_"):
            continue
        base, _, name = key.partition(":")
        title = TITLES[base] + (f" — {label(s, name)}" if name else "")
        lines += ["", title, "-" * len(title), df.to_string(index=False)]
        if base == "cider":
            lines.append(f"(Bonferroni α = {r.cider[name].attrs['alpha_bonferroni']:.5f})")
    v = r.variants.drop(columns=[c for c in ("csv", "notes", "warning") if c in r.variants.columns])
    v = v.dropna(axis=1, how="all")
    if v.shape[1] > 2:
        def cell(x):
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "–"
            if isinstance(x, float) and x.is_integer():
                return str(int(x))
            return str(x)
        lines += ["", "Prediction sets — provenance", "-" * 28,
                  v.astype(object).apply(lambda col: col.map(cell)).to_string(index=False)]
    return "\n".join(lines)


def save(r: Results) -> list[Path]:
    import yaml
    out = Path(r.settings["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def w(df, name, index=False):
        f = out / name
        df.to_csv(f, index=index)
        written.append(f)

    w(r.headline, "summary_headline.csv")
    w(r.clip, "clip_metrics.csv")
    w(r.r_precision_class, "clip_r_precision_per_class.csv")
    w(r.data.coverage, "coverage.csv")
    w(r.variants, "variants.csv")
    for n in r.collapse:
        w(r.collapse[n], f"collapse_{n}.csv")
        w(r.cider[n], f"cider_r_bootstrap_{n}.csv")
    w(r.collapse_refs, "collapse_references.csv")
    w(r.fracture, "fracture_type.csv")
    w(r.per_caption, "per_caption_metrics.csv")
    if len(r.human):
        w(r.human, "human_scores.csv")
        w(r.human_by_class, "human_scores_by_class.csv")

    tabs = summary_tables(r)
    tex = []
    for key, df in tabs.items():
        if key.startswith("_"):
            continue
        base, _, name = key.partition(":")
        tex.append(to_latex(df, TITLES[base] + (f" ({label(r.settings, name)})" if name else ""),
                            f"tab:{base}{'-' + name if name else ''}"))
    (out / "tables.tex").write_text("\n\n".join(tex) + "\n")
    (out / "summary.txt").write_text(summary_text(r) + "\n")
    meta = {k: v for k, v in r.settings.items()}
    (out / "eval_config_resolved.yml").write_text(yaml.safe_dump(meta, sort_keys=False))
    written += [out / "tables.tex", out / "summary.txt", out / "eval_config_resolved.yml"]
    if r.settings.get("config") and Path(r.settings["config"]).is_file():   # exact input file
        src = out / f"source_{Path(r.settings['config']).name}"
        src.write_text(Path(r.settings["config"]).read_text())
        written.append(src)
    return written


def worst_examples(r: Results, n: int, by: str = "refclip") -> pd.DataFrame:
    """Lowest-scoring primary predictions (per prediction set), with references."""
    p = r.settings["primary_pred"]
    pc = r.per_caption[(r.per_caption["caption_idx"] == p)].dropna(subset=[by])
    rows = []
    for name, g in pc.groupby("set"):
        for _, x in g.nsmallest(n, by).iterrows():
            i = int(np.flatnonzero(r.data.objects["path"].to_numpy() == x["path"])[0])
            rows.append({"set": name, "path": x["path"], "class": x["class"], by: x[by],
                         "prediction": r.data.preds[name].loc[i, f"caption_pred_{p}"],
                         **{f"ref_{j+1}": r.data.objects.loc[i, c] for j, c in enumerate(r.data.true_cols)},
                         "images_used": r.data.objects.loc[i, "images_used"]})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Caption evaluation from the run YAML")
    ap.add_argument("--config", required=True, help="configs/eval.yml, or a run config such as configs/mae_v3.yml")
    ap.add_argument("--pred", action="append", default=[], metavar="NAME=CSV",
                    help="add (or replace) a prediction set on top of the config's")
    ap.add_argument("--only", action="append", default=[], metavar="NAME",
                    help="evaluate only these prediction sets")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an eval setting, value parsed as YAML (e.g. n_bootstrap=2000)")
    a = ap.parse_args(argv)
    import yaml
    ov = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        ov[k.strip()] = yaml.safe_load(v)
    if a.pred or a.only:
        entries = prediction_entries(load_settings(a.config, ov))
        entries.update({k: {"csv": v} for k, v in (x.split("=", 1) for x in a.pred)})
        if a.only:
            unknown = set(a.only) - set(entries)
            if unknown:
                ap.error(f"--only: unknown prediction set(s) {sorted(unknown)}; have {list(entries)}")
            entries = {k: v for k, v in entries.items() if k in a.only}
        ov["predictions"] = entries
    s = load_settings(a.config, ov)
    log.info("Eval settings:\n" + describe_settings(s))
    r = run(s)
    print(summary_text(r))
    for f in save(r):
        log.info(f"wrote {f}")


if __name__ == "__main__":
    main()
