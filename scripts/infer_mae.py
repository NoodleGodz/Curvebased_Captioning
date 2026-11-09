"""
infer_mae.py
============
Inference for the MAE Curvature-Token-Encoder + LoRA Qwen2.5-VL fracture
captioner.

The model code (GeometryProjector, prompts, target schema, prefix pooling) is
IMPORTED from train_mae.py, so training and inference cannot drift apart.

Usage
-----
  # Everything from the run config (test_csv, ckpt, geometry, generation)
  python scripts/infer_mae.py --config configs/mae_v3.yml

  # No --config: settings are read from <ckpt>/run_config.yml (saved by training)
  python infer_mae.py --ckpt ./checkpoints_mae_v3/ckpt_best \
      --csv caption_breakingbad_test_new.csv --root . --out preds.csv

  # Single OBJ
  python scripts/infer_mae.py --config configs/mae_v3.yml --obj path/to/piece_0.obj

Before generating, the geometry/encoder settings are compared with the
checkpoint's run_config.yml; any mismatch stops the run
(override with --allow_config_mismatch).

Output CSV: one row per (object, sample) with caption_idx = 0..num_samples-1,
which caption_evaluation.ipynb pivots into caption_pred_0, caption_pred_1, ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_config import (  # noqa: E402
    SAVED_CONFIG_NAME, check_consistency, log_resolved, parse_with_config,
)
from train_mae import (  # noqa: E402
    CURV_CHANNEL_MAP, CURV_CHANNEL_NAMES, DEFAULT_TARGET_FIELDS, GeometryProjector,
    _get_embed_tokens, build_chat_text, load_projector_state, pool_geo_tokens,
    pos_enc_dim_for,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inference — MAE GeometryProjector + Qwen2.5-VL LoRA fracture captioner")

    # Checkpoint / model
    p.add_argument("--ckpt", required=True, help="Checkpoint dir (e.g. checkpoints/ckpt_best)")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--load_4bit", action="store_true", default=False,
                   help="Load the base model in NF4 (as in training). Default bf16.")
    p.add_argument("--no_4bit", dest="load_4bit", action="store_false")
    p.add_argument("--flash_attn", action="store_true", default=False)
    p.add_argument("--device", default="cuda")

    # Input — single file OR batch CSV
    p.add_argument("--obj",  default=None, help="Single OBJ file path")
    p.add_argument("--cls",  default="unknown", help="Object class (single-file mode, logging only)")
    p.add_argument("--csv",  default=None, help="CSV with columns: path, class")
    p.add_argument("--root", default=".", help="Dataset root prepended to CSV paths")
    p.add_argument("--out",  default=None, help="Output CSV (batch mode)")

    # Geometry + encoder — must match training (checked against ckpt config)
    p.add_argument("--n_fps",            type=int,   default=512)
    p.add_argument("--disk_grid",        type=int,   default=16)
    p.add_argument("--disk_radius_frac", type=float, default=0.07)
    p.add_argument("--disk_fill",        default="rbf", choices=["rbf", "nearest", "zero"])
    p.add_argument("--curv_knn",         type=int,   default=30)
    p.add_argument("--fourier_bands",    type=int,   default=0)
    p.add_argument("--curv_channels",    type=int,   default=3, choices=[2, 3, 4, 6])
    p.add_argument("--use_tangent_disk", action="store_false", default=True,
                   help="PASSING this flag turns tangent-disk mode OFF (point mode). "
                        "In YAML use `use_tangent_disk: false`.")
    p.add_argument("--proj_hidden",      type=int,   default=256)
    p.add_argument("--mae_enc_layers",   type=int,   default=4)
    p.add_argument("--mae_dec_layers",   type=int,   default=2)
    p.add_argument("--mae_n_heads",      type=int,   default=4)
    p.add_argument("--mae_chunk_size",   type=int,   default=128,
                   help="VRAM knob only — may differ from training")
    p.add_argument("--max_geo_tokens",   type=int,   default=512)
    p.add_argument("--target_fields", nargs="*", default=DEFAULT_TARGET_FIELDS,
                   help="Fields written to the output CSV (should match training)")

    # Generation
    p.add_argument("--max_new_tokens", type=int,   default=512)
    p.add_argument("--do_sample",      action="store_true", default=False)
    p.add_argument("--greedy",         dest="do_sample", action="store_false")
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--top_p",          type=float, default=0.9)
    p.add_argument("--num_samples",    type=int,   default=1,
                   help="Captions per object (caption_idx 0..n-1); >1 needs do_sample")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--allow_config_mismatch", action="store_true", default=False)
    return p


def parse_args(argv=None) -> argparse.Namespace:
    args = parse_with_config(build_parser(), "infer", argv)
    if args.num_samples > 1 and not args.do_sample:
        raise SystemExit("[error] num_samples > 1 with greedy decoding gives identical "
                         "captions — set do_sample: true (or --do_sample).")
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(args):
    """Base Qwen2.5-VL + LoRA adapter + GeometryProjector from args.ckpt."""
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    ckpt_dir    = Path(args.ckpt)
    adapter_dir = ckpt_dir / "lora_adapter"
    proj_path   = ckpt_dir / "geometry_projector.pt"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"No lora_adapter dir at {adapter_dir}")
    if not proj_path.exists():
        raise FileNotFoundError(f"No geometry_projector.pt at {proj_path}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    attn_impl = "sdpa"
    if args.flash_attn:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            log.warning("flash-attn not installed — falling back to SDPA")

    bnb_cfg = None
    if args.load_4bit:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.bfloat16)
    log.info(f"Loading base model {args.model_id} ({'NF4' if args.load_4bit else 'bf16'}, {attn_impl})")
    base_llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, quantization_config=bnb_cfg,
        torch_dtype=None if args.load_4bit else torch.bfloat16,
        device_map=args.device, trust_remote_code=True, attn_implementation=attn_impl,
    )
    llm = PeftModel.from_pretrained(base_llm, adapter_dir)
    llm.config.use_cache = True
    llm.eval()

    projector = GeometryProjector(
        c_in=args.curv_channels, d_grid=args.disk_grid,
        d_model=llm.config.text_config.hidden_size,
        pos_enc_dim=pos_enc_dim_for(args.fourier_bands), hidden=args.proj_hidden,
        n_heads=args.mae_n_heads, enc_layers=args.mae_enc_layers,
        dec_layers=args.mae_dec_layers, mask_ratio=0.0, dropout=0.0,
        chunk_size=args.mae_chunk_size,
    ).to(args.device)
    load_projector_state(projector, proj_path, args.device)
    projector.eval()
    log.info(f"Projector params: {sum(p.numel() for p in projector.parameters()):,}")
    return llm, projector, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Geometry + generation
# ──────────────────────────────────────────────────────────────────────────────

def process_single_obj(obj_path, args, curv_ch_idx):
    """Same DatasetConfig as training → reuses the training .pt cache."""
    from BreakingBadDataset import DatasetConfig, process_mesh
    cfg = DatasetConfig(
        n_fps=args.n_fps, disk_grid=args.disk_grid,
        disk_radius_frac=args.disk_radius_frac, disk_channels=6,
        disk_fill=args.disk_fill, curv_knn=args.curv_knn, cache_processed=True,
        fourier_bands=args.fourier_bands, use_tangent_disk=args.use_tangent_disk,
    )
    t = process_mesh(Path(obj_path), cfg)
    patches = t["data"][:, curv_ch_idx, :, :].unsqueeze(0).to(args.device)
    pos_enc = t["pos_enc"].unsqueeze(0).to(args.device)
    return patches, pos_enc


@torch.no_grad()
def generate_captions(llm, projector, tokenizer, patches, pos_enc, args) -> list[str]:
    """Geometry prefix + prompt → num_samples generations."""
    geo_embeds  = projector(patches, pos_enc)
    prompt_ids  = tokenizer(build_chat_text("", tokenizer), add_special_tokens=False,
                            return_tensors="pt").input_ids.to(args.device)
    text_embeds = _get_embed_tokens(llm)(prompt_ids)
    geo_embeds  = pool_geo_tokens(geo_embeds.to(text_embeds.dtype), args.max_geo_tokens)
    full_embeds = torch.cat([geo_embeds, text_embeds], dim=1)

    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                  eos_token_id=tokenizer.eos_token_id)
    if args.do_sample:
        gen_kw.update(temperature=args.temperature, top_p=args.top_p,
                      num_return_sequences=args.num_samples)
    out_ids = llm.generate(
        inputs_embeds=full_embeds,
        attention_mask=torch.ones(1, full_embeds.shape[1], dtype=torch.long, device=args.device),
        **gen_kw,
    )
    return [tokenizer.decode(o, skip_special_tokens=True).strip() for o in out_ids]


def parse_output(raw: str) -> dict:
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return {"_parse_error": True}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    log_resolved(args)

    # ── Train/infer consistency check ────────────────────────────────────────
    saved = Path(args.ckpt) / SAVED_CONFIG_NAME
    if saved.is_file():
        problems = check_consistency(args, saved)
        if problems:
            msg = "Settings differ from the checkpoint's training config:\n  - " + \
                  "\n  - ".join(problems)
            if args.allow_config_mismatch:
                log.warning(msg + "\n(continuing: --allow_config_mismatch)")
            else:
                raise SystemExit(f"[error] {msg}\nFix the config, or pass "
                                 f"--allow_config_mismatch if this is intentional.")
        log.info(f"Config consistent with {saved}")
    else:
        log.warning(f"{saved} not found (checkpoint predates run configs) — cannot "
                    f"verify that geometry/encoder settings match training.")

    curv_ch_idx = CURV_CHANNEL_MAP[args.curv_channels]
    log.info(f"Curvature channels: {[CURV_CHANNEL_NAMES[i] for i in curv_ch_idx]}")
    torch.manual_seed(args.seed)

    # ── Jobs ─────────────────────────────────────────────────────────────────
    jobs: list[tuple[str, str]] = []
    if args.obj:
        jobs.append((args.obj, args.cls))
    elif args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv, sep=",", dtype=str)
        df.columns = df.columns.str.strip()
        if "path" not in df.columns or "class" not in df.columns:
            sys.exit("[error] CSV must have 'path' and 'class' columns")
        df = df.drop_duplicates("path")       # GT CSVs have one row per caption
        root = Path(args.root)
        jobs = [(str(root / r["path"].strip()), r["class"].strip()) for _, r in df.iterrows()]
        log.info(f"Batch mode: {len(jobs)} objects × {args.num_samples} sample(s)")
    else:
        sys.exit("[error] Provide --obj or --csv (or data.test_csv in the config)")

    llm, projector, tokenizer = load_model(args)

    rows = []
    for i, (obj_path, obj_class) in enumerate(jobs):
        log.info(f"[{i+1}/{len(jobs)}] {obj_path}  class={obj_class}")
        try:
            patches, pos_enc = process_single_obj(obj_path, args, curv_ch_idx)
            raws = generate_captions(llm, projector, tokenizer, patches, pos_enc, args)
        except Exception as e:
            log.error(f"  failed: {e}")
            raws = [f"<error: {e}>"] * args.num_samples
        for k, raw in enumerate(raws):
            parsed = parse_output(raw)
            row = {"path": obj_path, "caption_idx": k, "true_class": obj_class}
            for f in args.target_fields:
                row[f] = parsed.get(f, "")
            row["parse_error"] = bool(parsed.get("_parse_error", False))
            row["raw_output"]  = raw
            rows.append(row)
            print(f"  [{k}] {row.get('caption', '') or raw[:200]}")

    if args.out and rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.out, index=False)
        log.info(f"Saved {len(rows)} predictions → {args.out}")
    log.info("Done.")


if __name__ == "__main__":
    main()
