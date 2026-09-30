"""
train_qwen_image_lora.py
=========================
Image-conditioned ABLATION baseline for the geometry-conditioned fracture
captioning pipeline (compare against train_mae.py, which injects a
MAE-compressed geometry-token prefix instead of real pixels).

This script fine-tunes Qwen2.5-VL-7B-Instruct with LoRA using its NATIVE
vision encoder on the actual rendered pair images — the same images produced
by the rendering pipeline and consumed by caption_fractures.py to generate
the ground-truth captions in the first place. No geometry extraction, no
GeometryProjector: images in, JSON out.

Held fixed vs. train_mae.py for a fair ablation
------------------------------------------------
  * Same 4-field JSON target schema (object_description, confidence,
    fracture_type, caption) — see TARGET_FIELDS_ORDER.
  * Same class-blind convention: the object's class name is withheld from
    the prompt by default (pass --include_class to disable this and study
    the effect separately).
  * Same LoRA recipe: 4-bit QLoRA, attention + MLP target modules
    (q/k/v/o_proj, gate/up/down_proj) — this is what fixed caption collapse
    in the geometry version (see learnings.md) and is applied here too so
    the comparison isn't confounded by a weaker LoRA config.

What's genuinely different
---------------------------
  * No BreakingBadDataset / no OBJ meshes / no curvature / no
    GeometryProjector are used at all.
  * The model's own vision tower does the encoding; we just build a normal
    multi-image chat turn via Qwen2.5-VL's AutoProcessor and let
    `pixel_values` / `image_grid_thw` flow through the model natively.
  * Train/val split is done at the OBJECT level (grouped by `path`) so the
    two caption_idx variants of the same object never straddle the split —
    mirrors the 80/20 object-level split described in main.tex.

Input CSV — output of caption_fractures.py
--------------------------------------------
  path                OBJ path (used only as an object-id / dedup key)
  class               object category (withheld from the prompt by default)
  caption_idx         which sampled caption variant this row is
  images_used         JSON list of image paths used to generate this row
                       (an overview pair render + several fracture-focused
                       pair renders — LEFT intact / RIGHT fractured with the
                       fracture surface highlighted in orange)
  object_description  target field
  fracture_type       target field (BOUNDARY | INTERIOR)
  confidence          target field (low | medium | high)
  caption              target field
  parse_error         True/False — rows with parse_error=True are skipped
  raw_output, elapsed_s  unused

Usage
-----
  python train_qwen_image_lora.py \
      --csv  captions.csv \
      --root /path/to/breaking_bad \
      --out  ./checkpoints_image_ablation

  # smaller footprint / fewer images per sample
  python train_qwen_image_lora.py \
      --csv captions.csv --root . --out ./ckpt \
      --max_images 4 --batch 1 --grad_accum 16

Dependencies
------------
  pip install torch transformers peft bitsandbytes pillow pandas numpy qwen-vl-utils
  pip install wandb   # optional
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Image-only ablation — LoRA fine-tune Qwen2.5-VL on rendered "
                     "fracture pair images (no geometry tokens)."
    )

    # Data
    p.add_argument("--csv",  required=True, help="captions.csv from caption_fractures.py")
    p.add_argument("--root", default=".",   help="Root dir images_used paths are relative to")
    p.add_argument("--out",  default="./checkpoints_image_ablation", help="Output directory")
    p.add_argument("--val_split", type=float, default=0.05,
                   help="Fraction of OBJECTS (grouped by `path`) held out for val")
    p.add_argument("--classes", nargs="*", default=None, help="Subset of object classes")

    # Images
    p.add_argument("--max_images", type=int, default=6,
                   help="Cap on images fed per sample (images_used is truncated to this "
                        "many, overview image kept first if present). Default 6 matches "
                        "caption_fractures.py's own --max_images default.")
    p.add_argument("--image_width",  type=int, default=560,
                   help="PIL thumbnail max width for each image before the processor's "
                        "own smart-resize (which snaps to the 28px patch grid).")
    p.add_argument("--image_height", type=int, default=280,
                   help="PIL thumbnail max height (pair renders are ~2:1, wide).")

    # Prompt
    p.add_argument("--include_class", action="store_true", default=False,
                   help="Include the object class name in the prompt. Default OFF "
                        "(class-blind) to match the geometry-token ablation in "
                        "train_mae.py — the model must describe the fracture from "
                        "the images alone, not from a class-name prior.")

    # Model
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--load_4bit", action="store_true", default=True,
                   help="Load base model in 4-bit (QLoRA) — saves ~12 GB VRAM")
    p.add_argument("--no_4bit",   dest="load_4bit", action="store_false")
    p.add_argument("--flash_attn", action="store_true", default=False,
                   help="Enable Flash Attention 2 (requires flash-attn installed). "
                        "Falls back to SDPA otherwise.")

    # LoRA
    p.add_argument("--lora_r",      type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--lora_dropout",type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="*",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
                   help="Default = attention + MLP, matching the fix for caption "
                        "collapse used in train_mae.py. Pass just "
                        "'q_proj k_proj v_proj o_proj' for attention-only.")

    # Training
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch",       type=int,   default=1,
                   help="Per-device batch size. Multi-image batches >1 are supported "
                        "(pixel_values/image_grid_thw are concatenated across the "
                        "batch, matching Qwen2.5-VL's native flattened packing), but "
                        "VRAM grows fast with --max_images — keep batch=1 unless "
                        "--max_images is small.")
    p.add_argument("--grad_accum",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--warmup_ratio",type=float, default=0.03)
    p.add_argument("--max_target_len", type=int, default=256,
                   help="Max token length of the JSON target (truncated beyond this — "
                        "should rarely trigger given the prompt's word limits).")
    p.add_argument("--seed",        type=int,   default=42)

    # Resume
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint dir to resume from (e.g. ckpt_step200). "
                        "Restores LoRA weights, optimizer, scheduler, global_step.")

    # Memory
    p.add_argument("--dataloader_workers", type=int, default=0)

    # Logging
    p.add_argument("--wandb",  default=None, help="W&B project name. Omit to disable.")
    p.add_argument("--log_steps",  type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Target JSON schema — identical to the reduced 4-field schema in train_mae.py
# ──────────────────────────────────────────────────────────────────────────────

TARGET_FIELDS_ORDER = ["object_description", "confidence", "fracture_type", "caption"]


def build_target_json(row: pd.Series) -> str:
    obj = {f: (row.get(f) or "") for f in TARGET_FIELDS_ORDER}
    return json.dumps(obj, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_MSG = """\
You are a precise 3D fracture analyst.

You will be shown rendered image views of a fractured 3D object. Each image is \
a side-by-side pair of renders of the same object:
- LEFT: the complete, intact object
- RIGHT: the larger remaining body after a fragment broke off, with the \
fracture surface highlighted in orange

If multiple images are shown, they are different viewpoints of the same pair. \
Use all views together before answering.

Return exactly one valid JSON object with no additional text, markdown, or code blocks.

The JSON must have exactly this structure:

{
  "object_description": "Shape, structure, proportions, and notable features of the intact object (LEFT). One or two sentences, maximum 30 words.",
  "confidence": "low | medium | high, based on image clarity and viewpoint coverage.",
  "fracture_type": "BOUNDARY or INTERIOR, plus a brief mechanism (e.g. impact, clean snap, diagonal shear). Max 15 words.",
  "caption": "One concise sentence describing the object and the location/nature of its damage, asserting only what is clearly visible. Max 25 words."
}

Rules:
- Exactly one sentence per field (two only for object_description), within the stated word limit.
- No hedging language ("appears to", "seems", "possibly") unless confidence is low.
- No repetition of content across fields.
- Never reference image annotations, highlights, colors, overlays, or other rendering artifacts — describe damage purely in geometric/structural terms.
- Describe all locations relative to the OBJECT'S OWN PARTS AND ANATOMY (e.g. "near the spout", "at the base", "along the handle", "on the rim", "at the foot", "near the shoulder seam") — NEVER relative to the image or viewpoint (e.g. never "upper left", "top right", "center of the frame").
- If the object class has no obvious named parts, use structural terms instead (e.g. "near the corner", "along one edge", "on the flat face").
- Use concrete structural vocabulary (edge, lip, border, perimeter, opening, corner, face, shoulder, junction) rather than vague terms.
"""

USER_MSG_CLASS_BLIND = """\
Classify the break as one of:
- BOUNDARY: the missing fragment was attached along an outer edge, tip, handle, or other protruding feature
- INTERIOR: the missing fragment came from a flat or recessed surface, not an edge or protrusion

Return the required JSON object.
"""

USER_MSG_WITH_CLASS_TMPL = """\
Object class: {class_name}

Classify the break as one of:
- BOUNDARY: the missing fragment was attached along an outer edge, tip, handle, or other protruding feature
- INTERIOR: the missing fragment came from a flat or recessed surface, not an edge or protrusion

Return the required JSON object.
"""


def build_user_text(class_name: Optional[str]) -> str:
    if class_name:
        return USER_MSG_WITH_CLASS_TMPL.format(class_name=class_name)
    return USER_MSG_CLASS_BLIND


# ──────────────────────────────────────────────────────────────────────────────
# CSV / image loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataframe(csv_path: str, classes: Optional[list[str]]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()

    required = {"path", "class", "images_used", "object_description",
                "fracture_type", "confidence", "caption"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[error] Input CSV missing columns: {missing}. Found: {list(df.columns)}")

    n_before = len(df)
    if "parse_error" in df.columns:
        df = df[df["parse_error"].str.upper().str.strip() != "TRUE"]
        n_skip = n_before - len(df)
        if n_skip:
            log.warning(f"Skipped {n_skip} rows with parse_error=TRUE")

    if classes is not None:
        df = df[df["class"].isin(classes)]

    df = df.reset_index(drop=True)
    log.info(f"Loaded {len(df)} caption rows over {df['path'].nunique()} distinct objects")
    return df


def split_by_object(df: pd.DataFrame, val_split: float, seed: int):
    """Object-level split: no object's `path` contributes rows to both sets."""
    paths = df["path"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_split))
    val_paths = set(paths[:n_val])
    val_df   = df[df["path"].isin(val_paths)].reset_index(drop=True)
    train_df = df[~df["path"].isin(val_paths)].reset_index(drop=True)
    log.info(f"Object-level split: {len(paths)-len(val_paths)} train objects "
             f"({len(train_df)} rows), {len(val_paths)} val objects ({len(val_df)} rows)")
    return train_df, val_df


def resolve_images(images_used_raw: str, root: Path, max_images: Optional[int]) -> list[Path]:
    try:
        imgs = ast.literal_eval(images_used_raw)
        if not isinstance(imgs, list):
            imgs = []
    except Exception:
        imgs = [p.strip().strip("'\"[]") for p in str(images_used_raw).split(",") if p.strip()]

    # Overview image first if present — matches caption_fractures.py's convention
    overview = [p for p in imgs if "overview" in Path(p).stem.lower()]
    others   = [p for p in imgs if "overview" not in Path(p).stem.lower()]
    ordered  = overview + others

    resolved = [root / p for p in ordered]
    resolved = [p for p in resolved if p.is_file()]

    if max_images is not None and len(resolved) > max_images:
        resolved = resolved[:max_images]

    return resolved


def load_and_resize(path: Path, max_size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(max_size, Image.LANCZOS)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class FractureImageDataset(Dataset):
    """
    Returns raw PIL images + text fields per sample. Tokenisation and the
    multimodal processor call happen later in collate (see build_sample_tensors)
    so that the (potentially large) HF processor call runs in the training loop
    rather than in DataLoader worker processes.
    """

    def __init__(self, df: pd.DataFrame, root: str | Path,
                 image_max_size: tuple[int, int], max_images: Optional[int]):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.image_max_size = image_max_size
        self.max_images = max_images

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_paths = resolve_images(row["images_used"], self.root, self.max_images)
        if not img_paths:
            raise FileNotFoundError(f"No valid images_used files for row {idx} "
                                     f"({row['path']})")
        images = [load_and_resize(p, self.image_max_size) for p in img_paths]
        return {
            "images": images,
            "class_name": row["class"],
            "target_json": build_target_json(row),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample tensor building + collate
# ──────────────────────────────────────────────────────────────────────────────

def build_sample_tensors(sample: dict, processor, include_class: bool,
                          max_target_len: int) -> dict:
    """
    Builds one sample's (input_ids, attention_mask, labels, pixel_values,
    image_grid_thw) by running the HF processor twice on the SAME images:
    once on the prompt only (to measure its expanded token length, since
    image placeholder tokens expand based on actual patch count) and once on
    the full prompt+target conversation (to get the tensors actually used for
    training). Labels are -100 over the prompt prefix, real target ids after.
    """
    from qwen_vl_utils import process_vision_info

    images = sample["images"]
    class_name = sample["class_name"] if include_class else None
    user_text = build_user_text(class_name)

    user_content = [{"type": "image", "image": im} for im in images]
    user_content.append({"type": "text", "text": user_text})

    prompt_messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_content},
    ]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": sample["target_json"]}
    ]

    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False,
    )

    image_inputs, video_inputs = process_vision_info(prompt_messages)

    prompt_enc = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        return_tensors="pt",
    )
    full_enc = processor(
        text=[full_text], images=image_inputs, videos=video_inputs,
        return_tensors="pt",
    )

    prompt_len = prompt_enc.input_ids.shape[1]
    input_ids  = full_enc.input_ids[0]
    attn_mask  = full_enc.attention_mask[0]
    labels     = input_ids.clone()
    labels[:prompt_len] = -100

    if labels.shape[0] - prompt_len > max_target_len:
        cut = prompt_len + max_target_len
        input_ids = input_ids[:cut]
        attn_mask = attn_mask[:cut]
        labels    = labels[:cut]

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
        "pixel_values": full_enc.pixel_values,
        "image_grid_thw": full_enc.image_grid_thw,
    }


def make_collate_fn(processor, include_class: bool, max_target_len: int, pad_token_id: int):
    """
    Pads text tensors to the batch max length and concatenates pixel_values /
    image_grid_thw across the batch in sample order — this is Qwen2.5-VL's
    native flattened multi-image packing (the model matches consumed patches
    against <|image_pad|> tokens found per-sample, in order, from this flat
    pool), so no per-sample segmentation of pixel_values is needed.
    """
    def collate(raw_batch: list[dict]) -> dict:
        tensorised = [
            build_sample_tensors(s, processor, include_class, max_target_len)
            for s in raw_batch
        ]

        max_len = max(t["input_ids"].shape[0] for t in tensorised)
        B = len(tensorised)

        input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels    = torch.full((B, max_len), -100, dtype=torch.long)

        for i, t in enumerate(tensorised):
            L = t["input_ids"].shape[0]
            input_ids[i, :L] = t["input_ids"]
            attn_mask[i, :L] = t["attention_mask"]
            labels[i, :L]    = t["labels"]

        pixel_values   = torch.cat([t["pixel_values"]   for t in tensorised], dim=0)
        image_grid_thw = torch.cat([t["image_grid_thw"] for t in tensorised], dim=0)

        return {
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "labels":         labels,
            "pixel_values":   pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    return collate


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(llm, processor, out_dir: Path, tag: str):
    ckpt_dir    = out_dir / f"ckpt_{tag}"
    adapter_dir = ckpt_dir / "lora_adapter"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")
    _prev_hf = os.environ.get("HF_DATASETS_OFFLINE")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"]  = "1"
    try:
        llm.save_pretrained(adapter_dir, safe_serialization=True)
        processor.save_pretrained(adapter_dir)
    finally:
        if _prev_tf is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = _prev_tf
        if _prev_hf is None:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
        else:
            os.environ["HF_DATASETS_OFFLINE"] = _prev_hf

    log.info(f"Checkpoint saved → {ckpt_dir}")


def save_training_state(optimizer, scheduler, global_step: int, epoch: int,
                        best_val_loss: float, out_dir: Path, tag: str):
    ckpt_dir = out_dir / f"ckpt_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        },
        ckpt_dir / "training_state.pt",
    )
    log.info(f"Training state saved → {ckpt_dir / 'training_state.pt'}")


def load_checkpoint_for_resume(ckpt_dir: str | Path, llm, optimizer, scheduler,
                               device: str) -> tuple[int, int, float]:
    from peft import set_peft_model_state_dict

    ckpt_dir    = Path(ckpt_dir)
    adapter_dir = ckpt_dir / "lora_adapter"
    state_path  = ckpt_dir / "training_state.pt"

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    if adapter_dir.exists():
        st_path  = adapter_dir / "adapter_model.safetensors"
        bin_path = adapter_dir / "adapter_model.bin"
        if st_path.exists():
            from safetensors.torch import load_file as st_load
            adapters = st_load(str(st_path), device=device)
        elif bin_path.exists():
            adapters = torch.load(str(bin_path), map_location=device, weights_only=True)
        else:
            adapters = None
            log.warning(f"No adapter weights found in {adapter_dir}")
        if adapters is not None:
            set_peft_model_state_dict(llm, adapters)
            log.info(f"Loaded LoRA adapter from {adapter_dir}")
    else:
        log.warning(f"No lora_adapter dir in {ckpt_dir} — starting LoRA from scratch")

    global_step, start_epoch, best_val_loss = 0, 0, float("inf")
    if state_path.exists():
        state = torch.load(str(state_path), map_location=device, weights_only=False)
        try:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        except (ValueError, RuntimeError, KeyError) as e:
            log.warning(f"Could not restore optimizer/scheduler state ({e}) — "
                        f"starting fresh optimizer state; LoRA weights above are "
                        f"still correctly restored.")
        global_step   = state["global_step"]
        start_epoch   = state["epoch"]
        best_val_loss = state["best_val_loss"]
        log.info(f"Resumed step={global_step} epoch={start_epoch} "
                 f"best_val_loss={best_val_loss:.4f}")
    else:
        log.warning(f"No training_state.pt in {ckpt_dir} — counters reset to zero")

    return global_step, start_epoch, best_val_loss


# ──────────────────────────────────────────────────────────────────────────────
# Eval — see evaluate_simple() below main(), used by the training loop
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    use_wandb = args.wandb is not None
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb, config=vars(args))
        except ImportError:
            log.warning("wandb not installed — logging disabled")
            use_wandb = False

    # ── Data ─────────────────────────────────────────────────────────────────
    df = load_dataframe(args.csv, args.classes)
    train_df, val_df = split_by_object(df, args.val_split, args.seed)

    image_max_size = (args.image_width, args.image_height)
    train_ds = FractureImageDataset(train_df, args.root, image_max_size, args.max_images)
    val_ds   = FractureImageDataset(val_df,   args.root, image_max_size, args.max_images)

    log.info(f"Prompt is class-{'aware' if args.include_class else 'blind'} "
             f"(--include_class={args.include_class})")
    log.info(f"Images: up to {args.max_images} per sample, "
             f"thumbnail max size {image_max_size}")

    # ── Model + processor ───────────────────────────────────────────────────
    from transformers import AutoProcessor, BitsAndBytesConfig
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType

    log.info(f"Loading processor + {args.model_id} ...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    bnb_cfg = None
    if args.load_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if args.flash_attn:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
            log.warning("flash-attn not installed — falling back to SDPA")
    else:
        attn_impl = "sdpa"

    llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16 if not args.load_4bit else None,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    llm.config.use_cache = False
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    log.info(f"LoRA target_modules: {args.lora_target_modules}")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
    )
    llm = get_peft_model(llm, lora_cfg)
    llm.print_trainable_parameters()

    # ── Dataloaders ──────────────────────────────────────────────────────────
    collate_fn = make_collate_fn(processor, args.include_class, args.max_target_len, pad_token_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.dataloader_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.dataloader_workers, collate_fn=collate_fn,
    )

    # ── Optimiser ────────────────────────────────────────────────────────────
    trainable_params = [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)

    total_steps = (len(train_ds) // (args.batch * args.grad_accum)) * args.epochs
    total_steps = max(total_steps, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=args.warmup_ratio,
    )
    log.info(f"Total optimizer steps: {total_steps}")

    global_step, start_epoch, best_val_loss = 0, 0, float("inf")
    if args.resume:
        global_step, start_epoch, best_val_loss = load_checkpoint_for_resume(
            args.resume, llm, optimizer, scheduler, str(device),
        )

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        llm.train()
        accum_loss = 0.0
        accum_count = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
            except Exception as e:
                log.warning(f"Skipping batch (device move error): {e}")
                continue

            outputs = llm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            accum_count += 1
            del outputs, batch

            if (step + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                if global_step < total_steps - 1:
                    scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                global_step += 1

                avg_loss = accum_loss / max(accum_count, 1)
                accum_loss = accum_count = 0

                if global_step % args.log_steps == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    log.info(f"Epoch {epoch+1}  step {global_step}  "
                             f"loss={avg_loss:.4f}  grad_norm={float(grad_norm):.3f}  "
                             f"lr={lr_now:.2e}")
                    if use_wandb:
                        wandb.log({
                            "loss/train": avg_loss,
                            "grad/norm": float(grad_norm),
                            "lr": lr_now,
                            "step": global_step,
                        })

                if global_step % args.eval_steps == 0:
                    val = evaluate_simple(llm, val_loader, tokenizer)
                    log.info(f"  → val_loss={val['loss']:.4f}  "
                             f"json_parse={val['json_parse_rate']*100:.1f}%")
                    if use_wandb:
                        wandb.log({
                            "loss/val": val["loss"],
                            "val/json_parse_rate": val["json_parse_rate"],
                            "step": global_step,
                        })
                    if val["loss"] < best_val_loss:
                        best_val_loss = val["loss"]
                        save_checkpoint(llm, processor, out_dir, tag="best")
                        save_training_state(optimizer, scheduler, global_step,
                                            epoch, best_val_loss, out_dir, tag="best")
                        log.info(f"  → saved best (val_loss={best_val_loss:.4f})")

                if global_step % args.save_steps == 0:
                    save_checkpoint(llm, processor, out_dir, tag=f"step{global_step}")
                    save_training_state(optimizer, scheduler, global_step,
                                        epoch, best_val_loss, out_dir, tag=f"step{global_step}")

        save_checkpoint(llm, processor, out_dir, tag=f"epoch{epoch+1}")
        save_training_state(optimizer, scheduler, global_step,
                            epoch + 1, best_val_loss, out_dir, tag=f"epoch{epoch+1}")

    log.info("Training complete.")
    if use_wandb:
        wandb.finish()


@torch.no_grad()
def evaluate_simple(llm, loader, tokenizer) -> dict:
    import json as _json
    llm.eval()
    total_loss, total_n = 0.0, 0
    parse_ok = parse_total = 0

    for batch in loader:
        batch = {k: v.to(llm.device) for k, v in batch.items()}
        outputs = llm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            labels=batch["labels"],
        )
        b = batch["input_ids"].shape[0]
        total_loss += outputs.loss.item() * b
        total_n += b

        for i in range(b):
            tgt_ids = batch["labels"][i][batch["labels"][i] != -100]
            if len(tgt_ids) == 0:
                continue
            try:
                decoded = tokenizer.decode(tgt_ids.cpu(), skip_special_tokens=True)
                _json.loads(decoded.strip())
                parse_ok += 1
            except Exception:
                pass
            parse_total += 1

    llm.train()
    n = max(total_n, 1)
    return {
        "loss": total_loss / n,
        "json_parse_rate": parse_ok / max(parse_total, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper (run after training)
# ──────────────────────────────────────────────────────────────────────────────

def load_for_inference(ckpt_dir: str | Path,
                       model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                       device: str = "cuda"):
    """
    >>> llm, processor = load_for_inference("checkpoints_image_ablation/ckpt_best")
    >>> # build messages with build_user_text(None) + PIL images, then llm.generate(...)
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    ckpt_dir = Path(ckpt_dir)
    adapter_dir = ckpt_dir / "lora_adapter"

    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    llm = PeftModel.from_pretrained(base, adapter_dir)
    llm.config.use_cache = True
    llm.eval()
    return llm, processor


if __name__ == "__main__":
    main()