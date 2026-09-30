"""
infer_mae_image_only.py
========================
Inference script for the image-only ablation checkpoint produced by
train_qwen_image_only.py (native Qwen2.5-VL vision encoder + LoRA, no
geometry tokens — compare against infer_mae.py, which loads the
MAE-GeometryProjector + LoRA checkpoint from train_mae.py instead).

Mirrors infer_mae.py's CLI shape (single item / batch CSV, --ckpt, --out) but
swaps the geometry pipeline (BreakingBadDataset / GeometryProjector /
embeds-prefix injection) for the real image pipeline: PIL images →
Qwen2.5-VL's own AutoProcessor → pixel_values / image_grid_thw → llm.generate().

To guarantee the inference prompt is byte-for-byte identical to what the
checkpoint was trained on, this script imports SYSTEM_MSG, build_user_text,
resolve_images, load_and_resize, and TARGET_FIELDS_ORDER directly from
train_qwen_image_only.py rather than re-declaring them — if you edit the
prompt there, this script picks up the change automatically instead of
silently drifting out of sync (train_qwen_image_only.py's own
`if __name__ == "__main__"` guard makes this import side-effect-free).

Usage
-----
  # Single object — explicit image paths
  python infer_mae_image_only.py \
      --ckpt ./checkpoints_image_ablation/ckpt_best \
      --images render_overview.png render_view1.png render_view2.png \
      --cls teapot

  # Batch from CSV (same columns as train_qwen_image_only.py's input:
  # path, class, images_used — ground-truth caption columns are optional
  # and ignored if present)
  python infer_mae_image_only.py \
      --ckpt ./checkpoints_image_ablation/ckpt_best \
      --csv  caption_breakingbad_train_new.csv --root . \
      --out  predictions_image_ablation.csv

Must match the training run for correct behaviour
---------------------------------------------------
  * --include_class must match what the checkpoint was trained with. The
    example training command (train_qwen_image_only.py ... --lora_r 32)
    did NOT pass --include_class, so the checkpoint is class-blind — leave
    --include_class off here too, or the model will see a class name it
    never saw during training.
  * --max_images / --image_width / --image_height should match training
    defaults (6 / 560 / 280) unless you specifically want to test
    off-distribution image counts or resolutions.

Notes
-----
  * lora_r, lora_alpha, etc. do NOT need to be passed here — PeftModel
    reads them from the saved adapter_config.json automatically.
  * use_cache is re-enabled at inference (train_qwen_image_only.py disables
    it for gradient checkpointing during training).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference — image-only Qwen2.5-VL LoRA fracture captioner "
                    "(native vision encoder, no geometry tokens)."
    )

    # Checkpoint
    p.add_argument("--ckpt", default="./checkpoints_image_ablation/ckpt_best",
                   help="Checkpoint dir (e.g. checkpoints_image_ablation/ckpt_best)")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="Base model ID (must match training)")

    # Input — single item OR batch CSV
    p.add_argument("--images", nargs="*", default=None,
                   help="Single-item mode: explicit list of pair-render image paths "
                        "(overview image auto-detected/ordered first if its filename "
                        "contains 'overview', matching resolve_images() convention)")
    p.add_argument("--cls", default="unknown",
                   help="Object class for single-item mode (only used in the prompt "
                        "if --include_class is also passed)")
    p.add_argument("--csv", default=None,
                   help="CSV with columns: path, class, images_used (batch mode). "
                        "Same format as train_qwen_image_only.py's input CSV — "
                        "ground-truth caption columns are ignored if present.")
    p.add_argument("--root", default=".",
                   help="Root dir images_used paths are relative to (batch mode)")
    p.add_argument("--classes", nargs="*", default=None,
                   help="Batch mode: restrict to these object classes")

    # Output
    p.add_argument("--out", default=None,
                   help="Output CSV path (batch mode). Omit to print to stdout.")

    # Images — should match training unless deliberately testing off-distribution
    p.add_argument("--max_images", type=int, default=6,
                   help="Must match training --max_images for on-distribution inference")
    p.add_argument("--image_width", type=int, default=560,
                   help="Must match training --image_width")
    p.add_argument("--image_height", type=int, default=280,
                   help="Must match training --image_height")

    # Prompt
    p.add_argument("--include_class", action="store_true", default=False,
                   help="Include the object class name in the prompt. Must match "
                        "whether the checkpoint was trained with --include_class — "
                        "default OFF (class-blind), matching train_qwen_image_only.py's "
                        "own default.")

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=300,
                   help="Should be >= the --max_target_len used in training (default "
                        "256 there) so the JSON target isn't cut off.")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Only used when --do_sample is passed")
    p.add_argument("--do_sample", action="store_true", default=False,
                   help="Sample instead of greedy decoding. Off by default for "
                        "deterministic structured output.")
    p.add_argument("--flash_attn", action="store_true", default=False,
                   help="Enable Flash Attention 2 for the LLM. "
                        "Requires: pip install flash-attn --no-build-isolation")
    p.add_argument("--device", default="cuda")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading — mirrors train_qwen_image_only.py's own load_for_inference(),
# with flash-attn selection added (matching infer_mae.py's pattern)
# ──────────────────────────────────────────────────────────────────────────────

def load_model(args):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    ckpt_dir    = Path(args.ckpt)
    adapter_dir = ckpt_dir / "lora_adapter"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"No lora_adapter dir at {adapter_dir}")

    log.info(f"Loading processor from {adapter_dir}")
    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.flash_attn:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            log.info("Flash Attention 2 enabled for LLM")
        except ImportError:
            attn_impl = "sdpa"
            log.warning("flash-attn not installed — falling back to SDPA")
    else:
        attn_impl = "sdpa"
        log.info("LLM attention: SDPA (PyTorch built-in Flash Attention)")

    log.info(f"Loading base model {args.model_id}")
    base_llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype         = torch.bfloat16,
        device_map           = args.device,
        trust_remote_code    = True,
        attn_implementation  = attn_impl,
    )

    log.info(f"Loading LoRA adapter from {adapter_dir}")
    llm = PeftModel.from_pretrained(base_llm, adapter_dir)

    # Re-enable KV cache — disabled during training for gradient checkpointing
    llm.config.use_cache = True
    llm.eval()

    log.info("Model ready.")
    return llm, processor


# ──────────────────────────────────────────────────────────────────────────────
# Batch CSV loading — slim version of train_qwen_image_only.load_dataframe()
# that does NOT require the ground-truth target columns (a fresh test-set CSV
# may only have path/class/images_used, with no captions yet)
# ──────────────────────────────────────────────────────────────────────────────

def load_batch_csv(csv_path: str, classes: Optional[list[str]]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()

    required = {"path", "class", "images_used"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[error] Input CSV missing columns: {missing}. Found: {list(df.columns)}")

    if "parse_error" in df.columns:
        n_before = len(df)
        df = df[df["parse_error"].str.upper().str.strip() != "TRUE"]
        n_skip = n_before - len(df)
        if n_skip:
            log.warning(f"Skipped {n_skip} rows with parse_error=TRUE")

    if classes is not None:
        df = df[df["class"].isin(classes)]

    df = df.reset_index(drop=True)
    log.info(f"Loaded {len(df)} rows over {df['path'].nunique()} distinct objects")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Generation — same processor call + trimming pattern as caption_fractures.py's
# run_inference(), adapted to the checkpoint's LoRA-wrapped model and the
# class-blind/class-aware prompt from train_qwen_image_only.py
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_caption(llm, processor, images, class_name, args, SYSTEM_MSG, build_user_text) -> str:
    from qwen_vl_utils import process_vision_info

    user_text = build_user_text(class_name)
    user_content = [{"type": "image", "image": im} for im in images]
    user_content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_content},
    ]

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        return_tensors="pt",
    ).to(llm.device)

    gen_kwargs = dict(
        max_new_tokens = args.max_new_tokens,
        do_sample      = args.do_sample,
        pad_token_id   = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        eos_token_id   = processor.tokenizer.eos_token_id,
    )
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature

    generated_ids = llm.generate(**inputs, **gen_kwargs)

    # Strip the input tokens from the output — same trimming as caption_fractures.py
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# ──────────────────────────────────────────────────────────────────────────────
# JSON parsing + pretty print — same shape as infer_mae.py
# ──────────────────────────────────────────────────────────────────────────────

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
    return {"_parse_error": True, "_raw": raw}


def pretty_print(result: dict, obj_path: str, target_fields: list[str]):
    print(f"\n{'─'*60}")
    print(f"  OBJ: {obj_path}")
    print(f"{'─'*60}")
    if result.get("_parse_error"):
        print(f"  [PARSE ERROR] Raw:\n{result.get('_raw', '')}")
        return
    for field in target_fields:
        print(f"  {field:<24} {result.get(field, '—')}")
    print()


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

    # Import the training script's own prompt + image-loading code so the
    # inference prompt is guaranteed identical to what the checkpoint saw.
    sys.path.insert(0, str(Path(__file__).parent))
    from train_qwen_image_only import (
        SYSTEM_MSG, build_user_text, resolve_images, load_and_resize,
        TARGET_FIELDS_ORDER,
    )

    log.info(
        f"Prompt is class-{'aware' if args.include_class else 'blind'} "
        f"(--include_class={args.include_class}) — must match the training run"
    )
    log.info(
        f"Images: up to {args.max_images} per sample, "
        f"thumbnail max size ({args.image_width}, {args.image_height})"
    )

    # ── Load model ────────────────────────────────────────────────────────────
    llm, processor = load_model(args)

    image_max_size = (args.image_width, args.image_height)

    # ── Build job list ────────────────────────────────────────────────────────
    # Each job: (obj_id, class_name, list[PIL.Image])
    jobs: list[tuple[str, str, list]] = []

    if args.images:
        imgs = [load_and_resize(Path(p), image_max_size) for p in args.images]
        jobs.append((",".join(args.images), args.cls, imgs))
    elif args.csv:
        df = load_batch_csv(args.csv, args.classes)
        root = Path(args.root)
        for _, row in df.iterrows():
            img_paths = resolve_images(row["images_used"], root, args.max_images)
            if not img_paths:
                log.warning(f"  No valid images_used files for {row['path']} — skipping")
                continue
            imgs = [load_and_resize(p, image_max_size) for p in img_paths]
            jobs.append((row["path"], row["class"], imgs))
        log.info(f"Batch mode: {len(jobs)} objects with valid images")
    else:
        sys.exit("[error] Provide either --images or --csv")

    # ── Run inference ─────────────────────────────────────────────────────────
    results = []

    for i, (obj_path, obj_class, images) in enumerate(jobs):
        log.info(f"[{i+1}/{len(jobs)}] {obj_path}  class={obj_class}  "
                 f"n_images={len(images)}")

        class_for_prompt = obj_class if args.include_class else None

        try:
            raw    = generate_caption(
                llm, processor, images, class_for_prompt, args,
                SYSTEM_MSG, build_user_text,
            )
            parsed = parse_output(raw)
        except Exception as e:
            log.error(f"  Generation failed: {e}")
            results.append({"path": obj_path, "class": obj_class,
                             "_parse_error": True, "_raw": str(e)})
            continue

        parsed["path"]  = obj_path
        parsed["class"] = obj_class
        results.append(parsed)
        pretty_print(parsed, obj_path, TARGET_FIELDS_ORDER)

    # ── Save results ──────────────────────────────────────────────────────────
    if args.out and results:
        rows = []
        for r in results:
            row = {"path": r.get("path", ""), "class": r.get("class", "")}
            for field in TARGET_FIELDS_ORDER:
                row[field] = r.get(field, "")
            row["parse_error"] = r.get("_parse_error", False)
            row["raw_output"]  = r.get("_raw", "")
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        log.info(f"Saved {len(rows)} predictions → {args.out}")

    log.info("Done.")


if __name__ == "__main__":
    main()