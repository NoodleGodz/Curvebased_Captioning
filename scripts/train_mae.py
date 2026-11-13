"""
train_mae.py
============
LoRA SFT fine-tuning of Qwen2.5-VL-7B-Instruct on the Breaking Bad fracture
dataset. Curvature-disk tokens from the MAE-style Curvature Token Encoder
(GeometryProjector) are projected into the LLM embedding space and prepended
as a prefix before the structured JSON target.

Architecture
------------
  MeshData.data  [N, C_in, D, D]          # C_in ∈ {2,3,4,6} — see CURV_CHANNEL_MAP
      ↓  GeometryProjector (masked Transformer encoder, mean-pool, project)
  geo_embeds     [N, d_model]              # prefix tokens (avg-pooled to max_geo_tokens)
      ↓  concat with text embeds
  full_seq       [N + T, d_model]
      ↓  Qwen2.5-VL-7B (LoRA)
  next-token loss over JSON target only    # prefix + prompt masked in loss

Usage
-----
  # recommended — everything from one YAML shared with check_cache / infer
  python scripts/train_mae.py --config configs/mae_v3.yml

  # CLI flags still override the YAML
  python scripts/train_mae.py --config configs/mae_v3.yml --epochs 1 --wandb null

  # no YAML: all flags on the command line (legacy)
  python train_mae.py --csv captions.csv --root . --out ./checkpoints ...

The fully resolved config is written to <out>/run_config.yml and into every
checkpoint dir; infer_mae.py checks against it.

Changes vs. the 2026-11-03 version
----------------------------------
  * --config YAML support (run_config.py); resolved config saved with ckpts.
  * MAE fix: decoder reconstructs masked CURVATURE cells from the ENCODED
    visible tokens, so L_mae trains the encoder (was detached pre-encoder
    embeddings + pos_enc target → encoder got no MAE gradient).
  * Previously hard-coded values are now flags: disk_radius_frac, disk_fill,
    curv_knn, fourier_bands (pos_enc_dim derived), proj_dropout, div_lam,
    weight_decay, grad_clip, target_fields, val_split_by.
  * --val_split_by object: groups rows by source mesh (no caption leakage).
  * LR schedule (--lr_schedule phase2): LoRA gets its own one-cycle that
    starts when it is unfrozen; before, OneCycleLR spent LoRA's warmup (and
    part of its decay) during the frozen phase 1. Optimizer holds only LoRA
    tensors, not all base-model params.
  * Resume: LR is computed from global_step (was restarting the schedule).
  * Fixed: samples/sec was always 0; phase label off-by-one; evaluate()
    used the default div weight.
  * Removed stale load_for_inference() (wrong defaults) — use infer_mae.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).parent))
from run_config import (  # noqa: E402
    SAVED_CONFIG_NAME, dump_resolved, log_resolved, parse_with_config,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
DEFAULT_TARGET_FIELDS = ["class", "object_description", "fracture_type",
                         "confidence", "caption"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA SFT — Qwen2.5-VL on Breaking Bad")

    # Data
    p.add_argument("--csv",   required=True,  help="Training captions CSV")
    p.add_argument("--root",  required=True,  help="Dataset root (OBJ files live here)")
    p.add_argument("--out",   default="./checkpoints", help="Output directory")
    p.add_argument("--val_split", type=float, default=0.05, help="Fraction held out for val")
    p.add_argument("--val_split_by", default="row", choices=["row", "object"],
                   help="row = legacy random_split over CSV rows (reproduces "
                        "old val sets; both captions of an object can straddle "
                        "train/val). object = every row of one source mesh "
                        "goes to the same split.")
    p.add_argument("--classes", nargs="*", default=None, help="Subset of object classes")
    p.add_argument("--seed", type=int, default=42)

    # Geometry (DatasetConfig) — must match check_cache.py and infer_mae.py
    p.add_argument("--n_fps",            type=int,   default=512)
    p.add_argument("--disk_grid",        type=int,   default=16)
    p.add_argument("--disk_radius_frac", type=float, default=0.07,
                   help="Disk radius as fraction of the bounding-sphere diameter")
    p.add_argument("--disk_fill",        default="rbf", choices=["rbf", "nearest", "zero"])
    p.add_argument("--curv_knn",         type=int,   default=30)
    p.add_argument("--fourier_bands",    type=int,   default=0,
                   help="0 = raw [xyz|normal] pos_enc (6 dims); >0 adds 6*bands dims")
    p.add_argument("--curv_channels",    type=int,   default=3, choices=[2, 3, 4, 6],
                   help="Curvature channels kept from the 6 available — see CURV_CHANNEL_MAP")
    p.add_argument("--use_tangent_disk", action="store_false", default=True,
                   help="Tangent-disk mode is ON by default; PASSING THIS FLAG "
                        "TURNS IT OFF (point-mode ablation, disk_grid forced to 1). "
                        "In YAML use `use_tangent_disk: false` instead.")

    # Curvature Token Encoder
    p.add_argument("--proj_hidden",    type=int,   default=256)
    p.add_argument("--mae_enc_layers", type=int,   default=4)
    p.add_argument("--mae_dec_layers", type=int,   default=2)
    p.add_argument("--mae_n_heads",    type=int,   default=4)
    p.add_argument("--mae_mask_ratio", type=float, default=0.5)
    p.add_argument("--mae_lam",        type=float, default=0.1)
    p.add_argument("--mae_chunk_size", type=int,   default=128,
                   help="Disks per encoder chunk (VRAM knob)")
    p.add_argument("--proj_dropout",   type=float, default=0.1)
    p.add_argument("--max_geo_tokens", type=int,   default=512,
                   help="Geometry prefix length fed to the LLM; if < n_fps the "
                        "tokens are adaptive-avg-pooled down")

    # Model
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--load_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit",   dest="load_4bit", action="store_false")
    p.add_argument("--flash_attn", action="store_true", default=False)

    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="*", default=DEFAULT_LORA_TARGETS)

    # Target schema
    p.add_argument("--target_fields", nargs="*", default=DEFAULT_TARGET_FIELDS,
                   help="Ordered JSON keys of the training target. 'caption' "
                        "comes from the caption column; others from CSV columns "
                        "('class' is always written as \"\").")

    # Training
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--freeze_lm_steps", type=int, default=300,
                   help="Stage 1: LoRA frozen, CTE trains alone for this many steps")
    p.add_argument("--refreeze_on_new_lora_modules", action="store_true", default=True)
    p.add_argument("--no_refreeze_on_new_lora_modules", dest="refreeze_on_new_lora_modules",
                   action="store_false")
    p.add_argument("--batch",        type=int,   default=1)
    p.add_argument("--grad_accum",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-4, help="LoRA LR")
    p.add_argument("--proj_lr",      type=float, default=2e-4, help="CTE LR")
    p.add_argument("--warmup_ratio", type=float, default=0.03,
                   help="Warmup fraction of each group's own schedule span")
    p.add_argument("--lr_schedule", default="phase2", choices=["phase2", "global"],
                   help="phase2 = LoRA's one-cycle (warmup + decay) starts when it is "
                        "unfrozen at freeze_lm_steps; projector cycles over the whole "
                        "run. global = legacy: one OneCycleLR for both groups from step "
                        "0, so LoRA's warmup is spent while frozen (v1–v3 behaviour).")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip",    type=float, default=1.0,
                   help="Max grad-norm, clipped separately for CTE and LoRA")
    p.add_argument("--div_lam",      type=float, default=0.05, help="Diversity-loss weight")
    p.add_argument("--max_seq_len",  type=int,   default=512)

    # Resume
    p.add_argument("--resume", default=None, help="Checkpoint dir to resume from")

    # Memory
    p.add_argument("--dataloader_workers", type=int, default=0)

    # Logging
    p.add_argument("--wandb", default=None, help="W&B project; omit or 'null' to disable")
    p.add_argument("--log_steps",   type=int, default=10)
    p.add_argument("--save_steps",  type=int, default=200)
    p.add_argument("--eval_steps",  type=int, default=200)
    p.add_argument("--log_target_samples", type=int, default=3)

    # Sanity-check generation
    p.add_argument("--sanity_check_every_save", action="store_true", default=True)
    p.add_argument("--no_sanity_check_every_save", dest="sanity_check_every_save",
                   action="store_false")
    p.add_argument("--sanity_sample_idx", type=int, default=0)
    p.add_argument("--sanity_max_new_tokens", type=int, default=200)
    return p


def parse_args(argv=None) -> argparse.Namespace:
    args = parse_with_config(build_parser(), "train", argv)
    if isinstance(args.wandb, str) and args.wandb.lower() in ("null", "none", ""):
        args.wandb = None
    return args


def pos_enc_dim_for(fourier_bands: int) -> int:
    """Must match BreakingBadDataset.positional_encoding."""
    return 6 + (2 * fourier_bands * 3 if fourier_bands > 0 else 0)


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────

class PhasedOneCycle:
    """
    One-cycle LR (same curve as torch OneCycleLR: cosine warmup from
    max_lr/div_factor to max_lr over pct_start of the span, then cosine decay to
    max_lr/div_factor/final_div_factor), but each param group runs its cycle
    over its OWN span [start_g, total_steps). Before start_g the group's LR is 0.

    Used so the LoRA group warms up when it is unfrozen (phase 2) instead of
    spending its warmup and part of its decay while frozen in phase 1.
    With every start = 0 it reproduces torch.optim.lr_scheduler.OneCycleLR.

    LR is a pure function of the step counter, so resuming only needs
    `step=global_step` — nothing to fast-forward.
    """

    def __init__(self, optimizer, max_lrs, total_steps, starts, pct_start=0.3,
                 div_factor=25.0, final_div_factor=1e4, step=0):
        assert len(max_lrs) == len(starts) == len(optimizer.param_groups)
        self.optimizer = optimizer
        self.max_lrs = [float(x) for x in max_lrs]
        self.total_steps = int(total_steps)
        self.starts = [int(s) for s in starts]
        self.pct_start = float(pct_start)
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)
        self.step_count = int(step)
        self._apply()

    @staticmethod
    def _cos(start, end, pct):
        import math
        return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1)

    def _span(self, g):
        span = self.total_steps - self.starts[g]
        warm_end = float(self.pct_start * span) - 1      # as in OneCycleLR
        return span, warm_end

    def peak_step(self, g) -> int:
        """Absolute step at which group g reaches its max LR."""
        return self.starts[g] + max(0, int(round(self._span(g)[1])))

    def lr_at(self, g: int, step: int) -> float:
        max_lr = self.max_lrs[g]
        initial = max_lr / self.div_factor
        minimum = initial / self.final_div_factor
        if step < self.starts[g]:
            return 0.0
        span, warm_end = self._span(g)
        if span < 2:
            return max_lr
        t = min(step - self.starts[g], span - 1)
        end = span - 1
        if warm_end <= 0:                                # degenerate: no warmup
            return self._cos(max_lr, minimum, t / end)
        if t <= warm_end:
            return self._cos(initial, max_lr, t / warm_end)
        return self._cos(max_lr, minimum, (t - warm_end) / (end - warm_end))

    def _apply(self):
        for g, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self.lr_at(g, self.step_count)

    def step(self):
        self.step_count += 1
        self._apply()

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {k: getattr(self, k) for k in ("max_lrs", "total_steps", "starts",
                                              "pct_start", "div_factor",
                                              "final_div_factor", "step_count")}

    def load_state_dict(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self._apply()


# ──────────────────────────────────────────────────────────────────────────────
# Curvature channel selection
# ──────────────────────────────────────────────────────────────────────────────

# Order in MeshData.data / .curv (BreakingBadDataset.estimate_curvature):
#   0: k1 (larger |k|)  1: k2  2: mean  3: gaussian  4: curvedness  5: shape_index
# NOTE: names below kept as "kmin/kmax" for log compatibility, but the dataset
# orders by |k|, not by value.
CURV_CHANNEL_MAP = {
    2: [0, 1],              # k1, k2
    3: [1, 4, 5],           # k2, curvedness, shape_index  (NOT mean curvature)
    4: [0, 1, 2, 3],        # k1, k2, mean, gaussian
    6: [0, 1, 2, 3, 4, 5],  # all six
}

CURV_CHANNEL_NAMES = {
    0: "kmin", 1: "kmax", 2: "mean_curv",
    3: "gauss_curv", 4: "curvedness", 5: "shape_index",
}


# ──────────────────────────────────────────────────────────────────────────────
# Geometry projector — MAE (Masked Autoencoder) style Curvature Token Encoder
# ──────────────────────────────────────────────────────────────────────────────
#
#  Input (B, N, C_in, D, D). Each disk is a sequence of S = D*D cell tokens.
#  Training: mask a fraction of cells, encode visible cells, mean-pool →
#  project → one LLM token per disk. The decoder reconstructs the masked cells'
#  curvature from the ENCODED visible tokens (L_mae trains the encoder).
#  Inference: no masking, decoder unused.
# ──────────────────────────────────────────────────────────────────────────────


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block, attention via SDPA (Flash kernel on CUDA)."""

    def __init__(self, hidden: int, n_heads: int, mlp_ratio: float = 2.0,
                 dropout: float = 0.0):
        super().__init__()
        assert hidden % n_heads == 0, f"hidden={hidden} must be divisible by n_heads={n_heads}"
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.dropout  = dropout

        self.norm1 = nn.LayerNorm(hidden)
        self.qkv   = nn.Linear(hidden, hidden * 3, bias=False)
        self.out   = nn.Linear(hidden, hidden,     bias=False)
        self.norm2 = nn.LayerNorm(hidden)
        dim_ff = int(hidden * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, dim_ff, bias=False),
            nn.GELU(),
            nn.Linear(dim_ff, hidden, bias=False),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        nh, hd  = self.n_heads, self.head_dim
        h   = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        dp = self.dropout if self.training else 0.0
        h  = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dp, is_causal=False,
        )
        h = h.permute(0, 2, 1, 3).reshape(B, S, H)
        x = x + self.drop(self.out(h))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class GeometryProjector(nn.Module):
    """
    MAE-style Curvature Token Encoder.

    c_in        : curvature channels (2/3/4/6)
    d_grid      : disk size D (spatial grid D×D)
    d_model     : LLM hidden size (3584 for Qwen2.5-VL-7B)
    pos_enc_dim : positional encoding dim (6 for fourier_bands=0)
    mask_ratio  : fraction of cells masked during training (0 at inference)
    chunk_size  : disks encoded at once (bounds attention memory)
    """

    def __init__(
        self,
        c_in: int,
        d_grid: int,
        d_model: int,
        pos_enc_dim: int = 6,
        hidden: int = 256,
        n_heads: int = 4,
        enc_layers: int = 4,
        dec_layers: int = 2,
        mask_ratio: float = 0.5,
        dropout: float = 0.1,
        chunk_size: int = 128,
    ):
        super().__init__()
        self.c_in        = c_in
        self.d_grid      = d_grid
        self.pos_enc_dim = pos_enc_dim
        self.hidden      = hidden
        self.mask_ratio  = mask_ratio
        self.chunk_size  = chunk_size
        self.n_spatial   = d_grid * d_grid

        self.patch_embed = nn.Linear(c_in + pos_enc_dim, hidden, bias=False)
        self.spatial_pos = nn.Parameter(torch.zeros(1, self.n_spatial, hidden))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.encoder = nn.ModuleList([
            _TransformerBlock(hidden, n_heads, dropout=dropout) for _ in range(enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(hidden)

        # Decoder — training only
        self.dec_embed = nn.Linear(hidden, hidden, bias=False)
        self.decoder   = nn.ModuleList([
            _TransformerBlock(hidden, n_heads, dropout=dropout) for _ in range(dec_layers)
        ])
        self.dec_norm  = nn.LayerNorm(hidden)
        # Reconstruct the CURVATURE channels at masked cells. pos_enc is
        # broadcast identically to every cell, so predicting it is trivial.
        self.dec_pred  = nn.Linear(hidden, c_in, bias=True)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

    def _spatial_tokens(self, patches: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        """patches (BN, C, S), pos_enc (BN, P) → (BN, S, hidden)."""
        BN, C, S = patches.shape
        x  = patches.permute(0, 2, 1)
        pe = pos_enc.unsqueeze(1).expand(-1, S, -1)
        x  = self.patch_embed(torch.cat([x, pe], dim=-1))
        return x + self.spatial_pos.reshape(1, S, self.hidden)

    def _random_mask(self, x: torch.Tensor, ratio: float):
        """Mask `ratio` of cells independently per disk → (x_vis, ids_keep, ids_mask)."""
        BN, S, H = x.shape
        n_keep = S - int(S * ratio)
        ids_shuf = torch.rand(BN, S, device=x.device).argsort(dim=1)
        ids_keep = ids_shuf[:, :n_keep]
        ids_mask = ids_shuf[:, n_keep:]
        x_vis = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, H))
        return x_vis, ids_keep, ids_mask

    def _decode_chunk(self, enc_vis, ids_keep, ids_mask) -> torch.Tensor:
        """MAE decoder for one chunk. enc_vis = ENCODED visible tokens (C, S_vis, H).
        Returns predicted curvature at masked cells (C, S_mask, c_in)."""
        C_ch, _, H = enc_vis.shape
        n_mask = ids_mask.shape[1]
        vis_dec  = self.dec_embed(enc_vis)
        mask_tok = self.mask_token.to(vis_dec.dtype).expand(C_ch, n_mask, H)
        full     = torch.cat([vis_dec, mask_tok], dim=1)                   # shuffled order
        ids_restore = torch.cat([ids_keep, ids_mask], dim=1).argsort(dim=1)
        full = full.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, H))  # original order
        full = full + self.spatial_pos
        for blk in self.decoder:
            full = checkpoint(blk, full, use_reentrant=False)
        full = self.dec_norm(full)
        pred = full.gather(1, ids_mask.unsqueeze(-1).expand(-1, -1, H))
        return self.dec_pred(pred)

    def encode_train(self, patches: torch.Tensor, pos_enc: torch.Tensor,
                     compute_mae: bool = True):
        """
        Training forward: mask → encode visible → pool → project, plus MAE
        reconstruction of masked curvature from the ENCODED visible tokens.
        Returns geo_embeds (B, N, d_model), unweighted mae_loss scalar
        (0 when disabled or nothing is masked).
        """
        B, N, C, D, _ = patches.shape
        BN, S = B * N, D * D
        x   = patches.reshape(BN, C, S)
        pe  = pos_enc.reshape(BN, self.pos_enc_dim)
        tok = self._spatial_tokens(x, pe)
        tok, ids_keep, ids_mask = self._random_mask(tok, self.mask_ratio)
        compute_mae = compute_mae and ids_mask.shape[1] > 0
        target = x.permute(0, 2, 1) if compute_mae else None               # (BN, S, C)

        pooled, sq_sum, n_el = [], patches.new_zeros((), dtype=torch.float32), 0
        for start in range(0, BN, self.chunk_size):
            end = min(start + self.chunk_size, BN)
            h = tok[start:end]
            for blk in self.encoder:
                h = checkpoint(blk, h, use_reentrant=False)
            h = self.enc_norm(h)
            pooled.append(h.mean(dim=1))
            if compute_mae:
                im   = ids_mask[start:end]
                pred = self._decode_chunk(h, ids_keep[start:end], im)
                tgt  = target[start:end].gather(1, im.unsqueeze(-1).expand(-1, -1, C))
                sq_sum = sq_sum + (pred.float() - tgt.float()).pow(2).sum()
                n_el  += tgt.numel()

        geo = self.out_proj(torch.cat(pooled, dim=0)).reshape(B, N, -1)
        mae = sq_sum / n_el if compute_mae else sq_sum
        return geo, mae

    def forward(self, patches: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        """Inference forward — no masking, decoder unused. → (B, N, d_model)."""
        B, N, C, D, _ = patches.shape
        BN = B * N
        tok = self._spatial_tokens(patches.reshape(BN, C, D * D),
                                   pos_enc.reshape(BN, self.pos_enc_dim))
        out = []
        for start in range(0, BN, self.chunk_size):
            h = tok[start:start + self.chunk_size]
            for blk in self.encoder:
                h = checkpoint(blk, h, use_reentrant=False) if torch.is_grad_enabled() else blk(h)
            out.append(self.out_proj(self.enc_norm(h).mean(dim=1)))
        return torch.cat(out, dim=0).reshape(B, N, -1)


def load_projector_state(projector: GeometryProjector, path, device) -> list[str]:
    """
    Load a projector state_dict, tolerating decoder weights saved before the
    MAE fix (dec_pred used to predict c_in + pos_enc_dim). Raises if any
    encoder/projection weight is missing or mismatched. Returns dropped keys.
    """
    sd  = torch.load(str(path), map_location=device, weights_only=True)
    cur = projector.state_dict()
    dropped = [k for k, v in sd.items() if k in cur and v.shape != cur[k].shape]
    for k in dropped:
        sd.pop(k)
    missing, unexpected = projector.load_state_dict(sd, strict=False)
    decoder_only = ("dec_", "decoder", "mask_token")
    bad = [k for k in list(missing) + dropped if not k.startswith(decoder_only)]
    if bad:
        raise RuntimeError(
            f"{path}: encoder/projection weights missing or shape-mismatched: {bad}. "
            f"Check proj_hidden / curv_channels / disk_grid / fourier_bands "
            f"against the checkpoint's run_config.yml.")
    if dropped or missing:
        log.warning(f"Projector: decoder weights re-initialised (pre-fix checkpoint): "
                    f"{sorted(set(dropped) | set(missing))}")
    if unexpected:
        log.warning(f"Projector: unexpected keys ignored: {list(unexpected)}")
    return dropped + list(missing)


# ──────────────────────────────────────────────────────────────────────────────
# JSON target builder + prompts
# ──────────────────────────────────────────────────────────────────────────────

def build_target_json(sample_meta: dict, caption: str, obj_class: str,
                      fields: list[str] = DEFAULT_TARGET_FIELDS) -> str:
    """JSON string the LLM learns to generate. Missing fields → "".
    'class' is never filled (not in meta) so the class name never enters training."""
    obj = {f: (caption if f == "caption" else sample_meta.get(f, "")) for f in fields}
    return json.dumps(obj, ensure_ascii=False)


def meta_cols_for(fields: list[str]) -> list[str]:
    """CSV columns to read into MeshData.meta for the target JSON."""
    return [f for f in fields if f not in ("caption", "class")]


SYSTEM_MSG = """\
You are a precise 3D fracture analyst.

You will receive geometry embeddings representing a fractured 3D object as a prefix to the conversation.

Your task is to infer the object's overall shape and fracture characteristics solely from the provided geometry. Do not rely on any external knowledge or assumptions.

Return exactly one valid JSON object with no additional text, markdown, or code blocks.

The JSON must have exactly this structure:

{
  "object_description": "Shape, structure, proportions, and notable features of the original object. One or two sentences, maximum 30 words.",
  "confidence": "low | medium | high",
  "fracture_type": "BOUNDARY | INTERIOR",
  "caption": "One concise sentence describing the object and the location and nature of the fracture. Maximum 25 words."
}

Rules:
- Output valid JSON only.
- Exactly one sentence for every field (two allowed only for object_description).
- Stay within the stated word limits.
- Base every field only on the supplied geometry.
- Do not mention geometry embeddings, tokens, prefixes, meshes, vertices, faces, triangles, point clouds, or any internal representation.
- Do not speculate beyond what the geometry supports. If the geometry is insufficient, express uncertainty only through the confidence field.
- Do not repeat information across fields.
- Describe fracture locations relative to the object's own structure (e.g. "along the rim", "near the handle", "at the base", "near one corner", "along an outer edge", "on a flat face"), never relative to a viewpoint.
- Prefer concrete structural vocabulary such as edge, lip, border, perimeter, opening, corner, face, shoulder, junction, base, rim, handle, and tip.
- The caption should describe only geometrically observable properties. Do not infer material, manufacturing process, cause of fracture, or object function unless clearly supported by the geometry.
"""

# CLASS NAME INTENTIONALLY NOT in the user message — it would let the LLM
# generate from the class prior and ignore the geometry prefix.
USER_MSG = """\
Analyze the provided fractured object geometry.

Classify the fracture as one of:
- BOUNDARY: the missing fragment was attached along an outer edge, tip, handle, rim, or other protruding feature.
- INTERIOR: the missing fragment originated from a flat, recessed, or enclosed surface rather than an outer boundary.

Return the required JSON object.
"""


def build_chat_text(obj_class: str, tokenizer) -> str:
    """Chat-template prompt (no geometry prefix, no class name)."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": USER_MSG},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ──────────────────────────────────────────────────────────────────────────────
# Collate + tokenise
# ──────────────────────────────────────────────────────────────────────────────

def collate_and_tokenise(batch, tokenizer, curv_ch_idx: list[int], max_seq_len: int,
                         device: torch.device, log_first_n: int = 0,
                         target_fields: list[str] = DEFAULT_TARGET_FIELDS):
    """Returns patches (B,N,C,D,D), pos_enc (B,N,P), input_ids, labels, attn_mask, n_geo_tok."""
    patches_list, pos_enc_list = [], []
    input_ids_list, label_list = [], []

    for s in batch:
        patches_list.append(s.data[:, curv_ch_idx, :, :])
        pos_enc_list.append(s.pos_enc)

        prompt_text = build_chat_text(s.x, tokenizer)
        target_text = build_target_json(s.meta, s.caption, s.x, target_fields)

        if collate_and_tokenise._logged < log_first_n:
            log.info(f"[target check #{collate_and_tokenise._logged}] "
                     f"class={s.x!r}  meta={s.meta}  →  target_json={target_text}")
            collate_and_tokenise._logged += 1

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                               return_tensors="pt").input_ids[0]
        target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt",
                               max_length=max_seq_len, truncation=True).input_ids[0]
        target_ids = torch.cat([target_ids,
                                torch.tensor([tokenizer.eos_token_id], dtype=torch.long)])

        input_ids_list.append(torch.cat([prompt_ids, target_ids]))
        label_list.append(torch.cat([
            torch.full((len(prompt_ids),), -100, dtype=torch.long), target_ids,
        ]))

    pad_id  = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_len = max(t.shape[0] for t in input_ids_list)
    input_ids = torch.stack([torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_id)
                             for t in input_ids_list])
    labels = torch.stack([torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=-100)
                          for t in label_list])
    attn_mask = (input_ids != pad_id).long()

    # Normalise N across the batch (meshes with < n_fps vertices give fewer points)
    N_min = min(p.shape[0] for p in patches_list)
    patches = torch.stack([p[:N_min] for p in patches_list])
    pos_enc = torch.stack([pe[:N_min] for pe in pos_enc_list])

    return (patches.to(device), pos_enc.to(device), input_ids.to(device),
            labels.to(device), attn_mask.to(device), patches.shape[1])


collate_and_tokenise._logged = 0


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass with geometry prefix injection
# ──────────────────────────────────────────────────────────────────────────────

def _get_embed_tokens(llm):
    """Find the token embedding layer regardless of PEFT wrapping depth."""
    for name, mod in llm.named_modules():
        if name.endswith("embed_tokens"):
            return mod
    raise AttributeError(
        f"embed_tokens not found. First 40 module names: "
        f"{[n for n, _ in llm.named_modules()][:40]}")


def geometry_diversity_loss(geo_embeds: torch.Tensor, lam: float = 0.05) -> torch.Tensor:
    """lam × mean off-diagonal cosine similarity of mesh-pooled geometry
    embeddings within the batch. 0 when B == 1."""
    B = geo_embeds.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=geo_embeds.device, requires_grad=True)
    pooled = torch.nn.functional.normalize(geo_embeds.mean(dim=1), dim=-1)
    sim = pooled @ pooled.T
    mask = ~torch.eye(B, dtype=torch.bool, device=pooled.device)
    return lam * sim[mask].mean()


def pool_geo_tokens(geo_embeds: torch.Tensor, max_geo: int) -> torch.Tensor:
    """Adaptive avg-pool (B, N, D) → (B, max_geo, D) when N > max_geo.
    NOTE: pools consecutive tokens in FPS order, which are NOT spatial neighbours."""
    if geo_embeds.shape[1] <= max_geo:
        return geo_embeds
    ge_t = torch.nn.functional.adaptive_avg_pool1d(geo_embeds.permute(0, 2, 1), max_geo)
    return ge_t.permute(0, 2, 1)


def forward_with_geo_prefix(llm, projector: GeometryProjector, patches, pos_enc,
                            input_ids, labels, attn_mask, n_geo_tok: int,
                            geo_div_lam: float = 0.05, mae_lam: float = 0.1,
                            max_geo_tokens: int | None = None):
    """Total loss = LM loss + λ_div·L_div + λ_mae·L_mae."""
    B = input_ids.shape[0]

    # 1. Geometry prefix
    if projector.training and projector.mask_ratio > 0:
        geo_embeds, mae_raw = projector.encode_train(patches, pos_enc,
                                                     compute_mae=mae_lam > 0)
    else:
        geo_embeds, mae_raw = projector(patches, pos_enc), None

    # 2. Text embeddings (LLM dtype) and cast geometry to match
    text_embeds = _get_embed_tokens(llm)(input_ids)
    geo_embeds  = geo_embeds.to(text_embeds.dtype)

    # 3. Cap prefix length
    if max_geo_tokens is not None:
        geo_embeds = pool_geo_tokens(geo_embeds, max_geo_tokens)
    n_geo_tok = geo_embeds.shape[1]

    # 4. [geo prefix | text], extended mask and labels
    full_embeds = torch.cat([geo_embeds, text_embeds], dim=1)
    full_mask   = torch.cat([torch.ones(B, n_geo_tok, dtype=attn_mask.dtype,
                                        device=attn_mask.device), attn_mask], dim=1)
    full_labels = torch.cat([torch.full((B, n_geo_tok), -100, dtype=labels.dtype,
                                        device=labels.device), labels], dim=1)

    outputs  = llm(inputs_embeds=full_embeds, attention_mask=full_mask, labels=full_labels)
    lm_loss  = outputs.loss
    div_loss = geometry_diversity_loss(geo_embeds, lam=geo_div_lam)
    if mae_raw is not None:
        mae_loss = mae_raw.to(lm_loss.device) * mae_lam
    else:
        mae_loss = torch.tensor(0.0, device=lm_loss.device)

    from types import SimpleNamespace
    return SimpleNamespace(
        loss=lm_loss + div_loss + mae_loss,
        lm_loss=lm_loss.detach(), div_loss=div_loss.detach(),
        mae_loss=mae_loss.detach(), geo_embeds=geo_embeds.detach(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sanity-check generation — real llm.generate() on a fixed sample
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sanity_check_generation(llm, projector, tokenizer, sample, curv_ch_idx, device,
                            max_geo_tokens: int, max_new_tokens: int = 200,
                            target_fields: list[str] = DEFAULT_TARGET_FIELDS) -> dict:
    was_training_llm, was_training_proj = llm.training, projector.training
    prev_use_cache = llm.config.use_cache
    llm.eval(); projector.eval(); llm.config.use_cache = True
    try:
        patch   = sample.data[:, curv_ch_idx, :, :].unsqueeze(0).to(device)
        pos_enc = sample.pos_enc.unsqueeze(0).to(device)
        geo_embeds = projector(patch, pos_enc)

        prompt_ids = tokenizer(build_chat_text(sample.x, tokenizer), add_special_tokens=False,
                               return_tensors="pt").input_ids.to(device)
        text_embeds = _get_embed_tokens(llm)(prompt_ids)
        geo_embeds  = pool_geo_tokens(geo_embeds.to(text_embeds.dtype), max_geo_tokens)

        full_embeds = torch.cat([geo_embeds, text_embeds], dim=1)
        out_ids = llm.generate(
            inputs_embeds=full_embeds,
            attention_mask=torch.ones(1, full_embeds.shape[1], dtype=torch.long, device=device),
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        target_json = build_target_json(sample.meta, sample.caption, sample.x, target_fields)
        expected_keys = set(json.loads(target_json).keys())

        parsed_ok, parsed_keys, cleaned = False, [], raw
        for fence in ["```json", "```"]:
            if fence in cleaned:
                cleaned = cleaned.split(fence, 1)[-1].rsplit("```", 1)[0]
        try:
            obj = json.loads(cleaned.strip())
            if isinstance(obj, dict):
                parsed_ok, parsed_keys = True, list(obj.keys())
        except json.JSONDecodeError:
            pass
        return {"raw_generation": raw, "target_json": target_json,
                "parsed_ok": parsed_ok, "parsed_keys": parsed_keys,
                "expected_keys": sorted(expected_keys),
                "keys_match_schema": parsed_ok and set(parsed_keys) == expected_keys}
    except Exception as e:
        return {"raw_generation": f"<sanity check failed: {e}>", "target_json": "",
                "parsed_ok": False, "parsed_keys": [], "expected_keys": [],
                "keys_match_schema": False}
    finally:
        llm.config.use_cache = prev_use_cache
        if was_training_llm:
            llm.train()
        if was_training_proj:
            projector.train()


# ──────────────────────────────────────────────────────────────────────────────
# Train / val split
# ──────────────────────────────────────────────────────────────────────────────

def source_mesh_key(path: str) -> str:
    """out/everyday/<Class>/<mesh_hash>/fractured_k/piece_0.obj → '<Class>/<mesh_hash>'."""
    parts = Path(str(path).strip()).parts
    return "/".join(parts[-4:-2]) if len(parts) >= 4 else str(path)


def split_dataset(full_ds, val_frac: float, seed: int, by: str):
    n = len(full_ds)
    if by == "row":
        n_val = max(1, int(n * val_frac))
        return random_split(full_ds, [n - n_val, n_val],
                            generator=torch.Generator().manual_seed(seed))
    keys = full_ds.records["path"].map(source_mesh_key).to_numpy()
    groups = np.unique(keys)
    perm = np.random.default_rng(seed).permutation(len(groups))
    n_val_groups = max(1, int(round(len(groups) * val_frac)))
    val_groups = set(groups[perm[:n_val_groups]])
    val_idx   = [i for i, k in enumerate(keys) if k in val_groups]
    train_idx = [i for i, k in enumerate(keys) if k not in val_groups]
    log.info(f"Object-level split: {len(groups)} source meshes, "
             f"{n_val_groups} held out for val")
    return Subset(full_ds, train_idx), Subset(full_ds, val_idx)


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "4")
    torch.set_num_threads(4)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_resolved(args)
    saved_cfg = dump_resolved(args, out_dir / SAVED_CONFIG_NAME, script="train")
    log.info(f"Resolved config → {saved_cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = args.wandb is not None
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb, config=vars(args))
            try:
                wandb.save(str(saved_cfg), base_path=str(saved_cfg.parent), policy="now")
            except Exception as e:
                log.warning(f"Could not upload run_config.yml to W&B: {e}")
        except ImportError:
            log.warning("wandb not installed — logging disabled")
            use_wandb = False

    # ── Dataset ──────────────────────────────────────────────────────────────
    from BreakingBadDataset import BreakingBadDataset, DatasetConfig

    curv_ch_idx = CURV_CHANNEL_MAP[args.curv_channels]
    log.info(f"Curvature channels used ({args.curv_channels}): "
             f"{[CURV_CHANNEL_NAMES[i] for i in curv_ch_idx]}")

    cfg = DatasetConfig(
        n_fps=args.n_fps, disk_grid=args.disk_grid,
        disk_radius_frac=args.disk_radius_frac, disk_channels=6,
        disk_fill=args.disk_fill, curv_knn=args.curv_knn, cache_processed=True,
        fourier_bands=args.fourier_bands, use_tangent_disk=args.use_tangent_disk,
    )
    full_ds = BreakingBadDataset(csv_path=args.csv, dataset_root=args.root, cfg=cfg,
                                 classes=args.classes,
                                 meta_cols=meta_cols_for(args.target_fields))
    train_ds, val_ds = split_dataset(full_ds, args.val_split, args.seed, args.val_split_by)
    n_train = len(train_ds)
    log.info(f"Train: {n_train}  Val: {len(val_ds)}  (split by {args.val_split_by})")

    sanity_sample = None
    if args.sanity_check_every_save:
        idx = min(args.sanity_sample_idx, len(val_ds) - 1)
        sanity_sample = val_ds[idx]
        log.info(f"Sanity-check sample pinned: val idx={idx}  class={sanity_sample.x!r}  "
                 f"target={build_target_json(sanity_sample.meta, sanity_sample.caption, sanity_sample.x, args.target_fields)}")

    def raw_collate(batch):
        return batch

    loader_kw = dict(batch_size=args.batch, num_workers=args.dataloader_workers,
                     pin_memory=args.dataloader_workers > 0,
                     persistent_workers=args.dataloader_workers > 0, collate_fn=raw_collate)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ── Load Qwen2.5-VL ──────────────────────────────────────────────────────
    log.info(f"Loading {args.model_id} ...")
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = None
    if args.load_4bit:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.bfloat16)

    attn_impl = "sdpa"
    if args.flash_attn:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            log.warning("flash-attn not installed — falling back to SDPA")
    log.info(f"LLM attention: {attn_impl}")

    llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16 if not args.load_4bit else None,
        device_map="auto", trust_remote_code=True, attn_implementation=attn_impl,
    )
    llm.config.use_cache = False
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── LoRA ─────────────────────────────────────────────────────────────────
    log.info(f"LoRA target_modules: {args.lora_target_modules}")
    llm = get_peft_model(llm, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.lora_target_modules, bias="none",
    ))
    llm.print_trainable_parameters()

    # ── Geometry projector (CTE) ─────────────────────────────────────────────
    d_model = llm.config.text_config.hidden_size
    projector = GeometryProjector(
        c_in=args.curv_channels, d_grid=args.disk_grid, d_model=d_model,
        pos_enc_dim=pos_enc_dim_for(args.fourier_bands), hidden=args.proj_hidden,
        n_heads=args.mae_n_heads, enc_layers=args.mae_enc_layers,
        dec_layers=args.mae_dec_layers, mask_ratio=args.mae_mask_ratio,
        dropout=args.proj_dropout, chunk_size=args.mae_chunk_size,
    ).to(device)

    S_vis = args.disk_grid ** 2 - int(args.disk_grid ** 2 * args.mae_mask_ratio)
    qkv_mb = args.mae_chunk_size * S_vis * args.proj_hidden * 3 * 4 / 1e6
    log.info(f"MAE chunk_size={args.mae_chunk_size}  S_vis={S_vis}  hidden={args.proj_hidden}"
             f"  → QKV per chunk ≈ {qkv_mb:.0f} MB (fp32)")
    log.info(f"GeometryProjector params: {sum(p.numel() for p in projector.parameters()):,}")
    if args.max_geo_tokens < args.n_fps:
        log.warning(f"max_geo_tokens={args.max_geo_tokens} < n_fps={args.n_fps}: geometry "
                    f"tokens are avg-pooled in FPS order (FPS-adjacent points are not "
                    f"spatial neighbours).")
    log.info(f"Sequence length: ≤{min(args.max_geo_tokens, args.n_fps)} geo + "
             f"≤{args.max_seq_len} target + prompt tokens")

    # ── Optimiser + schedule ─────────────────────────────────────────────────
    # Only the LoRA tensors go into the LLM group (the frozen base never gets
    # gradients; listing all 7B params only slowed optimizer bookkeeping).
    lora_params = [p for n, p in llm.named_parameters() if "lora_" in n]
    optimizer = torch.optim.AdamW(
        [{"params": list(projector.parameters()), "lr": args.proj_lr},
         {"params": lora_params,                  "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    total_steps = (n_train // (args.batch * args.grad_accum)) * args.epochs

    # ── Resume ───────────────────────────────────────────────────────────────
    global_step, start_epoch, best_val_loss = 0, 0, float("inf")
    phase1_end_step = args.freeze_lm_steps

    if args.resume:
        log.info(f"Resuming from checkpoint: {args.resume}")
        global_step, start_epoch, best_val_loss, new_modules_detected = load_checkpoint_for_resume(
            ckpt_dir=args.resume, llm=llm, projector=projector,
            optimizer=optimizer, device=str(device),
        )
        if new_modules_detected:
            if args.refreeze_on_new_lora_modules and args.freeze_lm_steps > 0:
                phase1_end_step = global_step + args.freeze_lm_steps
                log.info(f"New LoRA target_modules in checkpoint — phase 1 restarted "
                         f"until global_step={phase1_end_step}.")
            else:
                log.warning("New LoRA target_modules detected on resume; no phase-1 "
                            "re-warm-up (freeze_lm_steps=0 or refreeze disabled).")
        if global_step >= total_steps:
            log.warning(f"global_step ({global_step}) >= total_steps ({total_steps}); "
                        f"increase --epochs to continue.")

    # Built after resume so the LoRA cycle starts at the (possibly shifted)
    # phase1_end_step. LR is a pure function of global_step → resume-exact.
    lora_start = phase1_end_step if args.lr_schedule == "phase2" else 0
    scheduler = PhasedOneCycle(
        optimizer, max_lrs=[args.proj_lr, args.lr], total_steps=total_steps,
        starts=[0, lora_start], pct_start=args.warmup_ratio, step=global_step,
    )
    log.info(f"Total steps: {total_steps}  schedule={args.lr_schedule}")
    log.info(f"  projector: warmup steps 0→{scheduler.peak_step(0)}, decay to {total_steps}")
    log.info(f"  LoRA     : " + (f"lr=0 until {lora_start}, " if lora_start else "")
             + f"warmup {lora_start}→{scheduler.peak_step(1)}, decay to {total_steps}")
    if args.resume:
        lrs = scheduler.get_last_lr()
        log.info(f"  resumed at step {global_step}: lr_proj={lrs[0]:.2e}  lr_lora={lrs[1]:.2e}")

    def _set_lm_trainable(trainable: bool):
        for name, param in llm.named_parameters():
            if "lora_" in name:
                param.requires_grad_(trainable)

    def _phase(step: int) -> int:
        return 1 if (phase1_end_step > 0 and step < phase1_end_step) else 2

    if _phase(global_step) == 1:
        _set_lm_trainable(False)
        log.info(f"Phase 1: LoRA frozen until step {phase1_end_step} — CTE trains alone")
    else:
        _set_lm_trainable(True)

    forward_kw = dict(geo_div_lam=args.div_lam, mae_lam=args.mae_lam,
                      max_geo_tokens=args.max_geo_tokens)

    # ── Training ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        llm.train()
        projector.train()
        accum = dict(loss=0.0, lm=0.0, div=0.0, mae=0.0, n=0)
        _last_geo, _t0 = None, time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            try:
                patches, pos_enc, input_ids, labels, attn_mask, n_geo = collate_and_tokenise(
                    batch, tokenizer, curv_ch_idx, args.max_seq_len, device,
                    log_first_n=args.log_target_samples, target_fields=args.target_fields,
                )
            except Exception as e:
                log.warning(f"Skipping batch (collate error): {e}")
                continue

            result = forward_with_geo_prefix(llm, projector, patches, pos_enc,
                                             input_ids, labels, attn_mask, n_geo, **forward_kw)
            (result.loss / args.grad_accum).backward()
            accum["loss"] += result.loss.item()
            accum["lm"]   += result.lm_loss.item()
            accum["div"]  += result.div_loss.item()
            accum["mae"]  += result.mae_loss.item()
            accum["n"]    += 1
            _last_geo = result.geo_embeds.detach().cpu()
            del result, patches, pos_enc, input_ids, labels, attn_mask

            if (step + 1) % args.grad_accum != 0:
                continue

            proj_params = list(projector.parameters())
            lora_params = [p for n, p in llm.named_parameters() if "lora_" in n and p.requires_grad]
            proj_gnorm = float(torch.nn.utils.clip_grad_norm_(proj_params, args.grad_clip)) \
                if proj_params else 0.0
            lora_gnorm = float(torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)) \
                if lora_params else 0.0
            grad_clipped = int(proj_gnorm > args.grad_clip or lora_gnorm > args.grad_clip)

            optimizer.step()
            if global_step < total_steps - 1:
                scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            global_step += 1

            if phase1_end_step > 0 and global_step == phase1_end_step:
                _set_lm_trainable(True)
                log.info(f"Phase 2 started at step {global_step}: LoRA unfrozen, joint training")

            n_micro = max(accum["n"], 1)
            avg = {k: accum[k] / n_micro for k in ("loss", "lm", "div", "mae")}
            accum = dict(loss=0.0, lm=0.0, div=0.0, mae=0.0, n=0)

            if global_step % args.log_steps == 0:
                lr_proj = optimizer.param_groups[0]["lr"]
                lr_lora = optimizer.param_groups[1]["lr"]
                geo_std = geo_cosim = 0.0
                if _last_geo is not None:
                    geo_std = float(_last_geo.std())
                    pooled = torch.nn.functional.normalize(_last_geo.mean(dim=1).float(), dim=-1)
                    sim = pooled @ pooled.T
                    Bp = pooled.shape[0]
                    geo_cosim = float(sim[~torch.eye(Bp, dtype=torch.bool)].mean()) if Bp >= 2 \
                        else float(sim[0, 0])
                elapsed = time.time() - _t0
                samples_sec = args.log_steps * args.grad_accum * args.batch / max(elapsed, 1e-6)
                _t0 = time.time()
                phase = _phase(global_step)
                log.info(
                    f"Epoch {epoch+1}  step {global_step}  [phase{phase}]  "
                    f"loss={avg['loss']:.4f}  lm={avg['lm']:.4f}  div={avg['div']:.4f}  "
                    f"mae={avg['mae']:.4f}  geo_std={geo_std:.3f}  cosim={geo_cosim:.3f}  "
                    f"gnorm_proj={proj_gnorm:.3f}  gnorm_lora={lora_gnorm:.3f}  "
                    f"lr_proj={lr_proj:.2e}  lr_lora={lr_lora:.2e}  {samples_sec:.1f} samp/s"
                )
                if use_wandb:
                    cuda = torch.cuda.is_available()
                    wandb.log({
                        "loss/train_total": avg["loss"], "loss/train_lm": avg["lm"],
                        "loss/train_div": avg["div"], "loss/train_mae": avg["mae"],
                        "geo/embed_std": geo_std, "geo/cosine_sim": geo_cosim,
                        "geo/div_loss_raw": avg["div"] / max(args.div_lam, 1e-8),
                        "grad/projector": proj_gnorm, "grad/lora": lora_gnorm,
                        "grad/clipped": grad_clipped,
                        "lr/projector": lr_proj, "lr/lora": lr_lora,
                        "perf/samples_per_sec": samples_sec,
                        "perf/gpu_memory_gb": torch.cuda.memory_allocated() / 1e9 if cuda else 0.0,
                        "perf/gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9 if cuda else 0.0,
                        "train/phase": phase, "step": global_step,
                    })

            if global_step % args.eval_steps == 0:
                val = evaluate(llm, projector, val_loader, tokenizer, curv_ch_idx,
                               args.max_seq_len, device, args.target_fields, forward_kw)
                log.info(f"  → val_loss={val['loss']:.4f}  lm={val['lm_loss']:.4f}  "
                         f"div={val['div_loss']:.4f}  json_parse={val['json_parse_rate']*100:.1f}%")
                if use_wandb:
                    wandb.log({"loss/val_total": val["loss"], "loss/val_lm": val["lm_loss"],
                               "loss/val_div": val["div_loss"],
                               "val/json_parse_rate": val["json_parse_rate"],
                               "step": global_step})
                if val["loss"] < best_val_loss:
                    best_val_loss = val["loss"]
                    save_checkpoint(llm, projector, tokenizer, out_dir, "best", saved_cfg)
                    save_training_state(optimizer, scheduler, global_step, epoch,
                                        best_val_loss, out_dir, "best")
                    log.info(f"  → saved best (val_loss={best_val_loss:.4f})")

            if global_step % args.save_steps == 0:
                tag = f"step{global_step}"
                save_checkpoint(llm, projector, tokenizer, out_dir, tag, saved_cfg)
                save_training_state(optimizer, scheduler, global_step, epoch,
                                    best_val_loss, out_dir, tag)
                if args.sanity_check_every_save and sanity_sample is not None:
                    check = sanity_check_generation(
                        llm, projector, tokenizer, sanity_sample, curv_ch_idx, device,
                        max_geo_tokens=args.max_geo_tokens,
                        max_new_tokens=args.sanity_max_new_tokens,
                        target_fields=args.target_fields,
                    )
                    log.info(f"  [sanity@{global_step}] parsed_ok={check['parsed_ok']}  "
                             f"schema_match={check['keys_match_schema']}  keys={check['parsed_keys']}")
                    log.info(f"    target : {check['target_json']}")
                    log.info(f"    gen    : {check['raw_generation'][:400]}")
                    if use_wandb:
                        wandb.log({"sanity/parsed_ok": int(check["parsed_ok"]),
                                   "sanity/schema_match": int(check["keys_match_schema"]),
                                   "sanity/n_keys": len(check["parsed_keys"]),
                                   "sanity/table": wandb.Table(
                                       columns=["step", "target", "generated", "schema_match"],
                                       data=[[global_step, check["target_json"],
                                              check["raw_generation"], check["keys_match_schema"]]]),
                                   "step": global_step})

        tag = f"epoch{epoch+1}"
        save_checkpoint(llm, projector, tokenizer, out_dir, tag, saved_cfg)
        save_training_state(optimizer, scheduler, global_step, epoch + 1,
                            best_val_loss, out_dir, tag)

    log.info("Training complete.")
    if use_wandb:
        wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
# Eval + checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(llm, projector, loader, tokenizer, curv_ch_idx, max_seq_len, device,
             target_fields, forward_kw) -> dict:
    """Val loss (teacher-forced) + JSON-parse rate of the decoded targets."""
    llm.eval()
    projector.eval()
    tot = dict(loss=0.0, lm=0.0, div=0.0)
    total_n = parse_ok = parse_total = 0
    for batch in loader:
        try:
            patches, pos_enc, input_ids, labels, attn_mask, n_geo = collate_and_tokenise(
                batch, tokenizer, curv_ch_idx, max_seq_len, device,
                target_fields=target_fields)
        except Exception:
            continue
        result = forward_with_geo_prefix(llm, projector, patches, pos_enc,
                                         input_ids, labels, attn_mask, n_geo, **forward_kw)
        b = len(batch)
        tot["loss"] += result.loss.item() * b
        tot["lm"]   += result.lm_loss.item() * b
        tot["div"]  += result.div_loss.item() * b
        total_n += b
        for i in range(b):
            tgt = labels[i][labels[i] != -100]
            if len(tgt) == 0:
                continue
            try:
                json.loads(tokenizer.decode(tgt.cpu(), skip_special_tokens=True).strip())
                parse_ok += 1
            except Exception:
                pass
            parse_total += 1
    llm.train()
    projector.train()
    n = max(total_n, 1)
    return {"loss": tot["loss"] / n, "lm_loss": tot["lm"] / n, "div_loss": tot["div"] / n,
            "json_parse_rate": parse_ok / max(parse_total, 1)}


def save_checkpoint(llm, projector, tokenizer, out_dir: Path, tag: str, config_file: Path):
    import shutil
    ckpt_dir    = out_dir / f"ckpt_{tag}"
    adapter_dir = ckpt_dir / "lora_adapter"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Avoid save_pretrained's HF HEAD request (hangs on restricted servers)
    prev = {k: os.environ.get(k) for k in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"]  = "1"
    try:
        llm.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    torch.save(projector.state_dict(), ckpt_dir / "geometry_projector.pt")
    shutil.copyfile(config_file, ckpt_dir / SAVED_CONFIG_NAME)
    log.info(f"Checkpoint saved → {ckpt_dir}")


def save_training_state(optimizer, scheduler, global_step: int, epoch: int,
                        best_val_loss: float, out_dir: Path, tag: str):
    ckpt_dir = out_dir / f"ckpt_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "global_step": global_step, "epoch": epoch, "best_val_loss": best_val_loss},
               ckpt_dir / "training_state.pt")


def load_checkpoint_for_resume(ckpt_dir, llm, projector, optimizer, device: str):
    """
    Restore LoRA, projector, optimizer and counters. The LR schedule is
    rebuilt by the caller from global_step. Returns (global_step, start_epoch,
    best_val_loss, new_lora_modules_detected).
    """
    from peft import set_peft_model_state_dict

    ckpt_dir    = Path(ckpt_dir)
    adapter_dir = ckpt_dir / "lora_adapter"
    proj_path   = ckpt_dir / "geometry_projector.pt"
    state_path  = ckpt_dir / "training_state.pt"
    new_modules_detected = False
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    # ── LoRA adapter ─────────────────────────────────────────────────────────
    if adapter_dir.exists():
        st_path, bin_path = adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin"
        adapters = None
        if st_path.exists():
            from safetensors.torch import load_file as st_load
            adapters = st_load(str(st_path), device=device)
        elif bin_path.exists():
            adapters = torch.load(str(bin_path), map_location=device, weights_only=True)
        else:
            log.warning(f"No adapter weights found in {adapter_dir}")
        if adapters is not None:
            res = set_peft_model_state_dict(llm, adapters)
            missing = [k for k in (getattr(res, "missing_keys", []) or []) if "lora_" in k]
            unexpected = [k for k in (getattr(res, "unexpected_keys", []) or []) if "lora_" in k]
            if missing:
                new_modules_detected = True
                mods = sorted({k.split(".lora_")[0].rsplit(".", 1)[-1] for k in missing})
                log.warning(f"{len(missing)} LoRA params not in checkpoint (freshly "
                            f"initialised) — new target_modules: {mods}")
            if unexpected:
                log.warning(f"{len(unexpected)} LoRA params in checkpoint unused: {unexpected[:5]}...")
            log.info(f"Loaded LoRA adapter from {adapter_dir}")
    else:
        log.warning(f"No lora_adapter in {ckpt_dir} — LoRA starts from scratch")

    # ── Projector ────────────────────────────────────────────────────────────
    proj_reinit = []
    if proj_path.exists():
        proj_reinit = load_projector_state(projector, proj_path, device)
        log.info(f"Loaded projector from {proj_path}")
    else:
        log.warning(f"No geometry_projector.pt in {ckpt_dir} — projector from scratch")

    # ── Optimizer + counters ─────────────────────────────────────────────────
    global_step, start_epoch, best_val_loss = 0, 0, float("inf")
    if state_path.exists():
        state = torch.load(str(state_path), map_location=device, weights_only=False)
        try:
            if proj_reinit:
                raise ValueError(f"projector weights re-initialised: {proj_reinit}")
            optimizer.load_state_dict(state["optimizer"])
            log.info("Optimizer state restored.")
        except (ValueError, RuntimeError, KeyError) as e:
            log.warning(f"Optimizer state NOT restored (Adam moments start fresh; "
                        f"weights are restored): {e}")
        global_step   = state["global_step"]
        start_epoch   = state["epoch"]
        best_val_loss = state["best_val_loss"]
        log.info(f"Resumed from step={global_step}  epoch={start_epoch}  "
                 f"best_val_loss={best_val_loss:.4f}")
    else:
        log.warning(f"No training_state.pt in {ckpt_dir} — counters reset to zero")
    return global_step, start_epoch, best_val_loss, new_modules_detected


if __name__ == "__main__":
    main()
