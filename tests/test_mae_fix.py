"""
test_mae_fix.py — sanity checks for the MAE encoder fix.
    python -m pytest -q tests/test_mae_fix.py     (or: python tests/test_mae_fix.py)

CPU-only, tiny model, a few seconds.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("torch.optim")          # needs real PyTorch (skipped otherwise)
import torch  # noqa: E402

from train_mae import GeometryProjector  # noqa: E402


def make(c_in=6, d_grid=8, mask_ratio=0.5, chunk=7):
    torch.manual_seed(0)
    return GeometryProjector(
        c_in=c_in, d_grid=d_grid, d_model=32, pos_enc_dim=6,
        hidden=32, n_heads=4, enc_layers=2, dec_layers=1,
        mask_ratio=mask_ratio, dropout=0.0, chunk_size=chunk,
    )


def test_mae_grad_reaches_encoder():
    proj = make().train()
    B, N, C, D = 2, 10, 6, 8           # BN=20, chunk=7 -> uneven chunks on purpose
    patches = torch.randn(B, N, C, D, D)
    pos_enc = torch.randn(B, N, 6)

    geo, mae = proj.encode_train(patches, pos_enc, compute_mae=True)
    assert geo.shape == (B, N, 32), geo.shape
    assert mae.ndim == 0 and mae.item() > 0

    proj.zero_grad()
    mae.backward()                      # MAE loss ONLY
    checks = {
        "patch_embed": proj.patch_embed.weight,
        "spatial_pos": proj.spatial_pos,
        **{f"encoder[{i}].qkv": b.qkv.weight for i, b in enumerate(proj.encoder)},
        "enc_norm": proj.enc_norm.weight,
        "dec_pred": proj.dec_pred.weight,
    }
    for name, p in checks.items():
        g = 0.0 if p.grad is None else p.grad.abs().sum().item()
        assert g > 0, f"no MAE gradient on {name}"
        print(f"  ok  grad|{name}| = {g:.3e}")
    # out_proj is not on the MAE path — should have no grad from MAE alone
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in proj.out_proj.parameters())


def test_unshuffle_order():
    # Reproduce _decode_chunk's restore step with identity tokens.
    proj = make()
    BN, S = 5, 64
    tok = torch.arange(S, dtype=torch.float32).view(1, S, 1).expand(BN, S, 1).contiguous()
    vis, keep, mask = proj._random_mask(tok, 0.5)
    masked_vals = mask.unsqueeze(-1).float()               # put true index into mask slots
    full = torch.cat([vis, masked_vals], dim=1)
    restore = torch.cat([keep, mask], dim=1).argsort(dim=1)
    full = full.gather(1, restore.unsqueeze(-1))
    assert torch.equal(full.squeeze(-1), torch.arange(S).float().expand(BN, S))
    print("  ok  unshuffle restores original cell order")


def test_disabled_paths():
    proj = make().train()
    p, pe = torch.randn(1, 4, 6, 8, 8), torch.randn(1, 4, 6)
    _, mae = proj.encode_train(p, pe, compute_mae=False)
    assert mae.item() == 0
    proj1 = make(d_grid=1).train()     # point mode: int(1*0.5)=0 cells masked
    _, mae = proj1.encode_train(torch.randn(1, 4, 6, 1, 1), pe, compute_mae=True)
    assert mae.item() == 0
    print("  ok  mae_lam=0 and D=1 paths return zero loss")


def test_old_checkpoint_loads(tmp_path=None):
    from train_mae import load_projector_state
    new = make()
    old_sd = new.state_dict()
    old_sd["dec_pred.weight"] = torch.randn(6 + 6, 32)     # pre-fix shape (c_in+pos_enc)
    old_sd["dec_pred.bias"] = torch.randn(6 + 6)
    import tempfile, os
    tmp = os.path.join(tempfile.mkdtemp(), "old_proj.pt")
    torch.save(old_sd, tmp)
    try:
        reinit = load_projector_state(make(), tmp, "cpu")
        assert set(reinit) == {"dec_pred.weight", "dec_pred.bias"}, reinit
        # an ENCODER mismatch must raise, not load silently
        old_sd["patch_embed.weight"] = torch.randn(32, 3)
        torch.save(old_sd, tmp)
        try:
            load_projector_state(make(), tmp, "cpu")
            raise AssertionError("encoder mismatch was not caught")
        except RuntimeError:
            pass
    finally:
        os.remove(tmp)
    print("  ok  pre-fix checkpoint loads (dec_pred re-initialised); encoder mismatch raises")


if __name__ == "__main__":
    for t in (test_mae_grad_reaches_encoder, test_unshuffle_order,
              test_disabled_paths, test_old_checkpoint_loads):
        print(t.__name__)
        t()
    print("\nAll MAE-fix checks passed.")
