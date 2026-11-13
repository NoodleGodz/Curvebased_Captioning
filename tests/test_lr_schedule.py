"""
test_lr_schedule.py — checks PhasedOneCycle (scripts/train_mae.py).
    python -m pytest -q tests/test_lr_schedule.py
With real torch installed it also compares against torch's OneCycleLR.
"""
import math

try:
    import torch
    HAVE_TORCH = hasattr(torch, "optim")
except ImportError:
    HAVE_TORCH = False

import pytest
from train_mae import PhasedOneCycle


class FakeOpt:
    def __init__(self, n):
        self.param_groups = [{"lr": 0.0} for _ in range(n)]


TOTAL, FREEZE, WARM = 5950, 2000, 0.1     # ≈ the v3 run
PROJ_LR, LORA_LR = 1e-3, 2e-4


def sched(starts, step=0):
    return PhasedOneCycle(FakeOpt(2), [PROJ_LR, LORA_LR], TOTAL, starts,
                          pct_start=WARM, step=step)


def test_lora_zero_while_frozen_then_full_warmup():
    s = sched([0, FREEZE])
    assert all(s.lr_at(1, k) == 0.0 for k in range(FREEZE))
    assert s.lr_at(1, FREEZE) == pytest.approx(LORA_LR / 25)          # starts at initial
    span = TOTAL - FREEZE
    peak = s.peak_step(1)
    assert peak == FREEZE + round(WARM * span - 1)
    assert s.lr_at(1, peak) == pytest.approx(LORA_LR, rel=1e-3)        # reaches max
    assert max(s.lr_at(1, k) for k in range(TOTAL)) <= LORA_LR + 1e-12
    assert s.lr_at(1, TOTAL - 1) == pytest.approx(LORA_LR / 25 / 1e4)  # ends at min


def test_old_schedule_wasted_lora_warmup():
    old = sched([0, 0])
    # legacy: LoRA peaked during phase 1 and was already decaying when unfrozen
    assert old.peak_step(1) < FREEZE
    assert old.lr_at(1, FREEZE) < LORA_LR


def test_projector_unchanged_and_global_mode_matches():
    new, old = sched([0, FREEZE]), sched([0, 0])
    for k in range(0, TOTAL, 37):
        assert new.lr_at(0, k) == old.lr_at(0, k)


def test_resume_is_exact():
    full = sched([0, FREEZE])
    for _ in range(3456):
        full.step()
    resumed = sched([0, FREEZE], step=3456)
    assert full.get_last_lr() == resumed.get_last_lr()
    s2 = sched([0, 0]); s2.load_state_dict(full.state_dict())
    assert s2.get_last_lr() == full.get_last_lr()


@pytest.mark.skipif(not HAVE_TORCH, reason="needs real torch")
def test_matches_torch_onecycle_when_start_is_zero():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [p], "lr": PROJ_LR}])
    ref = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[PROJ_LR], total_steps=TOTAL,
                                              pct_start=WARM)
    mine = PhasedOneCycle(FakeOpt(1), [PROJ_LR], TOTAL, [0], pct_start=WARM)
    for k in range(TOTAL - 1):
        assert mine.get_last_lr()[0] == pytest.approx(ref.get_last_lr()[0], rel=1e-9)
        opt.step(); ref.step(); mine.step()
