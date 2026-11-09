"""Config tests for configs/mae_v3.yml:  python -m pytest -q tests/test_run_config.py"""
import shlex, yaml
import pytest
import run_config as rc
import train_mae, infer_mae, check_cache
from conftest import CONFIGS

CFG = str(CONFIGS / "mae_v3.yml")
_Y = yaml.safe_load(open(CFG))
DATA, INFER = _Y["data"], _Y["infer"]

V3_CMD = f"""--csv {DATA['train_csv']} --root {DATA['root']} --out {_Y['train']['out']}
 --n_fps 1024 --disk_grid 32 --curv_channels 6 --batch 2 --grad_accum 8 --epochs 5
 --load_4bit --wandb test_drive --save_steps 1000 --freeze_lm_steps 2000 --proj_lr 1e-3
 --log_steps 20 --proj_hidden 256 --warmup_ratio 0.1 --mae_mask_ratio 0.5 --mae_lam 0.2
 --max_geo_tokens 512 --max_seq_len 512 --mae_chunk_size 32 --dataloader_workers 2
 --mae_enc_layers 4 --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
 --lora_r 32"""

def train_args(argv): return train_mae.parse_args(argv)

def test_yaml_reproduces_v3_command():
    cli = vars(train_args(shlex.split(V3_CMD)))
    yml = vars(train_args(["--config", CFG]))
    intended = {"config", "cli_overrides", "val_split_by"}   # object split is a deliberate change
    diff = {k: (cli[k], yml[k]) for k in cli if k not in intended and cli[k] != yml[k]}
    assert diff == {}, diff
    assert yml["val_split_by"] == "object" and cli["val_split_by"] == "row"
    assert isinstance(yml["lr"], float) and yml["lr"] == 2e-4

def test_cli_overrides_yaml():
    a = train_args(["--config", CFG, "--epochs", "1", "--no_4bit", "--mae_lam=0.05"])
    assert (a.epochs, a.load_4bit, a.mae_lam) == (1, False, 0.05)
    assert set(a.cli_overrides) == {"epochs", "load_4bit", "mae_lam"}
    assert a.wandb == "test_drive"

def test_wandb_null_disables(tmp_path):
    c = yaml.safe_load(open(CFG)); c["train"]["wandb"] = None
    p = tmp_path / "c.yml"; p.write_text(yaml.safe_dump(c))
    assert train_args(["--config", str(p)]).wandb is None

def test_unknown_key_and_bad_choice_rejected(tmp_path):
    c = yaml.safe_load(open(CFG)); c["geometry"]["disk_gird"] = 32
    p = tmp_path / "typo.yml"; p.write_text(yaml.safe_dump(c))
    with pytest.raises(rc.ConfigError, match="disk_gird"):
        train_args(["--config", str(p)])
    c = yaml.safe_load(open(CFG)); c["geometry"]["curv_channels"] = 5
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(rc.ConfigError, match="curv_channels"):
        train_args(["--config", str(p)])

def test_string_float_coerced(tmp_path):
    c = yaml.safe_load(open(CFG)); c["train"]["lr"] = "2e-4"   # YAML 1.1 string
    p = tmp_path / "c.yml"; p.write_text(yaml.safe_dump(c))
    assert train_args(["--config", str(p)]).lr == 2e-4

def test_point_mode_forces_grid1(tmp_path):
    c = yaml.safe_load(open(CFG)); c["geometry"]["use_tangent_disk"] = False
    p = tmp_path / "c.yml"; p.write_text(yaml.safe_dump(c))
    assert train_args(["--config", str(p)]).disk_grid == 1
    a = train_args(["--config", CFG, "--use_tangent_disk"])
    assert a.use_tangent_disk is False and a.disk_grid == 1

def test_three_scripts_agree_on_geometry():
    t = train_args(["--config", CFG])
    i = infer_mae.parse_args(["--config", CFG])
    c = rc.parse_with_config(check_cache.build_parser(), "check_cache", ["--config", CFG])
    for k in ["n_fps", "use_tangent_disk", "disk_grid", "disk_radius_frac", "disk_fill",
              "curv_knn", "fourier_bands"]:
        assert getattr(t, k) == getattr(i, k) == getattr(c, k), k
    for k in ["curv_channels", "proj_hidden", "mae_enc_layers", "mae_dec_layers",
              "mae_n_heads", "max_geo_tokens", "model_id", "target_fields"]:
        assert getattr(t, k) == getattr(i, k), k
    assert check_cache.dataset_config(c).disk_radius_frac == 0.07
    assert c.csv == [DATA["train_csv"], DATA["test_csv"]]
    assert i.csv == DATA["test_csv"]
    assert i.ckpt == INFER["ckpt"]
    assert (i.load_4bit, i.num_samples, i.do_sample) == (False, 2, True)

def test_saved_config_roundtrip_and_infer_fallback(tmp_path):
    t = train_args(["--config", CFG, "--epochs", "1"])
    ckpt = tmp_path / "ckpt_best"; ckpt.mkdir()
    rc.dump_resolved(t, ckpt / rc.SAVED_CONFIG_NAME, "train")
    # saved file is itself a valid config that reproduces the run
    t2 = train_args(["--config", str(ckpt / rc.SAVED_CONFIG_NAME)])
    skip = {"config", "cli_overrides"}
    assert {k: v for k, v in vars(t2).items() if k not in skip} == \
           {k: v for k, v in vars(t).items() if k not in skip}
    # infer with only --ckpt picks up the saved geometry and stays bf16
    i = infer_mae.parse_args(["--ckpt", str(ckpt), "--csv", "x.csv"])
    assert (i.disk_grid, i.n_fps, i.curv_channels, i.load_4bit) == (32, 1024, 6, False)
    assert rc.check_consistency(i, ckpt / rc.SAVED_CONFIG_NAME) == []
    i_bad = infer_mae.parse_args(["--ckpt", str(ckpt), "--csv", "x.csv", "--disk_grid", "16"])
    probs = rc.check_consistency(i_bad, ckpt / rc.SAVED_CONFIG_NAME)
    assert len(probs) == 1 and "disk_grid" in probs[0]

def test_greedy_multi_sample_rejected():
    with pytest.raises(SystemExit):
        infer_mae.parse_args(["--config", CFG, "--greedy"])

def test_legacy_cli_still_works():
    a = train_args(["--csv", "a.csv", "--root", "."])
    assert a.config is None and a.n_fps == 512 and a.disk_radius_frac == 0.07
    with pytest.raises(SystemExit):
        train_args([])            # --csv/--root still required without a config

def test_source_mesh_key():
    k = train_mae.source_mesh_key("out/everyday/Plate/1389e9/fractured_11/piece_0.obj")
    assert k == "Plate/1389e9"
