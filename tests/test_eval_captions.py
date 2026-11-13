"""
test_eval_captions.py — CPU-only checks for evaluation/eval_captions.py (no CLIP download).
    python -m pytest -q tests/test_eval_captions.py
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest
import yaml

import eval_captions as ec


# ── deterministic stand-in for CLIP ──────────────────────────────────────────
def _h(tok, d=64):
    return np.random.default_rng(int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)).standard_normal(d)


class FakeEmbedder:
    def texts(self, texts):
        out = []
        for t in texts:
            toks = re.sub(r"[^a-z0-9\s]", " ", str(t).lower()).split() or ["<empty>"]
            v = sum(_h(x) for x in toks)
            out.append(v / np.linalg.norm(v))
        return np.stack(out).astype(np.float32)

    def images(self, paths):
        out = []
        for p in paths:
            v = _h("img:" + str(p)) + _h(re.search(r"everyday/([^/]+)/", str(p)).group(1).lower())
            out.append(v / np.linalg.norm(v))
        return np.stack(out).astype(np.float32)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    gt, pred = [], []
    for cls, n in {"Vase": 12, "Bowl": 8}.items():
        for i in range(n):
            p = f"out/everyday/{cls}/h{i}/fractured_0/piece_0.obj"
            d = tmp_path / p
            d.parent.mkdir(parents=True, exist_ok=True)
            img = d.parent / "piece_0_overview_pair.png"
            Image.new("RGB", (4, 4)).save(img)
            ims = json.dumps([str(img.relative_to(tmp_path))])
            ft = ["BOUNDARY", "INTERIOR"][i % 2]
            for k in range(2):
                gt.append(dict(path=p, caption_idx=k, images_used=ims, **{"class": cls},
                               caption=f"{cls} {ft} break near the {rng.choice(['rim', 'base', 'neck'])} {i} {k}"))
                cap = "same caption" if (cls == "Vase" and i < 6) else f"{cls} cracked at part {i}{k}"
                pred.append(dict(path=p, caption_idx=k, caption=cap, true_class=cls,
                                 fracture_type=ft, parse_error=False))
    pd.DataFrame(gt).to_csv("gt.csv", index=False)
    pd.DataFrame(pred).to_csv("pred.csv", index=False)
    pd.DataFrame(pred).assign(caption="identical").to_csv("collapsed.csv", index=False)
    cfg = {"run_name": "t", "data": {"test_csv": "gt.csv", "root": "."},
           "infer": {"out": "pred.csv"}, "eval": {"n_bootstrap": 500, "embed_cache": False}}
    (tmp_path / "t.yml").write_text(yaml.safe_dump(cfg))
    return tmp_path


def test_settings_derived_from_run_config(dataset):
    s = ec.load_settings("t.yml")
    assert s["true_csv"] == "gt.csv" and s["predictions"] == {"t": "pred.csv"}
    assert s["output_dir"] == "eval_outputs_t" and s["image_root"] == "."
    with pytest.raises(KeyError):
        ec.load_settings("t.yml", {"n_boostrap": 1})        # typo caught
    with pytest.raises(FileNotFoundError):
        ec.load_settings("t.yml", {"predictions": {"x": "nope.csv"}})


def test_end_to_end_and_collapse_detection(dataset):
    s = ec.load_settings("t.yml", {"predictions": {"model": "pred.csv", "collapsed": "collapsed.csv"}})
    r = ec.run(s, embedder=FakeEmbedder())
    h = r.headline.set_index("set")
    assert h.loc["collapsed", "Unique"] == 0 and h.loc["collapsed", "Top-1"] == 1
    assert h.loc["collapsed", "NearDup"] == 1
    assert h.loc["collapsed", "R@1"] == pytest.approx(1 / 20)          # ties → chance, not luck
    assert h.loc["model", "Unique"] > h.loc["collapsed", "Unique"]
    assert h.loc["model", "FracAcc"] == 1.0
    files = {f.name for f in ec.save(r)}
    assert {"summary_headline.csv", "tables.tex", "summary.txt", "per_caption_metrics.csv"} <= files
    assert "CAPTION EVALUATION" in ec.summary_text(r)


def test_r_precision_tie_modes():
    I = np.eye(3)
    T = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1.0]])     # captions 0 and 1 identical
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    exp = ec.r_precision(I, T, [1], "expected")[1]
    idx = ec.r_precision(I, T, [1], "index")[1]
    # image 0: tie between captions 0,1 → expected 0.5, index order → hit
    # image 1: similarity 0 to all three captions → 3-way tie → expected 1/3,
    #          index order puts caption 0 first → miss
    # image 2: unique best match → hit in both modes
    assert exp == pytest.approx((0.5 + 1 / 3 + 1) / 3)
    assert idx == pytest.approx((1 + 0 + 1) / 3)


def test_cider_multi_ref_is_mean_of_single_refs():
    refs = [["a vase broken at the rim", "the rim of the vase is chipped"],
            ["a bowl with a crack at the base", "the base of the bowl is broken"]]
    c = ec.CiderR(refs)
    cand = "the vase rim is broken"
    both = c.score(cand, refs[0])
    singles = np.mean([c.score(cand, [r]) for r in refs[0]])
    assert both == pytest.approx(singles)
    assert c.score("", refs[0]) == 0.0
    assert c.score("rim rim rim rim", refs[0]) < c.score("rim of vase", refs[0])   # repetition penalty


def test_bootstrap_reproducible(dataset):
    s = ec.load_settings("t.yml")
    a = ec.run(s, embedder=FakeEmbedder()).cider["t"]
    b = ec.run(s, embedder=FakeEmbedder()).cider["t"]
    pd.testing.assert_frame_equal(a, b)


# ── eval.yml format ──────────────────────────────────────────────────────────
@pytest.fixture
def eval_file(dataset):
    run = {"run_name": "v3", "data": {"test_csv": "gt.csv", "root": "."},
           "geometry": {"n_fps": 1024, "disk_grid": 32, "use_tangent_disk": True},
           "infer": {"out": "pred.csv"}}
    point = {**run, "geometry": {"n_fps": 1024, "disk_grid": 1, "use_tangent_disk": False}}
    (dataset / "run_v3.yml").write_text(yaml.safe_dump(run))
    (dataset / "run_point.yml").write_text(yaml.safe_dump(point))
    pd.read_csv("pred.csv").head(10).assign(score=3, human_reviewed=True).to_csv("scored.csv", index=False)
    cfg = {"name": "abl", "run_config": "run_v3.yml",
           "references": {"image_root": "."},                       # true_csv from run_config
           "predictions": {
               "curvature_disk": {"csv": "pred.csv", "label": "Curvature disk (ours)",
                                  "run_config": "run_v3.yml", "human_scores": "scored.csv"},
               "point_only": {"csv": "collapsed.csv", "label": "Point-only",
                              "run_config": "run_point.yml"},
               "image_only": {"csv": "missing.csv", "enabled": False},
               "bare": "pred.csv"},
           "metrics": {"n_bootstrap": 300, "embed_cache": False}}
    (dataset / "eval.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dataset


def test_eval_file_resolution(eval_file):
    s = ec.load_settings("eval.yml")
    assert s["config_format"] == "eval file" and s["run_name"] == "abl"
    assert s["true_csv"] == "gt.csv"                                 # inherited
    assert list(s["predictions"]) == ["curvature_disk", "point_only", "bare"]   # order, disabled dropped
    assert ec.label(s, "point_only") == "Point-only" and ec.label(s, "bare") == "bare"
    assert s["human_scores"] == {"curvature_disk": "scored.csv"}
    assert s["output_dir"] == "eval_outputs_abl"
    v = ec.variants_table(s).set_index("set")
    assert v.loc["point_only", "disk_grid"] == 1 and v.loc["curvature_disk", "disk_grid"] == 32


def test_eval_file_tables_use_labels(eval_file):
    s = ec.load_settings("eval.yml")
    r = ec.run(s, embedder=FakeEmbedder())
    sets = ec.summary_tables(r)["headline"]["set"].tolist()
    assert sets == ["references (Ref 1)", "Curvature disk (ours)", "Point-only", "bare"]
    assert "Curvature disk (ours) · Prediction 1" in ec.summary_tables(r)["clip"]["Caption"].tolist()
    assert len(r.human) == 1 and r.human["matched"].iloc[0] == 10
    names = {f.name for f in ec.save(r)}
    assert {"variants.csv", "source_eval.yml", "collapse_point_only.csv"} <= names
    assert "Point-only" in (eval_file / "eval_outputs_abl" / "tables.tex").read_text()


@pytest.mark.parametrize("patch, msg", [
    ({"predictions": {"x": {"csv": "pred.csv", "lable": "typo"}}}, "lable"),
    ({"metrics": {"n_boostrap": 1}}, "n_boostrap"),
    ({"refs": {}}, "refs"),
    ({"predictions": {"bad name": "pred.csv"}}, "file-safe"),
    ({"predictions": {"x": {"label": "no csv"}}}, "missing 'csv'"),
])
def test_eval_file_strict(eval_file, patch, msg):
    cfg = yaml.safe_load((eval_file / "eval.yml").read_text())
    cfg.update(patch)
    (eval_file / "bad.yml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=msg):
        ec.load_settings("bad.yml")


def test_run_config_format_still_works(eval_file):
    s = ec.load_settings("run_v3.yml")
    assert s["config_format"] == "run config" and s["predictions"] == {"v3": "pred.csv"}
    assert s["variant_run_configs"] == {"v3": "run_v3.yml"}


def test_cli_add_and_only(eval_file, capsys, monkeypatch):
    monkeypatch.setattr(ec, "CLIPEmbedder", lambda *a, **k: FakeEmbedder())
    ec.main(["--config", "eval.yml", "--only", "point_only", "--pred", "extra=pred.csv",
             "--only", "extra", "--set", "n_bootstrap=100"])
    out = capsys.readouterr().out
    assert "Point-only" in out and "extra" in out and "Curvature disk" not in out
