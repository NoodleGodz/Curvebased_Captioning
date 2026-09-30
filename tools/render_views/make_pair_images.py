"""
make_pair_images.py
-------------------
Stitch the renders produced by render_multiview.py into the side-by-side
pair images that caption_fractures.py expects:

    LEFT  : complete, intact object   (<stem>_base_view_<name>.png)
    RIGHT : fractured main body       (<stem>_frac_view_<name>.png)

For every row of the input CSV ('path' = fractured .obj, 'base_path' = base
.obj) it finds all <stem>_base_view_*.png / <stem>_frac_view_*.png pairs next
to the fractured .obj and writes

    <stem>_pair_view_overview.png
    <stem>_pair_view_frac00.png ... <stem>_pair_view_fracNN.png

The file name of the overview keeps the word "overview", which
caption_fractures.py uses to always include that image.

It also writes an output CSV with the columns caption_fractures.py needs:
    path, class, pair_images_path   (list literal of absolute PNG paths)
'class' is copied from the input CSV if present, otherwise set to "unknown".

Usage:
    python tools/render_views/make_pair_images.py meshes.csv --out pairs.csv
    python tools/render_views/make_pair_images.py meshes.csv --out pairs.csv --gap 8

Requirements:
    pip install pillow pandas
"""

import argparse
import glob
import os
import sys

import pandas as pd
from PIL import Image


def view_names(out_dir: str, stem: str) -> list:
    """View names that exist for BOTH base and frac renders, overview first."""
    def names(tag):
        pat = os.path.join(out_dir, f"{stem}_{tag}_view_*.png")
        pre = f"{stem}_{tag}_view_"
        return {os.path.basename(p)[len(pre):-4] for p in glob.glob(pat)}

    common = names("base") & names("frac")
    return sorted(common, key=lambda n: (n != "overview", n))


def stitch(left_path: str, right_path: str, out_path: str, gap: int):
    left  = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    h = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + gap + right.width, h), (255, 255, 255))
    canvas.paste(left,  (0, (h - left.height) // 2))
    canvas.paste(right, (left.width + gap, (h - right.height) // 2))
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser(
        description="Stitch base/frac renders into side-by-side pair images "
                    "and write the CSV for caption_fractures.py."
    )
    ap.add_argument("csv", help="Input CSV with columns: path, base_path [, class]")
    ap.add_argument("--out", default="pairs.csv", help="Output CSV (default: pairs.csv)")
    ap.add_argument("--gap", type=int, default=8, help="White gap in pixels (default: 8)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in ("path", "base_path") if c not in df.columns]
    if missing:
        sys.exit(f"[error] CSV missing column(s): {missing}. Found: {list(df.columns)}")
    has_class = "class" in df.columns

    out_rows, skipped = [], 0
    for _, row in df.iterrows():
        frac_obj = os.path.abspath(str(row["path"]))
        out_dir  = os.path.dirname(frac_obj)
        stem     = os.path.splitext(os.path.basename(frac_obj))[0]

        names = view_names(out_dir, stem)
        if not names:
            print(f"[skip] no matching base/frac renders for {frac_obj}")
            skipped += 1
            continue

        pair_paths = []
        for name in names:
            left  = os.path.join(out_dir, f"{stem}_base_view_{name}.png")
            right = os.path.join(out_dir, f"{stem}_frac_view_{name}.png")
            out   = os.path.join(out_dir, f"{stem}_pair_view_{name}.png")
            stitch(left, right, out, args.gap)
            pair_paths.append(out)

        out_rows.append({
            "path": str(row["path"]),
            "class": str(row["class"]) if has_class else "unknown",
            "pair_images_path": repr(pair_paths),
        })
        print(f"✔ {stem}: {len(pair_paths)} pair images")

    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"\nDone — {len(out_rows)} objects, {skipped} skipped. CSV: {args.out}")


if __name__ == "__main__":
    main()