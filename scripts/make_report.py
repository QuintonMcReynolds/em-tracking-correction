"""Figures + headline numbers for the README from reports/*.parquet|json.

Palette: the dataviz reference categorical slots 1-4 (validated: CVD dE >= 9.1, normal-vision >= 22.9 on the light
surface). Slots 3-4 are under 3:1 contrast, so every series is also direct-labelled and marker-coded, and every
figure has a table in the README.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#8a8984"
COLORS = {"site_attention": "#2a78d6", "fieldmap": "#eb6834", "attention+fieldmap": "#1baf7a", "global+fieldmap": "#eda100"}
MARKERS = {"site_attention": "o", "fieldmap": "s", "attention+fieldmap": "D", "global+fieldmap": "^"}
LABELS = {"site_attention": "attention (few-shot)", "fieldmap": "per-site field map", "attention+fieldmap": "attention + field map",
          "global+fieldmap": "global GBM + field map", "global_lgb": "global GBM (no site data)", "native": "native tracker"}
STRESS = ["near_emitter", "under_table", "clutter", "strong", "clean"]


def place_labels(ax, items, x, min_gap):
    """items: [(y, text)] -> labels at x, nudged apart vertically so none overlap (keeps their order)."""
    items = sorted(items)
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for (_, t), y in zip(items, ys, strict=True):
        ax.annotate(t, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK2,
                    annotation_clip=False)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def budget_curves(res: pd.DataFrame, split="test_clinic", protocol="random"):
    d = res[res.split == split]
    ref = d[d.protocol == "-"].groupby("method").agg(pos=("pos_med", "median"), meets=("meets_target", "mean"))
    d = d[d.protocol == protocol]
    g = d.groupby(["method", "minutes"]).agg(pos=("pos_med", "median"), meets=("meets_target", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURFACE, gridspec_kw={"wspace": 0.85})
    xmax = g.minutes.max()
    for ax, col, ylab, title in ((axes[0], "pos", "median position error per site (mm)", "Error vs calibration time"),
                                 (axes[1], "meets", "sites meeting ±10 mm & ±1° (%)", "Sites meeting the target")):
        scale = 100 if col == "meets" else 1
        labels = []
        for m, c in COLORS.items():
            s = g[(g.method == m) & (g.minutes > 0)].sort_values("minutes")
            if s.empty:
                continue
            ax.plot(s.minutes, s[col] * scale, color=c, lw=2, marker=MARKERS[m], ms=6, mec=SURFACE, mew=1.5)
            labels.append((s[col].iloc[-1] * scale, LABELS[m]))
        refs = {m: ref.loc[m, col] * scale for m in ("native", "global_lgb")}
        for m, ls in (("native", ":"), ("global_lgb", "--")):
            ax.plot([0.8, xmax], [refs[m]] * 2, color=REF, lw=1.5, ls=ls)
        if abs(refs["native"] - refs["global_lgb"]) < (3 if col == "meets" else 0.3):
            labels.append((refs["native"], "native tracker & global GBM"))
        else:
            labels += [(refs[m], LABELS[m]) for m in refs]
        place_labels(ax, labels, xmax, 4.5 if col == "meets" else 0.32)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([1, 2, 4, 8, 16, 30, 60])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "30", "60"])
        ax.set_xlabel("site calibration time (min, 4 s per pose)", color=INK2, fontsize=9)
        style(ax, title, ylab)
        ax.set_xlim(0.8, 60)
    axes[0].axhline(10, color=INK2, lw=0.8)
    axes[0].annotate("±10 mm target", (0.85, 10), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2)
    axes[1].set_ylim(0, 100)
    fig.subplots_adjust(left=0.06, right=0.84, bottom=0.14, top=0.9)
    fig.savefig(FIG / f"budget_{split}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def protocol_curves(res: pd.DataFrame, method="fieldmap"):
    d = res[(res.method == method) & (res.split == "test_clinic") & (res.minutes > 0)]
    g = d.groupby(["protocol", "minutes"]).agg(pos=("pos_med", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    pc = {"stratified": "#2a78d6", "random": "#eb6834", "convenient": "#1baf7a", "coverage": "#eda100"}
    pl = {"stratified": "stratified (k-means in pose space)", "random": "random poses",
          "convenient": "unguided operator (upright, central)", "coverage": "farthest-point (extremes first)"}
    pm = {"stratified": "o", "random": "s", "convenient": "D", "coverage": "^"}
    labels = []
    for p, c in pc.items():
        s = g[g.protocol == p].sort_values("minutes")
        if s.empty:
            continue
        ax.plot(s.minutes, s.pos, color=c, lw=2, marker=pm[p], ms=6, mec=SURFACE, mew=1.5)
        labels.append((s.pos.iloc[-1], pl[p]))
    place_labels(ax, labels, g.minutes.max(), 0.18)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([1, 2, 4, 8, 16, 30, 60])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "30", "60"])
    ax.set_xlim(0.8, 60)
    ax.set_xlabel("site calibration time (min)", color=INK2, fontsize=9)
    style(ax, f"Which poses to capture ({LABELS[method]})", "median position error per site (mm)")
    fig.subplots_adjust(left=0.1, right=0.66, bottom=0.14, top=0.88)
    fig.savefig(FIG / f"protocols_{method.replace('+', '_')}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def headline(res: pd.DataFrame) -> dict:
    out = {}
    for split in res.split.unique():
        d = res[res.split == split]
        rows = {}
        for (m, p, k), g in d.groupby(["method", "protocol", "K"]):
            rows[f"{m}|{p}|{k}"] = {"pos_med": round(float(g.pos_med.median()), 2), "ang_med": round(float(g.ang_med.median()), 2),
                                    "meets": round(float(g.meets_target.mean()), 3), "n": int(g.env_id.nunique())}
        out[split] = rows
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.read_parquet(REP / "site_eval.parquet")
    if (REP / "site_eval_stratified.parquet").exists():
        st = pd.read_parquet(REP / "site_eval_stratified.parquet")
        res = pd.concat([res, st[st.protocol == "stratified"]], ignore_index=True)
    budget_curves(res)
    for m in ("fieldmap", "attention+fieldmap"):
        protocol_curves(res, m)
    (REP / "headline.json").write_text(json.dumps(headline(res), indent=1))
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main(*sys.argv[1:])
