"""Regenerate README result tables from reports/ and runs/ (between <!-- gen:NAME --> markers)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emtrack.data import valid_rate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
NAMES = {"native": "native tracker", "global_lgb": "global LightGBM (no site data)", "site_attention": "encoder-decoder attention",
         "fieldmap": "per-site polynomial field map", "global+fieldmap": "global LightGBM + field map",
         "attention+fieldmap": "attention + field map"}
STRESS = ["near_emitter", "under_table", "clutter", "strong", "clean"]


def site():
    r = pd.read_parquet(REP / "site_eval.parquet")
    st = REP / "site_eval_stratified.parquet"
    if st.exists():
        s = pd.read_parquet(st)
        r = pd.concat([r, s[s.protocol == "stratified"]], ignore_index=True)
    return r


def cell(g):
    return f"{g.pos_med.median():.1f} mm / {g.ang_med.median():.2f}° · **{100 * g.meets_target.mean():.0f} %**"


def budget_table():
    r = site()
    d = r[(r.split == "test_clinic") & (r.protocol.isin(["-", "random"]))]
    Ks = [0, 15, 60, 240, 900]
    rows = ["| method | " + " | ".join(f"{k} poses ({k * 4 / 60:g} min)" for k in Ks) + " |", "|---|" + "---|" * len(Ks)]
    for m in ["native", "global_lgb", "site_attention", "fieldmap", "global+fieldmap", "attention+fieldmap"]:
        cells = []
        for k in Ks:
            g = d[(d.method == m) & (d.K == k)]
            cells.append(cell(g) if len(g) else "–")
        rows.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def stress_table():
    r = site()
    cols = [("native", 0), ("global_lgb", 0), ("site_attention", 15), ("fieldmap", 240), ("attention+fieldmap", 240)]
    hdr = "| family (unseen in training) | valid frames | " + " | ".join(
        f"{NAMES[m]}{'' if k == 0 else f' @ {k * 4 / 60:g} min'}" for m, k in cols) + " |"
    rows = [hdr, "|---|---|" + "---|" * len(cols)]
    for s in ["test_clinic"] + STRESS:
        d = r[(r.split == s) & (r.protocol.isin(["-", "random"]))]
        cells = []
        for m, k in cols:
            g = d[(d.method == m) & (d.K == k)]
            cells.append(f"{100 * g.meets_target.mean():.0f} % ({g.pos_med.median():.1f} mm)")
        name = "clinic (in-family, unseen sites)" if s == "test_clinic" else s.replace("_", " ")
        rows.append(f"| {name} | {100 * valid_rate(s):.0f} % | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def protocol_table():
    r = site()
    d = r[(r.split == "test_clinic") & (r.method.isin(["fieldmap", "attention+fieldmap"])) & (r.minutes > 0)]
    Ks = [15, 60, 120, 240, 900]
    rows = ["| capture protocol | method | " + " | ".join(f"{k * 4 / 60:g} min" for k in Ks) + " |", "|---|---|" + "---|" * len(Ks)]
    pn = {"random": "random poses", "stratified": "stratified (k-means in pose space)", "coverage": "farthest-point (extremes first)",
          "convenient": "unguided operator (upright, central)"}
    for p in ["stratified", "random", "convenient", "coverage"]:
        for m in ["fieldmap", "attention+fieldmap"]:
            g = d[(d.protocol == p) & (d.method == m)]
            if g.empty:
                continue
            cells = []
            for k in Ks:
                x = g[g.K == k]
                cells.append(f"{x.pos_med.median():.2f} mm" if len(x) else "–")
            rows.append(f"| {pn[p]} | {NAMES[m]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def split_table():
    s = json.loads((REP / "split_study.json").read_text())
    names = {"random_rows": "random row split (leaky)", "grouped_envs": "held-out environments",
             "grouped_spatial": "held-out environments + outer shell (extrapolation)",
             "native_on_grouped_test": "native tracker, held-out environments", "native_on_spatial_test": "native tracker, outer shell"}
    rows = ["| evaluation | median pos error | median angle error | sites meeting target |", "|---|---|---|---|"]
    for k, n in names.items():
        v = s[k]
        rows.append(f"| {n} | {v['pos_med_mm']:.2f} mm | {v['ang_med_deg']:.2f}° | {100 * v['envs_meeting_target']:.0f} % |")
    return "\n".join(rows)


def shift_table():
    s = json.loads((REP / "shift_study.json").read_text())
    a, e = s["spatial_gate"], s["env_shift"]
    rows = ["| spatial gate (outer shell, model trained on the central volume) | median pos error |", "|---|---|",
            f"| native tracker | {a['native']['pos_med_mm']:.1f} mm |",
            f"| learned correction, ungated | {a['ungated']['pos_med_mm']:.1f} mm |",
            f"| learned correction, gated (fall back to native when flagged) | {a['gated']['pos_med_mm']:.1f} mm |",
            "", f"Frames flagged: {100 * a['flag_rate_outer_shell']:.0f} % in the outer shell (Mahalanobis alone: "
            f"{100 * a['flag_rate_outer_shell_mahalanobis_only']:.0f} %), {100 * a['flag_rate_inner_holdout']:.1f} % inside the trained volume.",
            "", "| split | environment-shift alarm rate (threshold: 99th pct of validation sites) |", "|---|---|"]
    for k, v in e["alarm_rate"].items():
        rows.append(f"| {k.replace('_', ' ')} | {100 * v:.1f} % |")
    rows.append("")
    rows.append(f"A site's alarm rate predicts that the global model will miss the target there with AUROC "
                f"{e['auroc_alarm_rate_predicts_global_miss']:.2f} ({e['n_miss']} of {e['n_envs']} sites miss).")
    return "\n".join(rows)


def ablation_table():
    reg = [json.loads(x) for x in (ROOT / "runs" / "registry.jsonl").read_text().splitlines() if x.strip()]
    last = {}
    for r in reg:
        last[r["run_id"]] = r
    names = {"global_lgb_all": "all 272 features", "global_lgb_no_ref": "without the reference receiver",
             "global_lgb_no_imu": "without the IMUs", "global_lgb_lag0": "current frame only (no lags)",
             "global_xgb_all": "XGBoost, all features", "global_mlp_all": "MLP, all features"}
    rows = ["| global model, features | clinic pos | clinic angle | strong-metal pos |", "|---|---|---|---|"]
    for k, n in names.items():
        if k in last:
            m = last[k]["metrics"]
            rows.append(f"| {n} | {m['test_clinic']['pos_med_mm']:.2f} mm | {m['test_clinic']['ang_med_deg']:.2f}° | {m['strong']['pos_med_mm']:.2f} mm |")
    return "\n".join(rows)


def denoise_table():
    p = REP / "denoise_summary.json"
    if not p.exists():
        return "_not run yet_"
    s = pd.DataFrame(json.loads(p.read_text())["by_method_dose"])
    order = ["single", "gauss3", "avg4", "gauss3+avg4", "n2n_single", "n2n_avg4", "sup_single"]
    nm = {"single": "1 frame, unfiltered", "gauss3": "1 frame + 3×3 Gaussian", "avg4": "4-frame average",
          "gauss3+avg4": "**current baseline**: 4-frame avg + 3×3 Gaussian", "n2n_single": "Noise2Noise U-Net, 1 frame",
          "n2n_avg4": "Noise2Noise U-Net, 4-frame avg", "sup_single": "supervised U-Net, 1 frame (upper bound)"}
    doses = sorted(s.dose.unique())
    rows = ["| method | frames (dose) | " + " | ".join(f"PSNR @ {int(d)}" for d in doses) + " | SSIM @ 200 | fracture-box SSIM @ 200 | false edges @ 200 | faint wire kept @ 200 |",
            "|---|---|" + "---|" * (len(doses) + 4)]
    for m in order:
        g = s[s.method == m].set_index("dose")
        if g.empty:
            continue
        fr = "4" if "avg4" in m else "1"
        w = g.loc[200].get("wire_transfer_0.04", float("nan"))
        rows.append(f"| {nm[m]} | {fr} | " + " | ".join(f"{g.loc[d, 'psnr']:.1f}" for d in doses)
                    + f" | {g.loc[200, 'ssim']:.3f} | {g.loc[200, 'fracture_ssim']:.3f} | {g.loc[200, 'false_edges']:.1f} | {100 * w:.0f} % |")
    return "\n".join(rows)


def flat_table():
    p = REP / "denoise_summary.json"
    if not p.exists():
        return "_not run yet_"
    f = pd.DataFrame(json.loads(p.read_text())["flat"])
    f = f[f.dose == 200].set_index("method")
    rows = ["| method (flat phantom, dose 200) | residual structure (high-pass std) | mean bias | spurious edge pixels |", "|---|---|---|---|"]
    for m in ["single", "gauss3+avg4", "n2n_single", "n2n_avg4", "sup_single"]:
        if m in f.index:
            rows.append(f"| {m} | {f.loc[m, 'hp_std']:.4f} | {f.loc[m, 'bias']:+.4f} | {f.loc[m, 'false_edges_px']:.0f} |")
    return "\n".join(rows)


GEN = {"budget": budget_table, "stress": stress_table, "protocols": protocol_table, "splits": split_table, "shift": shift_table,
       "ablations": ablation_table, "denoise": denoise_table, "flat": flat_table}


def main():
    p = ROOT / "README.md"
    s = p.read_text()
    for name, fn in GEN.items():
        s = re.sub(rf"(<!-- gen:{name} -->\n).*?(<!-- /gen:{name} -->)", lambda m, fn=fn: m.group(1) + fn() + "\n" + m.group(2), s, flags=re.S)
    p.write_text(s)
    print("README tables regenerated")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(GEN[sys.argv[1]]())
    else:
        main()
