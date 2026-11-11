"""Run every study and regenerate results/results.json, results/results.md and figures/.

    python run_all.py

Every number in results.md is read from results.json, which this script
writes from the run itself. Nothing in the write-up is typed by hand.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lobofi import checks, io, maker as M, studies as S  # noqa: E402
from lobofi import report  # noqa: E402

BOOT_REPS = 2000
M_TICKERS = ["INTC", "MSFT"]   # one-tick-spread stocks, where queue position decides fills


def main():
    t0 = time.time()
    res = {"tickers": io.TICKERS, "data": {}, "replication": {}, "predictive": {},
           "leak": {}, "costs": {}, "transfer": {}}

    frames, leaky, cks_rows = {}, {}, {}
    for t in io.TICKERS:
        df = io.load(t)
        res["data"][t] = checks.summary(df)
        cks_rows[t] = S.replicate_cks(df)
        frames[t] = S.grid_frame(df)
        leaky[t] = S.leaky_grid_frame(df)
        print(f"{t}: loaded {len(df):,} messages", flush=True)

    # ---- study 1
    for t, rows in cks_rows.items():
        r2 = np.array([r["r2_ofi"] for r in rows]); r2ti = np.array([r["r2_ti"] for r in rows])
        tb = np.array([r["t_beta"] for r in rows])
        res["replication"][t] = {
            "windows": len(rows), "mean_r2_ofi": r2.mean(), "median_r2_ofi": float(np.median(r2)),
            "min_r2_ofi": r2.min(), "min_window": int(np.argmin(r2)), "max_r2_ofi": r2.max(),
            "mean_r2_ti": r2ti.mean(), "share_beta_significant": float(np.mean(tb > 1.96)),
            "mean_depth": float(np.mean([r["avg_depth"] for r in rows])), "rows": rows,
        }
    lam, se, n = S.depth_exponent(cks_rows)
    res["replication"]["_pooled"] = {
        "mean_r2_ofi": float(np.mean([res["replication"][t]["mean_r2_ofi"] for t in io.TICKERS])),
        "mean_r2_ti": float(np.mean([res["replication"][t]["mean_r2_ti"] for t in io.TICKERS])),
        "lambda": lam, "lambda_se": se, "lambda_n": n,
    }

    # ---- study 2 and 3
    models = {"Queue imbalance": ["qi"], "Lagged OFI": ["ofi_1s", "ofi_5s", "ofi_30s"],
              "Both": S.FEATURES}
    for t in io.TICKERS:
        fr = frames[t]
        res["predictive"][t] = {"spread_median": float(np.median(fr["spread"])),
                                "share_one_tick": float(np.mean(fr["spread"] == 1)),
                                "open_spread_median": float(np.median(fr["spread"][fr["t"] <= io.OPEN + S.HALF_HOUR]))}
        res["costs"][t] = {}
        for h in S.HORIZONS:
            y = fr[f"y_{h}s"]
            res["predictive"][t][f"zero_share_{h}s"] = float(np.nanmean(y == 0))
            for name, feats in models.items():
                wf = S.walk_forward(fr, feats, h)
                r2 = S.oos_r2(wf["y"], wf["pred"])
                ci = S.block_bootstrap(S.oos_r2, (wf["y"], wf["pred"]), reps=BOOT_REPS)
                hr = S.hit_rate(wf["y"], wf["pred"])
                hci = S.block_bootstrap(S.hit_rate, (wf["y"], wf["pred"]), reps=BOOT_REPS)
                res["predictive"][t][f"{name}|{h}"] = {
                    "r2": r2, "r2_ci": list(ci), "hit": hr, "hit_ci": list(hci), "n": len(wf["y"])}
                if name == "Both":
                    res["costs"][t][h] = S.taker_costs(wf)
            # leak demonstration, 1 s OFI only
            clean = S.walk_forward(fr, ["ofi_1s"], h)
            dirty = S.walk_forward(leaky[t], ["ofi_1s"], h)
            res["leak"].setdefault(t, {})[h] = {
                "clean_r2": S.oos_r2(clean["y"], clean["pred"]), "clean_hit": S.hit_rate(clean["y"], clean["pred"]),
                "leaky_r2": S.oos_r2(dirty["y"], dirty["pred"]), "leaky_hit": S.hit_rate(dirty["y"], dirty["pred"]),
            }
        print(f"{t}: predictive study done", flush=True)

    for h in S.HORIZONS:
        names, mat = S.transfer(frames, S.FEATURES, h)
        res["transfer"][h] = {"names": names, "r2": mat.tolist()}

    res["maker"] = {}
    for t in M_TICKERS:
        res["maker"][t] = M.maker_study(io.load(t))
        print(f"{t}: market-maker study done", flush=True)

    res["runtime_s"] = time.time() - t0
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "results.json").write_text(json.dumps(res, indent=1, default=float))
    report.write_markdown(res, ROOT / "results" / "results.md")
    report.write_figures(res, frames, ROOT / "figures")
    print(f"done in {res['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
