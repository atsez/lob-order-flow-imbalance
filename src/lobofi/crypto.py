"""Binance USD-M futures: the same study on 24 recent days.

Data: Tardis.dev 'quotes' files for exchange 'binance-futures' (the best bid
and ask, one row each time the top of the book changes). Tardis publishes the
first day of every month without an API key, which gives one day a month from
November 2024 to October 2026. Binance's own bookTicker files stop in March
2024, so they cannot give recent days.

Columns used: local_timestamp (microseconds, when the update was received),
ask_amount, ask_price, bid_price, bid_amount.

Features are built on local_timestamp, the time the update was *received*,
because that is when a trader could have known it. The exchange timestamp can
run slightly out of order across Binance's feeds.

A day is processed in chunks, so memory stays flat however many rows the file
has. The test suite checks that the chunked result matches an in-memory
computation exactly.
"""
import datetime as dt
import gzip
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import studies as S

EXCHANGE = "binance-futures"
TICKS = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}   # exchange tick sizes (price filter)
DAY = 86_400
URL = "https://datasets.tardis.dev/v1/{ex}/quotes/{y:04d}/{m:02d}/{d:02d}/{sym}.csv.gz"
COLS = ["local_timestamp", "ask_amount", "ask_price", "bid_price", "bid_amount"]
MIN_TRAIN_DAYS = 3

# Binance USD-M futures, regular (VIP 0) account: maker 0.02%, taker 0.05%
FEE_BPS = {"maker": 2.0, "taker": 5.0}


# --------------------------------------------------------------------------- download

def month_starts(end="2026-10", months=24):
    y, m = map(int, end.split("-"))
    out = []
    for _ in range(months):
        out.append(dt.date(y, m, 1))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return sorted(out)


def url_for(day, symbol):
    return URL.format(ex=EXCHANGE, y=day.year, m=day.month, d=day.day, sym=symbol)


def download(day, symbol, dest_dir, retries=4):
    """Download one day to dest_dir, resuming nothing but retrying with backoff."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{symbol}_{day.isoformat()}.csv.gz"
    if out.exists():
        return out
    part = out.with_suffix(".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url_for(day, symbol), timeout=120) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            part.rename(out)
            return out
        except Exception as e:  # network errors: retry, then give up loudly
            if attempt == retries - 1:
                raise RuntimeError(f"download failed for {day} {symbol}: {e}") from e
            time.sleep(2 ** attempt)


# --------------------------------------------------------------------------- one day

def to_ticks(price, tick):
    """Prices as integer ticks; refuses if they are not on the tick grid."""
    q = price / tick
    r = np.rint(q)
    bad = np.abs(q - r) > 1e-6
    if bad.any():
        raise ValueError(f"{bad.sum()} prices are off the {tick} tick grid, e.g. {price[bad][:3]}")
    return r.astype(np.int64)


def process_day(path, day, tick, chunksize=2_000_000, step=1.0):
    """Stream one Tardis quotes file into level-1 states on a 1-second UTC grid.

    Returns (sample dict like features.sample, grid seconds, checks dict).
    Grid point k is (k+1) seconds after midnight; its state is the last update
    received at or before that time.
    """
    day0 = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()
    g = F.grid(step, DAY + step / 2, step)          # 1 .. 86400
    n = len(g)
    out = {k: np.full(n, np.nan) for k in ("bid", "ask", "bidsize", "asksize", "cumofi")}
    filled = 0
    last = None            # (pb, qb, pa, qa) of the previous row
    cum0 = 0.0             # cumulative OFI before this chunk
    prev_t = -np.inf
    chk = {"rows": 0, "dropped_empty_side": 0, "dropped_crossed": 0,
           "time_backwards": 0, "outside_day": 0}

    for c in pd.read_csv(path, usecols=COLS, chunksize=chunksize, dtype="float64"):
        chk["rows"] += len(c)
        ok = c[["ask_price", "bid_price", "ask_amount", "bid_amount"]].notna().all(axis=1).to_numpy()
        chk["dropped_empty_side"] += int((~ok).sum())
        c = c[ok]
        t = c["local_timestamp"].to_numpy() / 1e6 - day0
        pb = to_ticks(c["bid_price"].to_numpy(), tick)
        pa = to_ticks(c["ask_price"].to_numpy(), tick)
        qb = c["bid_amount"].to_numpy()
        qa = c["ask_amount"].to_numpy()
        crossed = pb >= pa
        chk["dropped_crossed"] += int(crossed.sum())
        keep = ~crossed
        t, pb, pa, qb, qa = t[keep], pb[keep], pa[keep], qb[keep], qa[keep]
        if len(t) == 0:
            continue
        # receipt times should not go backwards; if they do, hold the previous time
        mono = np.maximum.accumulate(np.concatenate([[prev_t], t]))[1:]
        chk["time_backwards"] += int(np.sum(mono > t))
        t = mono
        chk["outside_day"] += int(np.sum((t < 0) | (t >= DAY)))
        prev_t = t[-1]

        # event OFI, continuing from the last row of the previous chunk
        if last is None:
            e = F.event_ofi(pb, qb, pa, qa).astype(float)
        else:
            e = F.event_ofi(np.r_[last[0], pb], np.r_[last[1], qb],
                            np.r_[last[2], pa], np.r_[last[3], qa]).astype(float)[1:]
        cum = cum0 + np.cumsum(e)

        # grid points up to the last time in this chunk
        hi = int(np.searchsorted(g, t[-1], side="right"))
        if hi > filled:
            gg = g[filled:hi]
            idx = np.searchsorted(t, gg, side="right") - 1
            inside = idx >= 0
            sl = slice(filled, hi)
            for key, arr in (("bid", pb), ("ask", pa), ("bidsize", qb), ("asksize", qa), ("cumofi", cum)):
                vals = np.where(inside, arr[np.maximum(idx, 0)], np.nan)
                if last is not None:   # grid points before this chunk's first row use the carried state
                    carry = {"bid": last[0], "bidsize": last[1], "ask": last[2],
                             "asksize": last[3], "cumofi": cum0}[key]
                    vals = np.where(inside, vals, carry)
                out[key][sl] = vals
            filled = hi
        last = (pb[-1], qb[-1], pa[-1], qa[-1])
        cum0 = cum[-1]

    if last is not None and filled < n:   # after the final update the book stays as it was
        for key, v in (("bid", last[0]), ("bidsize", last[1]), ("ask", last[2]),
                       ("asksize", last[3]), ("cumofi", cum0)):
            out[key][filled:] = v
    chk["grid_points_without_state"] = int(np.isnan(out["bid"]).sum())
    return out, g, chk


def save_grid(path, s, g, chk):
    np.savez_compressed(path, t=g, **s, **{f"chk_{k}": v for k, v in chk.items()})


def load_grid(path):
    z = np.load(path)
    s = {k: z[k] for k in ("bid", "ask", "bidsize", "asksize", "cumofi")}
    chk = {k[4:]: int(z[k]) for k in z.files if k.startswith("chk_")}
    return s, z["t"], chk


def day_frame(s, g):
    """Same features and labels as the Nasdaq study (prices are already in ticks)."""
    fr = S.frame_from_sample(g, s, 1.0)
    fr["mid_bps_per_tick"] = 1e4 / fr["mid"]      # 1 tick in basis points at that moment
    return fr


# --------------------------------------------------------------------------- study

def cks_day(s, g, step=10, window=1800):
    """Mean R^2 of 10-second mid change (ticks) on 10-second OFI, one OLS per half hour."""
    idx = np.arange(step - 1, len(g), step)
    mid = (s["bid"][idx] + s["ask"][idx]) / 2.0
    cum = s["cumofi"][idx]
    dp, ofi = np.diff(mid), np.diff(cum)
    tend = g[idx][1:]
    win = ((tend - 1e-9) // window).astype(int)
    r2 = []
    for w in np.unique(win):
        m = (win == w) & np.isfinite(dp) & np.isfinite(ofi)
        if m.sum() > 30 and np.var(ofi[m]) > 0 and np.var(dp[m]) > 0:
            r2.append(S.ols(ofi[m], dp[m])[2])
    return float(np.mean(r2)) if r2 else float("nan")


def walk_forward_days(frames, feats, h, min_train=MIN_TRAIN_DAYS):
    """Expanding window by day: fit on all earlier days, test on the next one.

    Days are a month apart and labels never cross midnight, so no training
    label overlaps a test day. Clip bounds and trading thresholds come from the
    training days only.
    """
    rows = []
    for i in range(min_train, len(frames)):
        Xtr, ytr = [], []
        for fr in frames[:i]:
            X, y, ok = S._design(fr, feats, h)
            Xtr.append(X[ok]); ytr.append(y[ok])
        Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
        bounds = S.winsor_fit(Xtr)
        coef, _, _ = S.ols(S.winsor_apply(Xtr, bounds), ytr)
        fit = np.abs(coef[0] + S.winsor_apply(Xtr, bounds) @ coef[1:])
        X, y, ok = S._design(frames[i], feats, h)
        pred = coef[0] + S.winsor_apply(X[ok], bounds) @ coef[1:]
        fr = frames[i]
        rows.append({
            "i": i, "y": y[ok], "pred": pred, "thr": np.quantile(fit, S.COST_QUANTILES),
            "spread": fr["spread"][ok], "spread_exit": fr[f"spread_exit_{h}s"][ok],
            "bps_per_tick": fr["mid_bps_per_tick"][ok],
        })
    return rows


def day_bootstrap(stat_rows, reps=2000, seed=0):
    """95% interval for a pooled statistic by resampling test days."""
    rng = np.random.default_rng(seed)
    n = len(stat_rows)
    sse = np.array([r[0] for r in stat_rows]); sst = np.array([r[1] for r in stat_rows])
    draws = rng.integers(0, n, (reps, n))
    return np.percentile(1 - sse[draws].sum(1) / sst[draws].sum(1), [2.5, 97.5])


def taker_bps(rows):
    """Gross move captured vs spread and taker fees, basis points per round trip."""
    out = []
    for k, q in enumerate(S.COST_QUANTILES):
        g, sp, n = [], [], 0
        for r in rows:
            m = (np.abs(r["pred"]) >= r["thr"][k]) & (r["pred"] != 0)
            g.append(np.sign(r["pred"][m]) * r["y"][m] * r["bps_per_tick"][m])
            sp.append((r["spread"][m] + r["spread_exit"][m]) / 2 * r["bps_per_tick"][m])
            n += int(m.sum())
        g, sp = np.concatenate(g), np.concatenate(sp)
        out.append({"top_share": 1 - q, "trades": n, "gross_bps": float(g.mean()),
                    "spread_bps": float(sp.mean()), "taker_fees_bps": 2 * FEE_BPS["taker"],
                    "net_bps": float(g.mean() - sp.mean() - 2 * FEE_BPS["taker"])})
    return out


MODELS = {"Queue imbalance": ["qi"], "Lagged OFI": ["ofi_1s", "ofi_5s", "ofi_30s"], "Both": S.FEATURES,
          # added after the first full run showed forecasts 2.4 to 4 times too large on quiet days
          "Queue imbalance x recent activity": ["qi_x_activity"]}
QUIET_DAYS = ("2025-11-01", "2026-01-01", "2026-08-01")   # the days that motivated the fix


def crypto_study(frames, dates, grids):
    res = {"dates": [d.isoformat() for d in dates], "days": {}, "pred": {}, "costs": {}}
    for d, fr, (s, g, chk) in zip(dates, frames, grids):
        res["days"][d.isoformat()] = {
            **chk, "median_spread_ticks": float(np.nanmedian(fr["spread"])),
            "share_one_tick": float(np.nanmean(fr["spread"] == 1)),
            "tick_bps": float(np.nanmedian(fr["mid_bps_per_tick"])),
            "mean_depth": float(np.nanmean((s["bidsize"] + s["asksize"]) / 2)),
            "cks_r2": cks_day(s, g),
        }
    for h in S.HORIZONS:
        for name, feats in MODELS.items():
            rows = walk_forward_days(frames, feats, h)
            per_day = []
            for r in rows:
                sse = float(np.sum((r["y"] - r["pred"]) ** 2)); sst = float(np.sum(r["y"] ** 2))
                per_day.append((sse, sst, S.hit_rate(r["y"], r["pred"]), res["dates"][r["i"]]))
            sse, sst = sum(p[0] for p in per_day), sum(p[1] for p in per_day)
            res["pred"][f"{name}|{h}"] = {
                "r2": 1 - sse / sst, "r2_ci": list(day_bootstrap(per_day)),
                "by_day": [{"date": p[3], "r2": 1 - p[0] / p[1], "hit": p[2]} for p in per_day],
                "share_days_positive": float(np.mean([1 - p[0] / p[1] > 0 for p in per_day])),
                "hit": float(np.mean([p[2] for p in per_day])),
            }
            if name == "Both":
                res["costs"][h] = taker_bps(rows)
            # calibration: regression slope of realised move on forecast, per test day
            for r, bd in zip(rows, res["pred"][f"{name}|{h}"]["by_day"]):
                bd["slope"] = float(np.sum(r["y"] * r["pred"]) / np.sum(r["pred"] ** 2))
                bd["mean_abs_move"] = float(np.mean(np.abs(r["y"])))
            # pooled R^2 excluding the days that motivated the activity fix
            keep = [p for p in per_day if p[3] not in QUIET_DAYS]
            res["pred"][f"{name}|{h}"]["r2_other_days"] = 1 - sum(p[0] for p in keep) / sum(p[1] for p in keep)
    return res
