"""The three studies.

1. replicate_cks   Contemporaneous OFI regression, set up as in Cont, Kukanov
                   and Stoikov (2014): 10-second grid, mid-price change in
                   ticks, one regression per half-hour window.
2. predictive      Does OFI or queue imbalance measured up to t predict the
                   mid-price change over (t, t+h]? Walk-forward by half-hour
                   block with an embargo, block-bootstrap confidence intervals,
                   and a deliberately leaky pipeline for comparison.
3. costs           Would a taker who trades on the forecast cover the spread?
"""
import numpy as np

from . import features as F
from .io import OPEN, CLOSE, TICK, book_arrays

HALF_HOUR = 1800.0


# --------------------------------------------------------------------------- utils

def ols(X, y):
    """OLS with intercept. Returns coefficients (intercept first), t-stats, R^2."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X1.T @ X1)
    tstat = beta / np.sqrt(np.diag(cov))
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
    return beta, tstat, r2


def trade_imbalance_events(df):
    """Signed executed volume per event: + buyer-initiated, - seller-initiated.

    For executions LOBSTER's direction is the side of the resting order, so a
    resting buy order (direction 1) being executed is a seller-initiated trade.
    Visible (4) and hidden (5) executions both count.
    """
    ty = df["type"].to_numpy()
    is_trade = (ty == 4) | (ty == 5)
    return np.where(is_trade, -df["direction"].to_numpy() * df["size"].to_numpy(), 0)


# --------------------------------------------------------------------------- study 1

def replicate_cks(df, step=10.0, window=HALF_HOUR):
    times, pb, qb, pa, qa = book_arrays(df)
    g = F.grid(OPEN + step, CLOSE + step / 2, step)     # 34210, 34220, ..., 57600
    s = F.sample(times, pb, qb, pa, qa, g)
    mid = F.mid_ticks(s)
    cum_ti = np.cumsum(trade_imbalance_events(df))[s["idx"]]
    depth = (s["bidsize"] + s["asksize"]) / 2.0

    # interval k is (g[k-1], g[k]]
    dp = np.diff(mid)
    ofi = np.diff(s["cumofi"]).astype(float)
    ti = np.diff(cum_ti).astype(float)
    t_end = g[1:]
    win = ((t_end - OPEN - 1e-9) // window).astype(int)

    rows = []
    for w in np.unique(win):
        m = win == w
        b, tb, r2 = ols(ofi[m], dp[m])
        _, _, r2_ti = ols(ti[m], dp[m])
        rows.append({
            "window": int(w), "n": int(m.sum()), "beta": b[1], "t_beta": tb[1],
            "r2_ofi": r2, "r2_ti": r2_ti, "avg_depth": depth[1:][m].mean(),
        })
    return rows


def depth_exponent(all_rows):
    """Pooled log beta = a_stock - lambda log AD, with stock fixed effects."""
    stocks = sorted(all_rows)
    y, X = [], []
    for j, st in enumerate(stocks):
        for r in all_rows[st]:
            if r["beta"] > 0:
                dummies = [1.0 if k == j else 0.0 for k in range(1, len(stocks))]
                y.append(np.log(r["beta"]))
                X.append([np.log(r["avg_depth"])] + dummies)
    b, t, _ = ols(np.array(X), np.array(y))
    lam = -b[1]
    se = abs(b[1] / t[1])
    return lam, se, len(y)


# --------------------------------------------------------------------------- study 2

FEATURES = ["ofi_1s", "ofi_5s", "ofi_30s", "qi"]
COST_QUANTILES = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99)
HORIZONS = [1, 5, 30]
DEPTH_LOOKBACK = 300  # seconds, causal


def grid_frame(df, step=1.0):
    """1-second grid with causal features and forward labels (in ticks)."""
    times, pb, qb, pa, qa = book_arrays(df)
    g = F.grid(OPEN + step, CLOSE + step / 2, step)
    s = F.sample(times, pb, qb, pa, qa, g)
    return frame_from_sample(g, s, TICK)


def frame_from_sample(g, s, tick):
    """Features and labels from level-1 states sampled on a 1-second grid.

    Shared by the Nasdaq and the Binance studies, so both markets are measured
    with exactly the same code. `tick` is the tick size in the units of s's prices.
    """
    depth = (s["bidsize"] + s["asksize"]) / 2.0
    norm = F.causal_mean(depth, DEPTH_LOOKBACK)
    mid = (s["bid"] + s["ask"]) / 2.0 / tick
    cum = s["cumofi"].astype(float)
    qb, qa = s["bidsize"].astype(float), s["asksize"].astype(float)
    out = {
        "t": g,
        "ofi_1s": F.window_sum(cum, 1) / norm,
        "ofi_5s": F.window_sum(cum, 5) / norm,
        "ofi_30s": F.window_sum(cum, 30) / norm,
        "qi": (qb - qa) / (qb + qa),
        "spread": (s["ask"] - s["bid"]) / tick,
        "mid": mid,
        "depth_norm": norm,
    }
    # recent activity: mean absolute 1-second mid change over the last 5 minutes,
    # using only changes that ended at or before t
    step_move = np.abs(np.diff(mid, prepend=np.nan))
    out["activity"] = F.causal_mean(step_move, DEPTH_LOOKBACK)
    out["qi_x_activity"] = out["qi"] * out["activity"]
    for h in HORIZONS:
        out[f"y_{h}s"] = F.forward_change(mid, h)
        # spread prevailing at t + h, when the taker exits
        out[f"spread_exit_{h}s"] = F.forward_change(out["spread"], h) + out["spread"]
    return out


def leaky_grid_frame(df, step=1.0):
    """The common mistake: bin events by floor(time) and label the bin by its
    left edge, as a default pandas resample does. The OFI 'at t' then contains
    events from (t, t+1s), which is the future relative to t."""
    times, pb, qb, pa, qa = book_arrays(df)
    good = grid_frame(df, step)
    cum = np.cumsum(F.event_ofi(pb, qb, pa, qa)).astype(float)
    g = good["t"]
    hi = np.searchsorted(times, g + step, side="left") - 1  # last event before t+1
    lo = np.searchsorted(times, g, side="left") - 1         # last event before t
    bad = dict(good)
    bad["ofi_1s"] = (cum[hi] - cum[lo]) / good["depth_norm"]
    return bad


def winsor_fit(X, q=0.005):
    """Clip bounds from the training rows only, so the test set never informs them."""
    return np.quantile(X, q, axis=0), np.quantile(X, 1 - q, axis=0)


def winsor_apply(X, bounds):
    return np.clip(X, bounds[0], bounds[1])


def _design(fr, feats, h):
    X = np.column_stack([fr[f] for f in feats])
    y = fr[f"y_{h}s"]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X, y, ok


def walk_forward(fr, feats, h, first_test_block=2):
    """Expanding-window walk-forward over half-hour blocks.

    Train on every block before the test block, minus an embargo: training
    rows whose label window (t, t+h] reaches into the test block are dropped.
    Features are clipped at the 0.5% and 99.5% quantiles of the training rows,
    because one outsized order can otherwise dominate a linear fit.
    Returns arrays of (t, y, prediction, spread_entry, spread_exit) for all
    out-of-sample rows, in time order.
    """
    X, y, ok = _design(fr, feats, h)
    t = fr["t"]
    block = ((t - OPEN - 1e-9) // HALF_HOUR).astype(int)
    out = {k: [] for k in ("t", "y", "pred", "spread", "spread_exit", "thr")}
    for b in range(first_test_block, block.max() + 1):
        test = ok & (block == b)
        test_start = OPEN + b * HALF_HOUR
        train = ok & (block < b) & (t + h <= test_start)
        if test.sum() == 0 or train.sum() < 100:
            continue
        bounds = winsor_fit(X[train])
        coef, _, _ = ols(winsor_apply(X[train], bounds), y[train])
        pred = coef[0] + winsor_apply(X[test], bounds) @ coef[1:]
        # trading thresholds come from the in-sample forecasts, so a "top 1%"
        # rule is something a trader could have set before the test block
        fit = np.abs(coef[0] + winsor_apply(X[train], bounds) @ coef[1:])
        thr = np.quantile(fit, COST_QUANTILES)
        out["thr"].append(np.tile(thr, (test.sum(), 1)))
        out["t"].append(t[test]); out["y"].append(y[test]); out["pred"].append(pred)
        out["spread"].append(fr["spread"][test])
        out["spread_exit"].append(fr[f"spread_exit_{h}s"][test])
    return {k: np.concatenate(v) for k, v in out.items()}


def oos_r2(y, pred):
    """1 - SSE / sum(y^2): skill against a forecast of 'no change'."""
    return 1 - np.sum((y - pred) ** 2) / np.sum(y ** 2)


def hit_rate(y, pred):
    m = (y != 0) & (pred != 0)
    return np.mean(np.sign(y[m]) == np.sign(pred[m]))


def block_bootstrap(stat, arrays, block=300, reps=2000, seed=0):
    """Moving-block bootstrap CI (2.5%, 97.5%) for stat(*arrays).

    Blocks of `block` consecutive grid points (5 minutes at 1 s) keep the
    autocorrelation from overlapping labels inside each resample.
    """
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    nb = int(np.ceil(n / block))
    vals = np.empty(reps)
    for r in range(reps):
        starts = rng.integers(0, n - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        vals[r] = stat(*[a[idx] for a in arrays])
    return np.percentile(vals, [2.5, 97.5])


def transfer(frames, feats, h, split_block=7):
    """Fit on stock A's morning, test on stock B's afternoon.

    Splitting in time as well as across stocks matters: the five stocks move
    together, so fitting on A and testing on B over the same minutes would
    leak common market moves into the test.
    """
    names = sorted(frames)
    mat = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        Xa, ya, oka = _design(frames[a], feats, h)
        ta = frames[a]["t"]
        split = OPEN + split_block * HALF_HOUR
        tr = oka & (ta + h <= split)
        bounds = winsor_fit(Xa[tr])
        coef, _, _ = ols(winsor_apply(Xa[tr], bounds), ya[tr])
        for j, b in enumerate(names):
            Xb, yb, okb = _design(frames[b], feats, h)
            te = okb & (frames[b]["t"] >= split)
            mat[i, j] = oos_r2(yb[te], coef[0] + winsor_apply(Xb[te], bounds) @ coef[1:])
    return names, mat


# --------------------------------------------------------------------------- study 3

def taker_costs(wf):
    """Trade sign(pred) at t when |pred| clears a threshold set from the
    in-sample forecasts, crossing the spread on entry at t and on exit at t+h.
    All values in ticks per trade."""
    y, pred = wf["y"], wf["pred"]
    cost = (wf["spread"] + wf["spread_exit"]) / 2.0
    gross_all = np.sign(pred) * y
    rows = []
    for k, q in enumerate(COST_QUANTILES):
        m = (np.abs(pred) >= wf["thr"][:, k]) & (pred != 0)
        g, c = gross_all[m], cost[m]
        rows.append({
            "top_share": 1 - q, "trades": int(m.sum()),
            "gross": g.mean(), "cost": c.mean(), "net": (g - c).mean(),
            "gross_over_cost": g.mean() / c.mean(),
        })
    return rows
