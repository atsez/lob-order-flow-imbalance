"""Order-flow imbalance, queue imbalance and clock-grid sampling.

Order-flow imbalance (OFI) follows Cont, Kukanov and Stoikov (2014),
"The Price Impact of Order Book Events", Journal of Financial Econometrics.
For consecutive level-1 states n-1 -> n,

    e_n = 1{Pb_n >= Pb_n-1} qb_n - 1{Pb_n <= Pb_n-1} qb_n-1
        - 1{Pa_n <= Pa_n-1} qa_n + 1{Pa_n >= Pa_n-1} qa_n-1

so e_n is positive when demand at the bid grows or supply at the ask shrinks.
OFI over an interval is the sum of e_n over the events inside it.

Every grid feature at time t is built only from events with time <= t.
Labels are built only from events with time in (t, t + h]. The test suite
checks both properties directly.
"""
import numpy as np

from .io import TICK


def event_ofi(pb, qb, pa, qa):
    """e_n for every event (e_0 = 0), as int64."""
    e = np.zeros(len(pb), dtype=np.int64)
    db_up = pb[1:] >= pb[:-1]
    db_dn = pb[1:] <= pb[:-1]
    da_dn = pa[1:] <= pa[:-1]
    da_up = pa[1:] >= pa[:-1]
    e[1:] = (db_up * qb[1:] - db_dn * qb[:-1]
             - da_dn * qa[1:] + da_up * qa[:-1])
    return e


def asof_index(times, t):
    """Index of the last event with time <= t (-1 if none)."""
    return np.searchsorted(times, t, side="right") - 1


def grid(start, end, step):
    """Clock grid start, start+step, ... < end. Integer steps avoid float drift."""
    n = int(round((end - start) / step))
    return start + step * np.arange(n)


def sample(times, pb, qb, pa, qa, t):
    """Level-1 state and cumulative OFI as of each grid time t (time <= t)."""
    idx = asof_index(times, t)
    if np.any(idx < 0):
        raise ValueError("grid starts before the first event")
    cum = np.cumsum(event_ofi(pb, qb, pa, qa))
    return {
        "idx": idx,
        "bid": pb[idx], "bidsize": qb[idx], "ask": pa[idx], "asksize": qa[idx],
        "cumofi": cum[idx],
    }


def mid_ticks(s):
    return (s["bid"] + s["ask"]) / 2.0 / TICK


def spread_ticks(s):
    return (s["ask"] - s["bid"]) / TICK


def queue_imbalance(s):
    """(qb - qa) / (qb + qa), in [-1, 1]."""
    qb, qa = s["bidsize"].astype(float), s["asksize"].astype(float)
    return (qb - qa) / (qb + qa)


def window_sum(cum, lag):
    """Sum over the last `lag` grid steps, from a cumulative series on the grid.

    out[k] = cum[k] - cum[k - lag], i.e. events in (t_{k-lag}, t_k].
    The first `lag` entries are NaN.
    """
    out = np.full(len(cum), np.nan)
    out[lag:] = cum[lag:] - cum[:-lag]
    return out


def forward_change(x, h):
    """x[k + h] - x[k], i.e. the change over (t_k, t_{k+h}]. Last h are NaN."""
    out = np.full(len(x), np.nan)
    out[:-h] = x[h:] - x[:-h]
    return out


def causal_mean(x, lag):
    """Mean of x over grid points k-lag+1 .. k (inclusive), skipping missing
    values. First lag-1 points, and windows with nothing valid, are NaN."""
    x = x.astype(float)
    valid = np.isfinite(x)
    c = np.cumsum(np.insert(np.where(valid, x, 0.0), 0, 0.0))
    n = np.cumsum(np.insert(valid.astype(float), 0, 0.0))
    out = np.full(len(x), np.nan)
    cnt = n[lag:] - n[:-lag]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[lag - 1:] = np.where(cnt > 0, (c[lag:] - c[:-lag]) / cnt, np.nan)
    return out
