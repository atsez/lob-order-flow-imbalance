"""Binance pipeline, tested on synthetic files in Tardis's exact 'quotes' format.

The synthetic generator plants a known effect (queue imbalance pushes the next
price move), so the end-to-end test checks the pipeline finds it.
"""
import datetime as dt
import gzip

import numpy as np
import pandas as pd
import pytest

from lobofi import crypto as C
from lobofi import features as F
from lobofi import studies as S

HEADER = ["exchange", "symbol", "timestamp", "local_timestamp", "ask_amount", "ask_price", "bid_price", "bid_amount"]


def synthetic_day(path, day, n=60_000, seed=0, strength=0.4, tick=0.1):
    """A full UTC day of top-of-book updates. Each update either resizes a queue
    or moves the price one tick, up with probability 0.5 + strength * QI."""
    rng = np.random.default_rng(seed)
    day0 = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()
    t = np.sort(rng.uniform(0.5, 86_399.5, n))
    bid, qb, qa = 600_000, 3.0, 3.0           # bid in ticks: 60,000.0 USDT
    rows = []
    for k in range(n):
        qi = (qb - qa) / (qb + qa)
        if rng.random() < 0.3:
            bid += 1 if rng.random() < 0.5 + strength * qi else -1
            qb, qa = rng.exponential(3) + 0.001, rng.exponential(3) + 0.001
        else:
            if rng.random() < 0.5:
                qb = max(0.001, qb + rng.normal(0, 1))
            else:
                qa = max(0.001, qa + rng.normal(0, 1))
        ask = bid + 1
        us = int((day0 + t[k]) * 1e6)
        rows.append(["binance-futures", "BTCUSDT", us - int(rng.integers(0, 3000)), us,
                     round(qa, 3), round(ask * tick, 1), round(bid * tick, 1), round(qb, 3)])
    df = pd.DataFrame(rows, columns=HEADER)
    df.loc[5, ["bid_price", "bid_amount"]] = np.nan          # an empty side, must be dropped
    df.to_csv(path, index=False, compression="gzip")
    return df


DAY = dt.date(2026, 3, 1)


@pytest.fixture(scope="module")
def one_day(tmp_path_factory):
    p = tmp_path_factory.mktemp("c") / "d.csv.gz"
    raw = synthetic_day(p, DAY)
    return p, raw


def test_month_starts_and_url():
    d = C.month_starts("2026-10", 24)
    assert len(d) == 24 and d[0] == dt.date(2024, 11, 1) and d[-1] == dt.date(2026, 10, 1)
    assert C.url_for(dt.date(2026, 3, 1), "BTCUSDT") == \
        "https://datasets.tardis.dev/v1/binance-futures/quotes/2026/03/01/BTCUSDT.csv.gz"


def test_off_grid_prices_are_refused():
    with pytest.raises(ValueError):
        C.to_ticks(np.array([60000.0, 60000.05]), 0.1)


def test_chunked_equals_in_memory(one_day):
    p, raw = one_day
    a, g, chk = C.process_day(p, DAY, 0.1, chunksize=997)
    b, _, _ = C.process_day(p, DAY, 0.1, chunksize=10_000_000)
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=0, atol=1e-6, equal_nan=True, err_msg=k)
    # and against the Nasdaq code path on the same rows
    ok = raw[["bid_price", "bid_amount"]].notna().all(axis=1)
    r = raw[ok]
    day0 = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc).timestamp()
    t = r["local_timestamp"].to_numpy() / 1e6 - day0
    ref = F.sample(t, C.to_ticks(r["bid_price"].to_numpy(), 0.1), r["bid_amount"].to_numpy(),
                   C.to_ticks(r["ask_price"].to_numpy(), 0.1), r["ask_amount"].to_numpy(), g[g > t[0]])
    m = g > t[0]
    for k in ("bid", "ask", "bidsize", "asksize", "cumofi"):
        np.testing.assert_allclose(a[k][m], ref[k], atol=1e-6, err_msg=k)
    assert chk["dropped_empty_side"] == 1 and chk["dropped_crossed"] == 0


def test_features_never_see_the_future(one_day, tmp_path):
    p, raw = one_day
    cut_us = int((dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc).timestamp() + 40_000) * 1e6)
    pert = raw.copy()
    late = pert["local_timestamp"] > cut_us
    pert.loc[late, "bid_amount"] = pert.loc[late, "bid_amount"] * 3 + 1
    p2 = tmp_path / "p.csv.gz"
    pert.to_csv(p2, index=False, compression="gzip")
    fa = C.day_frame(*C.process_day(p, DAY, 0.1)[:2])
    fb = C.day_frame(*C.process_day(p2, DAY, 0.1)[:2])
    known = fa["t"] <= 40_000
    for f in S.FEATURES + ["spread", "mid"]:
        np.testing.assert_array_equal(fa[f][known], fb[f][known], err_msg=f)
    for h in S.HORIZONS:
        k2 = fa["t"] + h <= 40_000
        np.testing.assert_array_equal(fa[f"y_{h}s"][k2], fb[f"y_{h}s"][k2])


def test_end_to_end_finds_a_planted_signal(tmp_path):
    dates = [dt.date(2026, m, 1) for m in range(1, 6)]
    frames, grids = [], []
    for i, d in enumerate(dates):
        p = tmp_path / f"{d}.csv.gz"
        synthetic_day(p, d, n=40_000, seed=i)
        s, g, chk = C.process_day(p, d, 0.1)
        frames.append(C.day_frame(s, g)); grids.append((s, g, chk))
    res = C.crypto_study(frames, dates, grids)
    qi = res["pred"]["Queue imbalance|1"]
    assert len(qi["by_day"]) == len(dates) - C.MIN_TRAIN_DAYS
    assert qi["r2"] > 0 and qi["hit"] > 0.55
    assert all(r["net_bps"] < 0 for r in res["costs"][5])   # fees dwarf a one-tick edge
