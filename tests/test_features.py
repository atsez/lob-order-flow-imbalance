"""Correctness of the OFI definition, the grid sampling and the no-lookahead rule."""
import numpy as np
import pandas as pd
import pytest

from lobofi import features as F
from lobofi import studies as S
from lobofi.io import OPEN


def test_ofi_hand_example():
    # bid price, bid size, ask price, ask size after each event
    pb = np.array([100, 100, 101, 101, 100])
    qb = np.array([10, 15, 4, 4, 20])
    pa = np.array([103, 103, 103, 102, 102])
    qa = np.array([8, 8, 8, 5, 9])
    # 1: bid queue +5 at same price                       -> +5
    # 2: bid improves to 101 with 4                         -> +4
    # 3: ask improves to 102 with 5 (new supply at better)  -> -5
    # 4: bid drops to 100 (old 4 gone, new queue not counted on the down move
    #    beyond removing the old one), ask queue +4          -> -4 - 4 = -8
    expected = np.array([0, 5, 4, -5, -8])
    assert np.array_equal(F.event_ofi(pb, qb, pa, qa), expected)


def test_ofi_buy_market_order_is_positive():
    # a buyer lifts 3 of the 8 shares at the ask
    e = F.event_ofi(np.array([100, 100]), np.array([5, 5]),
                    np.array([101, 101]), np.array([8, 5]))
    assert e[1] == 3


def test_asof_uses_events_at_or_before_t():
    times = np.array([1.0, 2.0, 2.0, 3.5])
    assert list(F.asof_index(times, np.array([1.0, 1.9, 2.0, 3.0, 3.5]))) == [0, 0, 2, 2, 3]


def test_window_and_forward_helpers():
    cum = np.array([0., 1., 3., 6., 10.])
    w = F.window_sum(cum, 2)
    assert np.isnan(w[:2]).all() and list(w[2:]) == [3., 5., 7.]
    f = F.forward_change(cum, 1)
    assert list(f[:-1]) == [1., 2., 3., 4.] and np.isnan(f[-1])
    m = F.causal_mean(np.array([1., 2., 3., 4.]), 2)
    assert np.isnan(m[0]) and list(m[1:]) == [1.5, 2.5, 3.5]


def synthetic(n=6000, seed=1):
    """A random but internally sensible level-1 path over the trading day."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(OPEN + 0.001, OPEN + 1200, n))
    t[1::7] = t[0::7][: len(t[1::7])]          # force shared timestamps
    t = np.sort(t)
    mid = 10_000 + 100 * np.cumsum(rng.choice([-1, 0, 0, 0, 1], n))
    half = 100 * rng.integers(1, 3, n)
    return pd.DataFrame({
        "time": t, "type": np.ones(n, dtype="int8"), "order_id": np.arange(n),
        "size": rng.integers(1, 300, n), "price": mid, "direction": np.ones(n, dtype="int8"),
        "ask1": mid + half, "asksize1": rng.integers(1, 500, n),
        "bid1": mid - half, "bidsize1": rng.integers(1, 500, n),
    })


def perturb_after(df, cut, seed=2):
    """Replace every event with time > cut by fresh random book states."""
    out = df.copy()
    m = out["time"] > cut
    rng = np.random.default_rng(seed)
    out.loc[m, "bidsize1"] = rng.integers(1, 500, m.sum())
    out.loc[m, "asksize1"] = rng.integers(1, 500, m.sum())
    out.loc[m, "bid1"] -= 100 * rng.integers(0, 3, m.sum())
    return out


@pytest.mark.parametrize("cut", [OPEN + 400.0, OPEN + 777.0])
def test_features_never_see_the_future(cut):
    df = synthetic()
    a, b = S.grid_frame(df), S.grid_frame(perturb_after(df, cut))
    known = a["t"] <= cut
    for f in S.FEATURES + ["spread", "mid"]:
        np.testing.assert_array_equal(a[f][known], b[f][known], err_msg=f)


@pytest.mark.parametrize("h", S.HORIZONS)
def test_labels_only_use_t_to_t_plus_h(h):
    cut = OPEN + 600.0
    df = synthetic()
    a, b = S.grid_frame(df), S.grid_frame(perturb_after(df, cut))
    known = a["t"] + h <= cut
    np.testing.assert_array_equal(a[f"y_{h}s"][known], b[f"y_{h}s"][known])


def test_leaky_pipeline_is_caught():
    """The same check flags the floor-binning mistake: its 1 s OFI at t moves
    when events in (t, t+1) change."""
    cut = OPEN + 600.0
    df = synthetic()
    a, b = S.leaky_grid_frame(df), S.leaky_grid_frame(perturb_after(df, cut))
    known = a["t"] <= cut
    assert not np.array_equal(a["ofi_1s"][known], b["ofi_1s"][known], equal_nan=True)


def test_walk_forward_embargo():
    df = synthetic(n=20000)
    df["time"] = np.sort(np.random.default_rng(3).uniform(OPEN + 0.001, OPEN + 3 * 1800, len(df)))
    fr = S.grid_frame(df)
    for h in S.HORIZONS:
        wf = S.walk_forward(fr, S.FEATURES, h)
        assert np.all(np.diff(wf["t"]) > 0)                     # time order, no duplicates
        assert wf["t"].min() >= OPEN + 2 * 1800                 # first two blocks are training only


def test_causal_mean_skips_missing_values():
    m = F.causal_mean(np.array([np.nan, 2., 4., np.nan, 6.]), 2)
    assert np.isnan(m[0]) and m[1] == 2. and m[2] == 3. and m[3] == 4. and m[4] == 6.


def test_activity_feature_is_causal():
    cut = OPEN + 600.0
    df = synthetic()
    a, b = S.grid_frame(df), S.grid_frame(perturb_after(df, cut))
    known = a["t"] <= cut
    for f in ("activity", "qi_x_activity"):
        np.testing.assert_array_equal(a[f][known], b[f][known], err_msg=f)
