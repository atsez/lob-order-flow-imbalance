"""Queue-position simulator: hand-checked cases and invariants on real data."""
import numpy as np
import pandas as pd
import pytest

from lobofi import io, maker as M

COLS = ["time", "type", "order_id", "size", "price", "direction", "ask1", "asksize1", "bid1", "bidsize1"]


def frame(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df["time"] = 34200.0 + 0.001 * np.arange(len(df))
    return df


def bid_fills(res):
    return [(k, p, sh, a0) for k, s, p, sh, a0, _ in res.fills if s == M.BID]


def test_fifo_queue_and_fill():
    df = frame([
        # t  ty oid  size price dir  ask  asz  bid  bsz
        [0, 1, 1, 300, 100, 1, 101, 500, 100, 300],   # order 1 rests at the bid; we join behind it
        [0, 1, 3, 200, 100, 1, 101, 500, 100, 500],   # order 3 joins behind us
        [0, 3, 3, 200, 100, 1, 101, 500, 100, 300],   # order 3 cancels: behind us, queue ahead stays 300
        [0, 2, 1, 100, 100, 1, 101, 500, 100, 200],   # order 1 cancels 100: ahead 200
        [0, 4, 1, 150, 100, 1, 101, 500, 100, 50],    # execution 150: ahead 50
        [0, 1, 4, 400, 100, 1, 101, 500, 100, 450],   # order 4 joins behind us
        [0, 4, 1, 120, 100, 1, 101, 500, 100, 330],   # execution 120: 50 ahead, then 70 fill us
        [0, 4, 4, 100, 100, 1, 101, 500, 100, 230],   # execution 100: the last 30 fill us
        [0, 1, 5, 10, 100, 1, 101, 500, 100, 240],
    ])
    cfg = M.Config(start=34200.0, end=34201.0, max_pos=10_000)
    res = M.simulate(df, cfg, check_invariant=True)
    assert bid_fills(res) == [(6, 100, 70, 300), (7, 100, 30, 300)]
    assert res.pos == 100


def test_front_of_queue_fills_first():
    df = frame([
        [0, 1, 1, 300, 100, 1, 101, 500, 100, 300],
        [0, 4, 1, 120, 100, 1, 101, 500, 100, 180],
        [0, 1, 5, 10, 100, 1, 101, 500, 100, 190],
    ])
    res = M.simulate(df, M.Config(start=34200.0, end=34201.0, queue=False, max_pos=10_000))
    assert bid_fills(res)[0][:3] == (1, 100, 100)


@pytest.mark.parametrize("orphan,expected", [("fill", 100), ("cancel", 0)])
def test_level_emptied_with_us_last(orphan, expected):
    df = frame([
        [0, 1, 1, 300, 100, 1, 101, 500, 100, 300],
        [0, 3, 1, 300, 100, 1, 101, 500, 99, 200],    # everyone ahead cancels and the bid drops a tick
        [0, 1, 5, 10, 99, 1, 101, 500, 99, 210],
    ])
    res = M.simulate(df, M.Config(start=34200.0, end=34201.0, orphan=orphan, max_pos=10_000))
    assert sum(sh for _, p, sh, _ in bid_fills(res) if p == 100) == expected


def test_latency_longer_than_window_means_no_fills():
    df = io.load("INTC")
    res = M.simulate(df, M.Config(start=40000.0, end=40600.0, latency=1000.0))
    assert res.fills == []


@pytest.mark.parametrize("ticker", ["INTC", "MSFT"])
def test_queue_never_exceeds_displayed_size(ticker):
    df = io.load(ticker)
    M.simulate(df, M.Config(start=34500.0, end=57300.0), check_invariant=True)


def test_front_of_queue_overstates_fills():
    df = io.load("INTC")
    arr = M.arrival_index(df)
    w = dict(start=34500.0, end=46800.0, orphan="cancel")
    q = M.simulate(df, M.Config(**w), arr)
    f = M.simulate(df, M.Config(**w, queue=False), arr)
    assert sum(x[3] for x in f.fills) > sum(x[3] for x in q.fills)
