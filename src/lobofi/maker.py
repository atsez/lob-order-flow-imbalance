"""Queue-position market-making simulator on LOBSTER messages.

The question it answers: does queue imbalance help a market maker, who earns
the spread instead of paying it? A taker study cannot answer that, because a
maker's fills depend on where its order sits in the queue.

How the queue is tracked
------------------------
LOBSTER gives every visible order an ID, and Nasdaq fills orders at the same
price in time priority (first in, first out). When our order joins the best
bid after message j, every share displayed at that price is ahead of us. Then:

    execution at our price (type 4)   eats the front of the queue first: it
                                      shrinks the queue ahead of us, and any
                                      size beyond it fills our order
    cancel at our price (type 2, 3)   shrinks the queue ahead only if that order
                                      arrived at or before message j (orders
                                      resting before the open count as ahead)
    new order at our price (type 1)   joins behind us and changes nothing

Assumptions, all stated in the README:
    our order is small (100 shares) and does not change anyone else's behaviour;
    hidden executions (type 5) never fill us;
    if our price level empties while our order is still unfilled, we would be
    the last order at a price the market is leaving, so by default we count it
    as filled there (the cautious choice, since such fills are adverse); the
    optimistic alternative, cancelling, is reported as a sensitivity;
    an order sent at time t takes effect at t + latency, and a join is dropped
    if the best price has moved by then.

The front-of-queue mode removes the queue entirely (every execution at our
price fills us first). It is the shortcut many backtests take, and it is here
to measure how much it overstates.
"""
from dataclasses import dataclass, field

import numpy as np

from . import features as F
from .io import TICK, book_arrays

BID, ASK = 1, -1


@dataclass
class Config:
    start: float                 # seconds after midnight
    end: float
    size: int = 100              # shares per quote
    max_pos: int = 1000          # inventory limit, shares
    latency: float = 0.0         # seconds from decision to effect
    theta: float = None          # pull bid if QI < -theta, ask if QI > theta; None = never pull
    queue: bool = True           # False = front-of-queue shortcut
    orphan: str = "fill"         # level emptied with our order unfilled: "fill" (cautious) or "cancel"


@dataclass
class Result:
    fills: list = field(default_factory=list)   # (msg index, side, price, shares, queue ahead at join, mid before)
    cash: float = 0.0                           # in LOBSTER price units x shares
    pos: int = 0
    final_value: float = 0.0
    counts: dict = field(default_factory=dict)
    quoted_time: dict = field(default_factory=dict)


def arrival_index(df):
    """Message index at which each order ID was first added (type 1)."""
    ty = df["type"].to_numpy()
    oid = df["order_id"].to_numpy()
    idx = np.flatnonzero(ty == 1)
    first = {}
    for i, o in zip(idx.tolist(), oid[idx].tolist()):
        if o not in first:
            first[o] = i
    return first


class _Order:
    __slots__ = ("price", "ahead", "left", "joined", "ahead0")

    def __init__(self, price, ahead, left, joined):
        self.price, self.ahead, self.left, self.joined, self.ahead0 = price, ahead, left, joined, ahead


def simulate(df, cfg, arrivals=None, check_invariant=False):
    """Run one maker over [cfg.start, cfg.end). Returns a Result."""
    if arrivals is None:
        arrivals = arrival_index(df)
    t = df["time"].to_numpy()
    i0 = int(np.searchsorted(t, cfg.start, side="left"))
    i1 = int(np.searchsorted(t, cfg.end, side="left"))
    T = t.tolist()
    TY = df["type"].to_numpy().tolist()
    OID = df["order_id"].to_numpy().tolist()
    SZ = df["size"].to_numpy().tolist()
    PX = df["price"].to_numpy().tolist()
    DR = df["direction"].to_numpy().tolist()
    PB = df["bid1"].to_numpy().tolist()
    QB = df["bidsize1"].to_numpy().tolist()
    PA = df["ask1"].to_numpy().tolist()
    QA = df["asksize1"].to_numpy().tolist()

    res = Result()
    c = {"joins": 0, "joins_dropped": 0, "pulls": 0, "cancel_improved": 0, "orphaned": 0}
    live = {BID: None, ASK: None}
    pend_join = {BID: None, ASK: None}      # (effective time, price)
    pend_cancel = {BID: None, ASK: None}    # effective time
    quoted = {BID: 0.0, ASK: 0.0}
    arr_get = arrivals.get

    def best(side, k):
        return (PB[k], QB[k]) if side == BID else (PA[k], QA[k])

    def fill(side, o, shares, k):
        o.left -= shares
        res.pos += side * shares
        res.cash -= side * shares * o.price
        mid_before = (PB[k - 1] + PA[k - 1]) / 2.0
        res.fills.append((k, side, o.price, shares, o.ahead0, mid_before))

    for i in range(i0, i1):
        ti = T[i]
        prev = i - 1
        # 1. orders whose latency has elapsed take effect against the book before message i
        for side in (BID, ASK):
            pc = pend_cancel[side]
            if pc is not None and pc <= ti:
                live[side] = None
                pend_cancel[side] = None
            pj = pend_join[side]
            if pj is not None and pj[0] <= ti:
                pend_join[side] = None
                px, qty = best(side, prev)
                if px == pj[1]:
                    live[side] = _Order(px, qty if cfg.queue else 0, cfg.size, prev)
                    c["joins"] += 1
                else:
                    c["joins_dropped"] += 1

        # 2. message i acts on our resting orders
        ty, dr, px = TY[i], DR[i], PX[i]
        o = live[BID] if dr == 1 else live[ASK]
        if o is not None and px == o.price:
            side = BID if dr == 1 else ASK
            if ty == 4:
                sz = SZ[i]
                if sz <= o.ahead:
                    o.ahead -= sz
                else:
                    excess = sz - o.ahead
                    o.ahead = 0
                    fill(side, o, min(excess, o.left), i)
                    if o.left == 0:
                        live[side] = None
                        pend_cancel[side] = None
            elif (ty == 2 or ty == 3) and cfg.queue:
                if arr_get(OID[i], -1) <= o.joined:
                    o.ahead = max(o.ahead - SZ[i], 0)

        # 3. book after message i: is each order still at the best price?
        for side in (BID, ASK):
            o = live[side]
            if o is None:
                continue
            bpx, bqty = best(side, i)
            if bpx != o.price:
                better = (bpx > o.price) if side == BID else (bpx < o.price)
                c["cancel_improved" if better else "orphaned"] += 1
                if not better and cfg.orphan == "fill":
                    fill(side, o, o.left, i)
                live[side] = None
                pend_cancel[side] = None
            elif check_invariant and cfg.queue:
                assert 0 <= o.ahead <= bqty, (i, side, o.ahead, bqty)

        # 4. decide what we want quoted after message i
        qb, qa = QB[i], QA[i]
        qi = (qb - qa) / (qb + qa)
        for side in (BID, ASK):
            want = True
            if side == BID and res.pos + cfg.size > cfg.max_pos:
                want = False
            if side == ASK and res.pos - cfg.size < -cfg.max_pos:
                want = False
            if cfg.theta is not None and want:
                if (side == BID and qi < -cfg.theta) or (side == ASK and qi > cfg.theta):
                    want = False
            o = live[side]
            if want:
                if o is None and pend_join[side] is None:
                    pend_join[side] = (ti + cfg.latency, best(side, i)[0])
            else:
                if pend_join[side] is not None:
                    pend_join[side] = None
                if o is not None and pend_cancel[side] is None:
                    pend_cancel[side] = ti + cfg.latency
                    c["pulls"] += 1

        # time quoted, measured until the next message
        if i + 1 < len(T):
            dt = T[i + 1] - ti
            for side in (BID, ASK):
                if live[side] is not None:
                    quoted[side] += dt

    # flatten at the end of the window by crossing the spread
    last = i1 - 1
    if res.pos > 0:
        res.final_value = res.cash + res.pos * PB[last]
    else:
        res.final_value = res.cash + res.pos * PA[last]
    res.counts = c
    span = T[last] - T[i0]
    res.quoted_time = {"bid": quoted[BID] / span, "ask": quoted[ASK] / span}
    return res


def markouts(df, res, h):
    """Per-fill value of each fill marked to the mid h seconds later, in price units per share."""
    f = np.array(res.fills, dtype=float)
    k, side, price = f[:, 0].astype(int), f[:, 1], f[:, 2]
    times, pb, qb, pa, qa = book_arrays(df)
    j = F.asof_index(times, times[k] + h)
    return side * ((pb[j] + pa[j]) / 2.0 - price)


def bootstrap_markout(df, res, h, bucket=300.0, reps=2000, seed=0):
    """95% interval for the share-weighted markout per share, resampling
    5-minute buckets of fills so that clustered fills stay together."""
    f = np.array(res.fills, dtype=float)
    times = df["time"].to_numpy()[f[:, 0].astype(int)]
    shares = f[:, 3]
    val = markouts(df, res, h) * shares
    b = ((times - times.min()) // bucket).astype(int)
    nb = b.max() + 1
    v_b = np.bincount(b, val, nb)
    s_b = np.bincount(b, shares, nb)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, nb, (reps, nb))
    stat = v_b[draws].sum(1) / s_b[draws].sum(1) / TICK
    return np.percentile(stat, [2.5, 97.5])


def summarise(df, res, horizons=(1, 5, 30, 60)):
    """Per-share fill quality in ticks and P&L in dollars."""
    if not res.fills:
        return {"fills": 0, "shares": 0, "pnl_usd": 0.0}
    f = np.array(res.fills, dtype=float)
    k, side, price, shares, ahead0, mid0 = f.T
    k = k.astype(int)
    times, pb, qb, pa, qa = book_arrays(df)
    w = shares / shares.sum()
    out = {
        "fills": len(f), "shares": int(shares.sum()),
        "edge_at_fill": float(np.sum(w * side * (mid0 - price)) / TICK),
        "mean_queue_ahead": float(np.sum(w * ahead0)),
        "pnl_usd": res.final_value / 10_000.0,
        "end_pos": res.pos,
    }
    out["pnl_per_share_ticks"] = res.final_value / out["shares"] / TICK
    for h in horizons:
        out[f"markout_{h}s"] = float(np.sum(w * markouts(df, res, h)) / TICK)
    out.update({f"n_{a}": b for a, b in res.counts.items()})
    out["quoted_bid"], out["quoted_ask"] = res.quoted_time["bid"], res.quoted_time["ask"]
    return out


# --------------------------------------------------------------------------- study

MORNING = (34200 + 300.0, 46800.0)    # 09:35 to 13:00, used to choose the threshold
AFTERNOON = (46800.0, 57600 - 300.0)  # 13:00 to 15:55, held out
THETAS = (None, 0.9, 0.8, 0.7, 0.6, 0.5)
LATENCIES = (0.0, 0.0005, 0.005, 0.05)
SELECT_H = 30


def maker_study(df):
    """Baseline queue model comparison, threshold choice on the morning,
    evaluation on the afternoon, and latency sensitivity."""
    arr = arrival_index(df)
    full = (MORNING[0], AFTERNOON[1])
    out = {"baseline": {}, "sweep": {}, "latency": {}, "afternoon_front": None}

    for name, kw in {"queue": {}, "front": {"queue": False},
                     "queue_orphan_cancel": {"orphan": "cancel"}}.items():
        r = simulate(df, Config(*full, **kw), arr)
        out["baseline"][name] = summarise(df, r)

    for th in THETAS:
        key = "off" if th is None else f"{th:.1f}"
        row = {}
        for wname, w in (("morning", MORNING), ("afternoon", AFTERNOON)):
            r = simulate(df, Config(*w, theta=th), arr)
            sm = summarise(df, r)
            sm["ci_select"] = list(bootstrap_markout(df, r, SELECT_H))
            row[wname] = sm
        out["sweep"][key] = row

    out["afternoon_front"] = summarise(df, simulate(df, Config(*AFTERNOON, queue=False), arr))
    pulled = [k for k in out["sweep"] if k != "off"]
    best = max(pulled, key=lambda k: out["sweep"][k]["morning"][f"markout_{SELECT_H}s"])
    out["chosen_theta"] = best
    for lat in LATENCIES:
        r = simulate(df, Config(*AFTERNOON, theta=float(best), latency=lat), arr)
        out["latency"][f"{lat * 1000:g}"] = summarise(df, r)
    return out
