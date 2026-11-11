"""Data integrity checks, run before any research.

The key check replays every message against the level-1 book: a message that
touches the best price must move the best size (or the best price) by exactly
the message size, and a message away from the best must leave level 1 alone.
If the book columns and the messages disagree, every feature built on them is
suspect, so this has to pass first.
"""
import numpy as np

from .io import EMPTY_ASK, EMPTY_BID, OPEN, CLOSE


def l1_consistency(df):
    """Fraction of messages whose effect on level 1 matches the message.

    Returns (n_checked, n_consistent, per-type counts of inconsistencies).
    Type 6 (cross), 7 (halt) and hidden executions (5) must leave level 1
    unchanged; the rest are replayed against the previous level-1 state.
    """
    t = df["type"].to_numpy()
    d = df["direction"].to_numpy()
    p = df["price"].to_numpy()
    s = df["size"].to_numpy()
    pb, qb = df["bid1"].to_numpy(), df["bidsize1"].to_numpy()
    pa, qa = df["ask1"].to_numpy(), df["asksize1"].to_numpy()

    ok = np.zeros(len(df) - 1, dtype=bool)
    for i in range(1, len(df)):
        ppb, pqb, ppa, pqa = pb[i - 1], qb[i - 1], pa[i - 1], qa[i - 1]
        same = pb[i] == ppb and qb[i] == pqb and pa[i] == ppa and qa[i] == pqa
        ty, side, price, size = t[i], d[i], p[i], s[i]

        if ty in (5, 6, 7):
            ok[i - 1] = same
            continue

        if side == 1:   # bid side
            best, bsize, nbest, nsize = ppb, pqb, pb[i], qb[i]
            other_same = pa[i] == ppa and qa[i] == pqa
            better = price > best
        else:           # ask side
            best, bsize, nbest, nsize = ppa, pqa, pa[i], qa[i]
            other_same = pb[i] == ppb and qb[i] == pqb
            better = price < best

        if ty == 1:
            if better:
                good = nbest == price and nsize == size
            elif price == best:
                good = nbest == best and nsize == bsize + size
            else:
                good = nbest == best and nsize == bsize
        else:  # 2 partial cancel, 3 delete, 4 visible execution
            if price == best:
                if size < bsize:
                    good = nbest == best and nsize == bsize - size
                else:  # level emptied, a new best price appears
                    good = nbest != best
            else:
                good = nbest == best and nsize == bsize
        ok[i - 1] = good and other_same

    bad_types = {}
    for ty in np.unique(t[1:][~ok]):
        bad_types[int(ty)] = int(np.sum(t[1:][~ok] == ty))
    return len(ok), int(ok.sum()), bad_types


def summary(df):
    """Basic facts and sanity checks for one ticker."""
    time = df["time"].to_numpy()
    pb, pa = df["bid1"].to_numpy(), df["ask1"].to_numpy()
    both = (pb != EMPTY_BID) & (pa != EMPTY_ASK)
    n, n_ok, bad = l1_consistency(df)
    uniq = np.unique(time)
    return {
        "messages": len(df),
        "time_monotone": bool(np.all(np.diff(time) >= 0)),
        "within_session": bool(time.min() >= OPEN and time.max() <= CLOSE),
        "crossed_or_locked": int(np.sum(both & (pa <= pb))),
        "empty_side_rows": int(np.sum(~both)),
        "halts": int(np.sum(df["type"].to_numpy() == 7)),
        "distinct_timestamps": len(uniq),
        "rows_sharing_a_timestamp": int(len(time) - len(uniq)),
        "l1_consistent_share": n_ok / n,
        "l1_inconsistent_by_type": bad,
    }
