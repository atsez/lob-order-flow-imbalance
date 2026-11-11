"""Loading LOBSTER level-1 data.

Each row is one Nasdaq message plus the best bid and ask *after* that message.
LOBSTER conventions (see the LOBSTER sample-file documentation):

    time       seconds after midnight, nanosecond resolution
    type       1 new limit order, 2 partial cancel, 3 full delete,
               4 execution of a visible order, 5 execution of a hidden order,
               6 cross trade, 7 trading halt indicator
    direction  1 = buy limit order (bid side), -1 = sell limit order (ask side).
               For executions it is the side of the resting order, so
               direction 1 on a type 4 means a seller hit the bid.
    prices     dollars x 10,000, so one cent (one tick) = 100
    empty side ask 9999999999, bid -9999999999
"""
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
TICK = 100                      # one cent in LOBSTER price units
OPEN, CLOSE = 34200.0, 57600.0  # 09:30 and 16:00 in seconds after midnight
EMPTY_ASK, EMPTY_BID = 9999999999, -9999999999

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DTYPES = {
    "time": "float64", "type": "int8", "order_id": "int64", "size": "int64",
    "price": "int64", "direction": "int8",
    "ask1": "int64", "asksize1": "int64", "bid1": "int64", "bidsize1": "int64",
}


def path_for(ticker, data_dir=DATA_DIR):
    return Path(data_dir) / f"LOBSTER_{ticker}_compact.csv.gz"


def load(ticker, data_dir=DATA_DIR):
    """Return the raw message + level-1 book frame, in file order."""
    # round_trip parsing gives the same doubles as C's strtod, so the Python
    # and C++ engines agree exactly on every timestamp
    df = pd.read_csv(path_for(ticker, data_dir), dtype=DTYPES, float_precision="round_trip")
    df.attrs["ticker"] = ticker
    return df


def book_arrays(df):
    """Best bid/ask prices and sizes as int64 numpy arrays, plus times."""
    return (df["time"].to_numpy(), df["bid1"].to_numpy(), df["bidsize1"].to_numpy(),
            df["ask1"].to_numpy(), df["asksize1"].to_numpy())
