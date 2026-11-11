"""The C++ engine must agree exactly with the Python reference on real data."""
import gzip
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lobofi import features as F
from lobofi import io

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "cpp" / "ofi_engine"


@pytest.fixture(scope="module", autouse=True)
def build():
    subprocess.run(["make", "-C", str(ROOT / "cpp")], check=True, capture_output=True)


def run_engine(ticker, *args):
    raw = gzip.decompress(io.path_for(ticker).read_bytes())
    res = subprocess.run([str(ENGINE), *args], input=raw, capture_output=True, check=True)
    return res.stdout.decode()


@pytest.mark.parametrize("ticker", io.TICKERS)
def test_event_ofi_matches_python(ticker):
    df = io.load(ticker)
    py = F.event_ofi(*io.book_arrays(df)[1:])
    cpp = np.array(run_engine(ticker, "events").split(), dtype=np.int64)
    assert len(cpp) == len(py)
    assert np.array_equal(cpp, py)


@pytest.mark.parametrize("ticker", ["AMZN", "INTC"])
def test_grid_matches_python(ticker):
    df = io.load(ticker)
    start, end, step = io.OPEN + 1, io.CLOSE + 0.5, 1.0
    g = F.grid(start, end, step)
    s = F.sample(*io.book_arrays(df), g)
    lines = run_engine(ticker, "grid", str(start), str(end), str(step)).strip().split("\n")[1:]
    arr = np.array([l.split(",") for l in lines])
    assert len(arr) == len(g)
    np.testing.assert_array_equal(arr[:, 1].astype(np.int64), s["idx"])
    for col, key in zip(range(2, 7), ["bid", "bidsize", "ask", "asksize", "cumofi"]):
        np.testing.assert_array_equal(arr[:, col].astype(np.int64), s[key], err_msg=key)
