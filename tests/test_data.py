"""The data must pass the integrity checks before any study is trusted."""
import pytest

from lobofi import checks, io


@pytest.mark.parametrize("ticker", io.TICKERS)
def test_level1_replays_every_message(ticker):
    s = checks.summary(io.load(ticker))
    assert s["time_monotone"] and s["within_session"]
    assert s["crossed_or_locked"] == 0 and s["empty_side_rows"] == 0
    assert s["l1_consistent_share"] == 1.0, s["l1_inconsistent_by_type"]
