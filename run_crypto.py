"""Binance USD-M futures study on the free first-of-month days from Tardis.dev.

    python run_crypto.py                       # BTCUSDT, 24 months to 2026-10
    python run_crypto.py --months 6            # a quicker first run
    python run_crypto.py --symbol ETHUSDT

For each day it downloads the quotes file, streams it into a 1-second grid
(data/crypto/grid/), deletes the raw file unless --keep-raw, then runs the
study across days and writes results/crypto_results.md, results/crypto_results.json
and figures/6_crypto_by_month.png. Days already on the grid are not downloaded
again, so an interrupted run picks up where it stopped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lobofi import crypto as C  # noqa: E402
from lobofi import crypto_report  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT", choices=sorted(C.TICKS))
    ap.add_argument("--end", default="2026-10", help="last month, YYYY-MM")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--keep-raw", action="store_true", help="keep the downloaded .csv.gz files")
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "crypto" / "raw"),
                    help="where raw files are downloaded, or already sit")
    ap.add_argument("--no-download", action="store_true", help="use files already in --raw-dir")
    a = ap.parse_args(argv)

    dates = C.month_starts(a.end, a.months)
    tick = C.TICKS[a.symbol]
    grid_dir = ROOT / "data" / "crypto" / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    frames, grids = [], []
    for d in dates:
        gpath = grid_dir / f"{a.symbol}_{d.isoformat()}.npz"
        if not gpath.exists():
            raw = Path(a.raw_dir) / f"{a.symbol}_{d.isoformat()}.csv.gz"
            if not a.no_download:
                print(f"{d}: downloading {C.url_for(d, a.symbol)}", flush=True)
                raw = C.download(d, a.symbol, a.raw_dir)
            size_mb = raw.stat().st_size / 1e6
            t1 = time.time()
            s, g, chk = C.process_day(raw, d, tick)
            C.save_grid(gpath, s, g, chk)
            print(f"{d}: {chk['rows']:,} updates ({size_mb:,.0f} MB) processed in {time.time() - t1:.0f}s", flush=True)
            if not a.keep_raw and not a.no_download:
                raw.unlink()
        s, g, chk = C.load_grid(gpath)
        frames.append(C.day_frame(s, g))
        grids.append((s, g, chk))

    res = C.crypto_study(frames, dates, grids)
    res["symbol"] = a.symbol
    res["runtime_s"] = time.time() - t0
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "crypto_results.json").write_text(json.dumps(res, indent=1, default=float))
    crypto_report.write(res, ROOT / "results" / "crypto_results.md", ROOT / "figures")
    print(f"done in {res['runtime_s']:.0f}s; see results/crypto_results.md")


if __name__ == "__main__":
    main()
