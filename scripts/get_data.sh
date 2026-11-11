#!/usr/bin/env bash
# Download the LOBSTER sample day (21 June 2012) used by the study and verify it.
#
# Source: the LOBSTER sample files (lobsterdata.com), reduced to every message
# plus the best bid and ask by the public repository m4rpg/algo-trading-liquidity.
# Pinned to one commit and checked by SHA-256 so the inputs can never drift.
set -euo pipefail
cd "$(dirname "$0")/../data"
COMMIT=e8bd603d4161c5dae2fc7afcb729c2e116127090
BASE="https://raw.githubusercontent.com/m4rpg/algo-trading-liquidity/${COMMIT}/raw_data"
for T in AAPL AMZN GOOG INTC MSFT; do
  F="LOBSTER_${T}_compact.csv.gz"
  [ -f "$F" ] || curl -fsSL -o "$F" "$BASE/$F"
done
sha256sum -c SHA256SUMS
