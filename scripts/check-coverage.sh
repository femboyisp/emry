#!/usr/bin/env bash
# Enforce minimum line coverage for the workspace (default 90%).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MIN="${EMRY_MIN_COVERAGE:-90}"

if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
  echo "error: cargo-llvm-cov not found. Install with: cargo install cargo-llvm-cov" >&2
  echo "       Also run: rustup component add llvm-tools-preview" >&2
  exit 1
fi

echo "==> emry coverage gate (minimum ${MIN}% lines)"
cargo llvm-cov --workspace --summary-only --fail-under-lines "${MIN}"
