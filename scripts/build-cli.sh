#!/usr/bin/env bash
# Compile the `emry` CLI and stage it where the wheel build bundles it
# (python/emry/_bin/). Run before `maturin build` so the binary lands in the
# wheel and `pip install emry` provides the `emry` command.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

cargo build --release -p emry-cli
mkdir -p python/emry/_bin
cp target/release/emry python/emry/_bin/emry
echo "staged $(python/emry/_bin/emry --version 2>/dev/null || echo emry) at python/emry/_bin/emry"
