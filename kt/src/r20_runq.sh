#!/bin/bash
# Usage: bash src/r20_runq.sh <jobs file> <parallel workers>   (env KT_ROOT, NT=1)
cd "$(dirname "$0")/.."
export NT=1 PYTHONPATH="$PWD/src"
xargs -P "${2:-2}" -L 1 python3 src/r20_controls.py < "$1"
