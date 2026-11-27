#!/bin/bash
# Usage: bash src/r23_runq20.sh <jobs file> <parallel workers>   (r20_controls.py jobs; env NT=1)
cd "$(dirname "$0")/.."
export NT=1 PYTHONPATH="$PWD/src"
xargs -P "${2:-1}" -L 1 python3 src/r20_controls.py < "$1"
