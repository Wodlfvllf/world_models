#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT/paper"
export TEXINPUTS="$PWD/vendor//:${TEXINPUTS:-}"
export BSTINPUTS="$PWD/vendor//:${BSTINPUTS:-}"
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build main.tex
"$ROOT/.venv/bin/python" audit_manuscript.py --root "$ROOT"