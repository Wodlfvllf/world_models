#!/usr/bin/env bash
# Bash 3.2+ (including macOS); run with bash, never source this script.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf '%s\n' 'Run this script with bash; do not source it.' >&2
    return 2
fi
set -Eeuo pipefail
IFS=$'\n\t'
umask 022

usage() {
    cat <<'HELP'
Usage: bash scripts/bootstrap.sh [--cpu] [--download] [--with-dataset]
                                [--cache PATH] [--trust-checkpoint]

Default: Linux NVIDIA, torch 2.10.0 / torchvision 0.25.0, official cu128 index.
--cpu              CPU wheels on Linux; native CPU-capable wheels on macOS.
--download         Fetch and verify the two pinned RGB releases and sources.
--with-dataset     Implies --download. OPTIONAL original scene-camera PushT:
                   13.14 GB download + approximately 46 GB extracted; keep
                   at least 65 GB free. Never needed for generated audit roots.
--cache PATH       Artifact cache (default: WM_AUDIT_CACHE or project/cache).
                   Relative paths are relative to the project, not the caller.
--trust-checkpoint Accepted for compatibility, but grants no trust here:
                   downloads NEVER deserialize DINO. Supply explicit trust
                   separately when loading that pinned pickle in the CLI.

Uses project-local environments/caches only. If uv 0.9.28 is not already
installed, requires python3.11 or python3 with venv/ensurepip to install that
pinned uv wheel into .bootstrap-venv. uv downloads managed Python 3.11 for
.venv. No sudo, global installs, shell installer, submodules, or CUDA builds.
Existing non-isolated/non-3.11 .venv directories are NOT removed or replaced.
HELP
}

die() { printf 'bootstrap: %s\n' "$*" >&2; exit 1; }
CPU=0
DOWNLOAD=0
WITH_DATASET=0
TRUST=0
CACHE="${WM_AUDIT_CACHE:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) CPU=1; shift ;;
        --download) DOWNLOAD=1; shift ;;
        --with-dataset) WITH_DATASET=1; DOWNLOAD=1; shift ;;
        --trust-checkpoint) TRUST=1; shift ;;
        --cache)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die '--cache needs a path'
            CACHE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
done

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)"
[[ -f "$ROOT/pyproject.toml" && -f "$ROOT/src/wm_audit/models.py" ]] || die 'not a wm-audit project directory'
cd -- "$ROOT"
VENV="$ROOT/.venv"
BOOTSTRAP="$ROOT/.bootstrap-venv"
STATE="$ROOT/.bootstrap-cache"
PYTHON="$VENV/bin/python"
UV_VERSION=0.9.28
[[ -n "$CACHE" ]] || CACHE="$ROOT/cache"
[[ ! -L "$VENV" && ! -L "$BOOTSTRAP" && ! -L "$STATE" ]] || die 'environment/cache directories must not be symlinks'
# Do not inherit Python import paths or pip/uv user configuration.
unset PYTHONHOME PYTHONPATH
export UV_NO_CONFIG=1
# Keep managed interpreters independent of the optional uv seed venv. This
# also permits switching between an installed uv and the private pinned uv.
export UV_CACHE_DIR="$STATE/uv"
export UV_PYTHON_INSTALL_DIR="$STATE/python"

check_environment() {
    [[ -x "$1" && -f "$2/pyvenv.cfg" && ! -L "$2/pyvenv.cfg" ]] || return 1
    "$1" -I - "$2" "$3" <<'PY'
from pathlib import Path
import sys
root, version = Path(sys.argv[1]).resolve(), sys.argv[2]
cfg = dict(line.split('=', 1) for line in (root / 'pyvenv.cfg').read_text().splitlines() if '=' in line)
cfg = {key.strip().lower(): value.strip().lower() for key, value in cfg.items()}
if (Path(sys.prefix).resolve() != root or sys.prefix == sys.base_prefix
        or cfg.get('include-system-site-packages') != 'false'
        or (version and '.'.join(map(str, sys.version_info[:2])) != version)):
    raise SystemExit('Existing environment has the wrong Python version or is not isolated.')
PY
}

if [[ -e "$VENV" ]]; then
    check_environment "$PYTHON" "$VENV" '3.11' || die 'preserving existing .venv; move it aside yourself before creating the required isolated Python 3.11 environment'
fi
OS="$(uname -s)"
case "$OS" in
    Linux)
        if [[ "$CPU" -eq 1 ]]; then
            TORCH_INDEX=https://download.pytorch.org/whl/cpu
            TORCH_PIN=2.10.0+cpu
            VISION_PIN=0.25.0+cpu
        else
            command -v nvidia-smi >/dev/null 2>&1 || die 'NVIDIA driver/nvidia-smi required; use --cpu for local tests'
            nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
            TORCH_INDEX=https://download.pytorch.org/whl/cu128
            TORCH_PIN=2.10.0+cu128
            VISION_PIN=0.25.0+cu128
        fi ;;
    Darwin)
        [[ "$CPU" -eq 1 ]] || die 'macOS requires --cpu; the Blackwell CUDA target is Linux'
        TORCH_INDEX=https://pypi.org/simple
        TORCH_PIN=2.10.0
        VISION_PIN=0.25.0 ;;
    *) die "unsupported OS: $OS" ;;
esac
if [[ "$DOWNLOAD" -eq 1 ]]; then
    command -v git >/dev/null 2>&1 || die 'git is required for --download'
fi
if [[ "$WITH_DATASET" -eq 1 ]]; then
    printf '%s\n' 'Opted into 13.14 GB of original training data (~46 GB extracted; archive retained).' >&2
fi

uv_is_pinned() {
    local version
    version="$("$1" --version)" || return 1
    # Official wheels append a commit/date, e.g. "uv 0.9.28 (hash date)".
    [[ "$version" == "uv $UV_VERSION" || "$version" == "uv $UV_VERSION "* ]]
}

UV="$(command -v uv || true)"
if [[ -z "$UV" ]] || ! uv_is_pinned "$UV"; then
    if [[ -e "$BOOTSTRAP" ]]; then
        check_environment "$BOOTSTRAP/bin/python" "$BOOTSTRAP" '' || die 'preserving invalid .bootstrap-venv; move it aside manually'
    else
        SEED_PYTHON="$(command -v python3.11 || command -v python3 || true)"
        [[ -n "$SEED_PYTHON" ]] || die 'install python3 with venv/ensurepip, or the official pinned uv binary, first'
        "$SEED_PYTHON" -I -m venv "$BOOTSTRAP" || die 'python3 with venv/ensurepip is required; no system packages will be installed'
    fi
    "$BOOTSTRAP/bin/python" -I -m pip --isolated install --disable-pip-version-check --no-cache-dir \
        --index-url https://pypi.org/simple "uv==$UV_VERSION"
    UV="$BOOTSTRAP/bin/uv"
fi
uv_is_pinned "$UV" || die 'unexpected uv version'
mkdir -p -- "$STATE"
if [[ ! -e "$VENV" ]]; then
    "$UV" venv --python 3.11 --managed-python "$VENV"
fi
check_environment "$PYTHON" "$VENV" '3.11' || die 'uv did not create an isolated Python 3.11 environment'

CONSTRAINTS="$(mktemp "$STATE/.torch-constraints.XXXXXX")"
trap 'rm -f -- "$CONSTRAINTS"' EXIT
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_PIN" "$VISION_PIN" > "$CONSTRAINTS"
# Install torch separately. Exact LOCAL build pins prevent the subsequent
# editable install from silently replacing cu128 with PyPI's default wheels.
"$UV" pip install --python "$PYTHON" --index-url "$TORCH_INDEX" \
    --constraint "$CONSTRAINTS" "torch==$TORCH_PIN" "torchvision==$VISION_PIN"
"$UV" pip install --python "$PYTHON" --index-url https://pypi.org/simple \
    --constraint "$CONSTRAINTS" -e '.[test]'
"$UV" pip check --python "$PYTHON"
"$PYTHON" - "$CPU" "$TORCH_PIN" "$VISION_PIN" <<'PY'
import importlib.metadata as md
import sys
import torch
import torchvision
for name, expected in zip(('torch', 'torchvision'), sys.argv[2:]):
    actual = md.version(name)
    if actual != expected:
        raise SystemExit(f'{name} build was replaced: expected {expected}, got {actual}')
device = 'cpu'
if sys.argv[1] == '0':
    if torch.version.cuda != '12.8' or not torch.cuda.is_available():
        raise SystemExit('cu128 installed, but CUDA is unavailable; check the NVIDIA driver.')
    device = 'cuda'
    print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0))
# Execute a kernel, not just an availability check (important on Blackwell).
x = torch.ones((8, 8), device=device)
assert (x @ x).sum().item() == 512
print('Verified:', sys.version.split()[0], 'torch', torch.__version__, 'torchvision', torchvision.__version__, device)
PY

if [[ "$TRUST" -eq 1 ]]; then
    printf '%s\n' '--trust-checkpoint does not load or authorize any pickle during bootstrap.' >&2
fi
if [[ "$DOWNLOAD" -eq 1 ]]; then
    "$PYTHON" - "$CACHE" "$WITH_DATASET" <<'PY'
import json
from pathlib import Path
import sys
from wm_audit.artifacts import prepare
print(json.dumps(prepare(Path(sys.argv[1]), dataset=sys.argv[2] == '1'), indent=2, sort_keys=True))
PY
fi
printf '\nReady. Python: %s\nCLI: %s\n' "$PYTHON" "$VENV/bin/wm-audit"
printf '%s\n' 'Dataset download is optional. Run tests with .venv/bin/python -m pytest.'