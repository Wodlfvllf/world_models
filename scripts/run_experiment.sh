#!/usr/bin/env bash
# End-to-end DEVELOPMENT run. Never silently generates/opens confirmation data.
set -Eeuo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$ROOT"
CONFIG=configs/smoke.json
OUTPUT=""
CACHE="${WM_AUDIT_CACHE:-$ROOT/cache}"
DEVICE=cuda
MODEL=both
TRUST=0
usage() {
    cat <<'HELP'
Usage: bash scripts/run_experiment.sh --output runs/NEW_NAME [options]
  --config PATH       Default configs/smoke.json; use configs/pilot.json next
  --model NAME        lewm, dinowm, or both (default both)
  --device NAME       cuda (default) or cpu
  --cache PATH        Artifact cache (default WM_AUDIT_CACHE or project/cache)
  --trust-checkpoint  Explicit consent to load the pinned DINO release pickle

Run bootstrap.sh --download first. Output must NOT already exist.
Runs preflight, generation, replay qualification, fitting, clean validation,
pilot evaluation and paired analysis. Confirmation always needs manual review.
Paths are relative to the project. Errors stop immediately; logs are retained.
HELP
}
die() { printf '%s\n' "$*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|--output|--cache|--device|--model)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"
            case "$1" in
                --config) CONFIG="$2" ;; --output) OUTPUT="$2" ;;
                --cache) CACHE="$2" ;; --device) DEVICE="$2" ;; --model) MODEL="$2" ;;
            esac
            shift 2 ;;
        --trust-checkpoint) TRUST=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "Unknown option: $1" ;;
    esac
done
[[ -n "$OUTPUT" ]] || die 'Supply --output with a NEW experiment directory'
[[ "$DEVICE" == cuda || "$DEVICE" == cpu ]] || die 'Device must be cuda or cpu'
case "$MODEL" in
    both) MODELS=(lewm dinowm) ;; lewm|dinowm) MODELS=("$MODEL") ;;
    *) die 'Model must be lewm, dinowm or both' ;;
esac
[[ "$MODEL" == lewm || "$TRUST" -eq 1 ]] || die 'DINO loading requires --trust-checkpoint; use --model lewm otherwise'
[[ -f "$CONFIG" ]] || die "Missing configuration: $CONFIG"
[[ -x "$ROOT/.venv/bin/python" ]] || die 'Run bash scripts/bootstrap.sh --download first'
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || die 'Output already exists; choose a NEW directory (no overwrites or implicit resume)'
mkdir -p -- "$OUTPUT"
exec > >(tee "$OUTPUT/console.log") 2>&1
trap 'printf "FAILED at line %s. Preserve this run and use a new output directory after fixing the error.\n" "$LINENO" >&2' ERR
CLI=("$ROOT/.venv/bin/python" -u -m wm_audit.cli --cache "$CACHE" --device "$DEVICE")
if [[ "$TRUST" -eq 1 ]]; then CLI+=(--trust-checkpoint); fi
step() { printf '\n>>> %s\n' "$*"; "${CLI[@]}" "$@"; }
step preflight --models "${MODELS[@]}" --output "$OUTPUT/preflight.json"
step generate --config "$CONFIG" --data "$OUTPUT/data"
ROOTS=("$OUTPUT"/data/pilot/*.npz)
[[ -f "${ROOTS[0]}" ]] || die 'A development run needs a nonempty pilot split'
step qualify "${ROOTS[0]}" --output "$OUTPUT/replay.json"
for model in "${MODELS[@]}"; do
    RUN="$OUTPUT/$model"
    step infer --model "$model" --root "${ROOTS[0]}" --output "$RUN/inference.json"
    step fit --model "$model" --config "$CONFIG" --data "$OUTPUT/data" --run "$RUN"
    step evaluate --model "$model" --config "$CONFIG" --data "$OUTPUT/data" --run "$RUN" --split validation
    step evaluate --model "$model" --config "$CONFIG" --data "$OUTPUT/data" --run "$RUN" --split pilot
    step analyze --config "$CONFIG" --data "$OUTPUT/data" --run "$RUN" --split pilot
done
printf '\nDONE. Reports: %s/<model>/report-pilot.json\n' "$OUTPUT"
printf '%s\n' 'These are exploratory diagnostics, not automatic confirmation of the research hypothesis.'