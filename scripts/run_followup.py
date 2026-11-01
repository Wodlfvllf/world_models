"""One frozen-model follow-up in a NEW directory; no downloads or implicit resume."""
import argparse
import json
from pathlib import Path

from wm_audit.models import load_model
from wm_audit.followup_probes import probe_followup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("lewm", "dinowm"), required=True)
    parser.add_argument("--stage", choices=("probes",), default="probes")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--trust-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve it and choose a new directory.")
    config, spec = json.loads(args.config.read_text()), json.loads(args.spec.read_text())
    model = load_model(args.model, args.cache, args.device, trust_checkpoint=args.trust_checkpoint)
    result = probe_followup(config, args.data, args.output, model, spec=spec)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()