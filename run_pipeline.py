#!/usr/bin/env python3
"""Orchestrator for the de novo binder design pipeline.

Stages (run in order, selectable with --stages):
  setup   - git clone + uv venv setup for each part (part1_diffusion,
            part2_proteinmpnn, part3_colabfold)
  diffuse - part 1: RFdiffusion backbone binder design against the target
  mpnn    - part 2: ProteinMPNN inverse folding (binder sequences)
  refold  - part 3: ColabFold AF2-multimer refolding of target+binder
  eval    - ipAE vs pLDDT metrics, pass/fail, plot

Examples:
  uv run python run_pipeline.py --settings settings.json --device gpu
  uv run python run_pipeline.py --settings settings.json --device cpu --smoke --mock
  uv run python run_pipeline.py --settings settings.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.config import load_settings
from pipeline.setup import setup_part
from pipeline.util import setup_logger
from pipeline import stage_diffuse, stage_mpnn, stage_refold, metrics

STAGES = ["setup", "diffuse", "mpnn", "refold", "eval"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", default="settings.json", help="Path to settings JSON")
    parser.add_argument("--device", choices=["gpu", "cpu"], default=None,
                        help="Override settings device. cpu hides CUDA devices (smoke tests)")
    parser.add_argument("--stages", default="all",
                        help="Comma-separated stages to run: " + ",".join(STAGES) + " or 'all'")
    parser.add_argument("--smoke", action="store_true",
                        help="Apply the 'smoke' overrides from settings.json (tiny run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all commands without executing them")
    parser.add_argument("--mock", nargs="?", const="diffuse,mpnn,refold", default=None,
                        help="Mock the given heavy stages (comma-separated subset of "
                             "diffuse,mpnn,refold; bare --mock mocks all three) for CPU smoke tests")
    parser.add_argument("--workdir", default=None, help="Override settings paths.workdir")
    parser.add_argument("--force-setup", action="store_true", help="Re-run setup even if done")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.settings, device=args.device, smoke=args.smoke,
                             workdir=args.workdir)
    settings.workdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("pipeline", settings.workdir / "pipeline.log")
    logger.info("Pipeline start: project=%s device=%s smoke=%s dry_run=%s mock=%s",
                settings.get("project"), settings.device, args.smoke, args.dry_run, args.mock)
    logger.info("Settings: %s | workdir: %s", Path(args.settings).resolve(), settings.workdir)

    mock = None
    if args.mock:
        mock = set(args.mock.split(",")) & {"diffuse", "mpnn", "refold"}

    ctx = {
        "settings": settings,
        "logger": logger,
        "dry_run": args.dry_run,
        "mock": mock,
        "manifest": [],
    }

    selected = STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    for stage in selected:
        if stage not in STAGES:
            logger.error("Unknown stage %r (choose from %s)", stage, STAGES)
            return 2
        logger.info("=== stage: %s ===", stage)
        t0 = time.time()
        if stage == "setup":
            for part in ("part1_diffusion", "part2_proteinmpnn", "part3_colabfold"):
                cfg = settings.get(part, {})
                if not cfg.get("enabled", True):
                    continue
                # skip setup for stages that are mocked in this run
                stage_name = {"part1_diffusion": "diffuse",
                              "part2_proteinmpnn": "mpnn",
                              "part3_colabfold": "refold"}[part]
                if mock and stage_name in mock and stage_name in selected:
                    logger.info("[%s] mocked in this run, skipping setup", part)
                    continue
                if args.dry_run:
                    setup_part(settings, part, logger, dry_run=True, force=args.force_setup)
                else:
                    setup_part(settings, part, logger, force=args.force_setup)
        elif stage == "diffuse":
            stage_diffuse.run(settings, ctx)
        elif stage == "mpnn":
            stage_mpnn.run(settings, ctx)
        elif stage == "refold":
            stage_refold.run(settings, ctx)
        elif stage == "eval":
            metrics.evaluate(settings, ctx)
        logger.info("=== stage %s done in %.1fs ===", stage, time.time() - t0)

    manifest_path = settings.workdir / "manifest.json"
    manifest = {
        "project": settings.get("project"),
        "device": settings.device,
        "smoke": args.smoke,
        "dry_run": args.dry_run,
        "mock": sorted(mock) if mock else [],
        "stages": selected,
        "target_pdb": str(settings.target_pdb),
        "workdir": str(settings.workdir),
        "entries": ctx["manifest"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest written to %s", manifest_path)
    logger.info("Pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
