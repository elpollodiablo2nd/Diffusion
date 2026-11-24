"""Per-part environment setup (git clone + uv venv) driven by settings.json.

Each part has a setup.sh in its own directory (part1_diffusion/, ...). The
orchestrator calls it with arguments taken from the part's "setup" section of
settings.json, so settings.json remains the single source of truth.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .util import run_cmd

PART_DIRS = {
    "part1_diffusion": "part1_diffusion",
    "part2_proteinmpnn": "part2_proteinmpnn",
    "part3_colabfold": "part3_colabfold",
}


def setup_marker(settings, part: str) -> Path:
    venv = settings.path(settings.get(f"{part}.setup.venv", f"envs/{part}"))
    return venv.parent / f".setup_{part}_{settings.device}.json"


def is_setup_done(settings, part: str) -> bool:
    return setup_marker(settings, part).exists()


def setup_part(settings, part: str, logger, *, dry_run: bool = False,
               force: bool = False, clone_only: bool = False) -> None:
    cfg = settings.get(part, {})
    if not cfg.get("enabled", True):
        return
    setup_cfg = cfg.get("setup", {})
    marker = setup_marker(settings, part)
    if marker.exists() and not force:
        logger.info("[%s] setup already done (%s), skipping", part, marker)
        return

    repo_dir = settings.path(setup_cfg.get("repo_dir", f"vendor/{part}"))
    venv = settings.path(setup_cfg.get("venv", f"envs/{part}"))
    script = settings.root / PART_DIRS[part] / "setup.sh"
    if not script.exists():
        raise FileNotFoundError(f"Missing setup script: {script}")

    cmd = [
        "bash", str(script),
        "--device", settings.device,
        "--repo-url", setup_cfg.get("repo_url", ""),
        "--repo-dir", str(repo_dir),
        "--venv", str(venv),
        "--python", str(setup_cfg.get("python", "3.10")),
    ]
    if setup_cfg.get("cuda_version"):
        cmd += ["--cuda-version", setup_cfg["cuda_version"]]
    if setup_cfg.get("torch_version"):
        cmd += ["--torch-version", setup_cfg["torch_version"]]
    if setup_cfg.get("dgl_version"):
        cmd += ["--dgl-version", setup_cfg["dgl_version"]]
    if setup_cfg.get("jax_version"):
        cmd += ["--jax-version", str(setup_cfg["jax_version"])]
    if setup_cfg.get("jax_cuda") is not None:
        cmd += ["--jax-cuda", "1" if setup_cfg.get("jax_cuda") else "0"]
    if part == "part1_diffusion":
        cmd += ["--models" if setup_cfg.get("download_models", True) else "--no-models"]
    if clone_only:
        cmd += ["--clone-only"]
    if force:
        cmd += ["--force"]

    run_cmd(cmd, cwd=settings.root, logger=logger, dry_run=dry_run)
    if not dry_run:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "part": part, "device": settings.device,
            "repo_dir": str(repo_dir), "venv": str(venv),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))
    logger.info("[%s] setup complete", part)


def ensure_setup(settings, part: str, logger, *, dry_run: bool = False) -> None:
    """Set up a part on demand if its venv/marker is missing."""
    if not is_setup_done(settings, part):
        setup_part(settings, part, logger, dry_run=dry_run)
