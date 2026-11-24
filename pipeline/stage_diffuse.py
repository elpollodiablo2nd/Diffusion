"""Part 1: RFdiffusion backbone generation against a target crystal structure.

Builds the contig string (fixed target segment + diffused binder segment),
runs scripts/run_inference.py from the cloned RFdiffusion repo inside its uv
venv, and collects the generated complex PDBs.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from .setup import ensure_setup
from .util import parse_residue_range, run_cmd, venv_python, device_env

CHAIN_ORDER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_contig_string(cfg: dict) -> str:
    """Return the hydra contig override string, e.g. '[A1-79/0 50-80]'."""
    contigs = cfg.get("contigs")
    if contigs:
        if isinstance(contigs, str):
            segments = contigs.strip()
        else:
            segments = " ".join(str(c) for c in contigs)
        if not segments.startswith("["):
            segments = f"[{segments}]"
        return segments
    chain = str(cfg.get("target_chain", "A"))
    lo, hi = parse_residue_range(cfg.get("target_residues", "1-79"))
    binder = str(cfg.get("binder_length", "50-80"))
    return f"[{chain}{lo}-{hi}/0 {binder}]"


def contig_chain_layout(contig_string: str) -> tuple[list[str], str]:
    """RFdiffusion assigns diffused segments to chains A,B,... in order and the
    receptor (target) fragments to the following chain letters.

    Returns (binder_chains, receptor_chain).
    """
    segments = contig_string.strip().strip("[]").split()
    inpaint = [s for s in segments if not re.match(r"^[A-Za-z]", s)]
    n_inpaint = len(inpaint)
    binder_chains = list(CHAIN_ORDER[:n_inpaint])
    receptor_chain = CHAIN_ORDER[n_inpaint] if n_inpaint < len(CHAIN_ORDER) else "Z"
    return binder_chains, receptor_chain


def build_command(settings, cfg: dict, ctx: dict) -> tuple[list[str], Path]:
    repo_dir = settings.path(cfg["setup"]["repo_dir"])
    venv = settings.path(cfg["setup"]["venv"])
    out_prefix = ctx["diffuse_prefix"]
    contig = ctx["contig_string"]
    hotspots = cfg.get("hotspot_res") or []
    cmd = [
        str(venv_python(venv)), str(repo_dir / "scripts" / "run_inference.py"),
        f"inference.output_prefix={out_prefix}",
        f"inference.input_pdb={settings.target_pdb}",
        f"contigmap.contigs={contig}",
        f"ppi.hotspot_res=[{','.join(hotspots)}]" if hotspots else "ppi.hotspot_res=null",
        f"inference.num_designs={int(cfg.get('num_designs', 10))}",
        f"diffuser.T={int(cfg.get('diffusion_steps', 50))}",
        f"denoiser.noise_scale_ca={cfg.get('noise_scale_ca', 0)}",
        f"denoiser.noise_scale_frame={cfg.get('noise_scale_frame', 0)}",
        f"inference.write_trajectory={bool(cfg.get('write_trajectory', False))}",
    ]
    for key, value in (cfg.get("extra_args") or {}).items():
        cmd.append(f"{key}={value}")
    return cmd, repo_dir


def mock_complex_pdb(target_pdb: Path, target_chain: str, res_spec: str,
                     binder_len: int, out_path: Path, *,
                     binder_chain: str, receptor_chain: str, seed: int) -> None:
    """Write a synthetic two-chain complex PDB (binder chain + receptor chain).

    Only used with --mock for CPU smoke tests; geometry is a translated copy of
    the target backbone, which is sufficient for downstream parsing.
    """
    lo, hi = parse_residue_range(res_spec)
    rng = random.Random(seed)
    lines = target_pdb.read_text().splitlines()
    receptor_atoms = [l for l in lines
                      if l.startswith("ATOM") and l[21] == target_chain
                      and lo <= int(l[22:26]) <= hi]
    binder_src = [l for l in lines
                  if l.startswith("ATOM") and l[21] == target_chain
                  and int(l[22:26]) <= binder_len]
    shift = 80.0 + rng.uniform(0, 20)

    def rechain(atom_lines: list[str], new_chain: str, x_shift: float) -> list[str]:
        out, res_map, next_res = [], {}, 1
        for line in atom_lines:
            key = line[22:27]
            if key not in res_map:
                res_map[key] = next_res
                next_res += 1
            x = float(line[30:38]) + x_shift
            out.append(line[:21] + new_chain + f"{res_map[key]:4d}" + line[26:30]
                       + f"{x:8.3f}" + line[38:])
        return out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("REMARK   634 mock diffused complex for smoke test\n")
        fh.write("".join(l + "\n" for l in rechain(binder_src, binder_chain, 0.0)))
        fh.write("TER\n")
        fh.write("".join(l + "\n" for l in rechain(receptor_atoms, receptor_chain, shift)))
        fh.write("TER\nEND\n")


def run(settings, ctx: dict) -> list[Path]:
    logger = ctx["logger"]
    part = "part1_diffusion"
    cfg = settings.require(part)
    workdir = settings.workdir
    out_dir = settings.stage_dir("1_diffusion")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx["diffuse_prefix"] = str(out_dir / "outputs" / "design")
    ctx["contig_string"] = build_contig_string(cfg)
    binder_chains, receptor_chain = contig_chain_layout(ctx["contig_string"])
    ctx["binder_chains"] = binder_chains
    ctx["receptor_chain"] = receptor_chain
    logger.info("[diffuse] contig string: %s (binder chains %s, receptor chain %s)",
                ctx["contig_string"], binder_chains, receptor_chain)

    if ctx["mock"] and "diffuse" in ctx["mock"]:
        outputs = []
        n = int(cfg.get("num_designs", 10))
        lo, hi = parse_residue_range(cfg.get("target_residues", "1-79"))
        b_lo, b_hi = parse_residue_range(cfg.get("binder_length", "50-80"))
        rng = random.Random(settings.seed)
        for i in range(n):
            binder_len = rng.randint(b_lo, b_hi)
            out = out_dir / "outputs" / f"design_{i}.pdb"
            mock_complex_pdb(settings.target_pdb, cfg.get("target_chain", "A"),
                             f"{lo}-{hi}", binder_len, out,
                             binder_chain=binder_chains[0],
                             receptor_chain=receptor_chain, seed=settings.seed + i)
            outputs.append(out)
            logger.info("[diffuse][mock] wrote %s", out)
        ctx["diffuse_outputs"] = outputs
        ctx["manifest"].append({"stage": "diffuse", "mode": "mock", "outputs": [str(p) for p in outputs]})
        return outputs

    if not ctx["dry_run"]:
        ensure_setup(settings, part, logger, dry_run=ctx["dry_run"])

    cmd, repo_dir = build_command(settings, cfg, ctx)
    run_cmd(cmd, cwd=repo_dir, env=device_env(settings.device), logger=logger,
            dry_run=ctx["dry_run"])
    ctx["manifest"].append({
        "stage": "diffuse", "mode": "dry-run" if ctx["dry_run"] else "run",
        "command": cmd, "cwd": str(repo_dir),
    })

    if ctx["dry_run"]:
        # expected output paths only, so downstream stages can build their commands
        outputs = [out_dir / "outputs" / f"design_{i}.pdb"
                   for i in range(int(cfg.get("num_designs", 10)))]
    else:
        outputs = sorted((out_dir / "outputs").glob("design_*.pdb"))
    ctx["diffuse_outputs"] = outputs
    logger.info("[diffuse] collected %d backbone PDBs", len(outputs))
    return outputs
