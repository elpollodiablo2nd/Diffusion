"""Part 3: refolding of designed target+binder complexes with ColabFold AF2-multimer.

Runs colabfold_batch (alphafold2_multimer_v3) once per design FASTA. In mock
mode (CPU smoke tests) synthetic ColabFold-style score JSONs are written so the
evaluation stage can be exercised end to end.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .setup import ensure_setup
from .util import device_env, fasta_chain_lengths, run_cmd


def gather_fastas(settings, cfg: dict, ctx: dict) -> list[Path]:
    if ctx.get("mpnn_rows"):
        fastas = [Path(r["fasta"]) for r in ctx["mpnn_rows"]]
    else:
        pattern = cfg.get("input_glob") or str(settings.stage_dir("3_refold") / "fastas" / "*.fasta")
        p = Path(pattern)
        if not p.is_absolute():
            p = settings.root / p
        fastas = sorted(p.parent.glob(p.name))
    max_d = cfg.get("max_designs")
    if max_d:
        fastas = fastas[: int(max_d)]
    return fastas


def build_command(settings, cfg: dict, fasta: Path, outdir: Path) -> list[str]:
    venv = settings.path(cfg["setup"]["venv"])
    skip_output = cfg.get("skip_output") or []
    cmd = [
        str(venv / "bin" / "colabfold_batch"),
        str(fasta), str(outdir),
        "--model-type", str(cfg.get("model_type", "alphafold2_multimer_v3")),
        "--model-order", str(cfg.get("model_order", "1")),
        "--num-models", str(int(cfg.get("num_models", 1))),
        "--num-recycle", str(int(cfg.get("num_recycle", 3))),
        "--num-seeds", str(int(cfg.get("num_seeds", 1))),
        "--pair-mode", str(cfg.get("pair_mode", "unpaired")),
        "--msa-mode", str(cfg.get("msa_mode", "single_sequence")),
        "--rank-by", str(cfg.get("rank_by", "multimer")),
        "--random-seed", str(settings.seed),
    ]
    if skip_output:
        cmd += ["--skip-output", ",".join(skip_output)]
    for key, value in (cfg.get("extra_args") or {}).items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    return cmd


def mock_scores_json(settings, cfg: dict, fasta: Path, outdir: Path, index: int) -> Path:
    """Write a synthetic ColabFold rank-1 scores JSON + unrelaxed PDB."""
    lengths = fasta_chain_lengths(fasta)
    n = sum(lengths)
    rng = random.Random(settings.seed + 31 * index)
    good = (index % 2 == 0)
    plddt = []
    for ci, length in enumerate(lengths):
        base = rng.uniform(0.88, 0.97) if good else rng.uniform(0.45, 0.62)
        plddt += [round(base * rng.uniform(0.9, 1.0) * 100, 2) for _ in range(length)]
    pae = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            same = any(sum(lengths[:k]) <= i < sum(lengths[: k + 1])
                       and sum(lengths[:k]) <= j < sum(lengths[: k + 1])
                       for k in range(len(lengths)))
            if same:
                pae[i][j] = round(rng.uniform(1.0, 8.0), 2)
            else:
                pae[i][j] = round(rng.uniform(2.0, 8.0) if good else rng.uniform(12.0, 30.0), 2)
    model_type = cfg.get("model_type", "alphafold2_multimer_v3")
    tag = f"rank_001_{model_type}_model_{cfg.get('model_order', '1')}_seed_{settings.seed:03d}"
    scores = {
        "plddt": plddt,
        "max_pae": max(max(row) for row in pae),
        "pae": pae,
        "ptm": round(rng.uniform(0.6, 0.9) if good else rng.uniform(0.2, 0.4), 2),
        "iptm": round(rng.uniform(0.6, 0.9) if good else rng.uniform(0.1, 0.4), 2),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    scores_path = outdir / f"{fasta.stem}_scores_{tag}.json"
    scores_path.write_text(json.dumps(scores))
    pdb_path = outdir / f"{fasta.stem}_unrelaxed_{tag}.pdb"
    pdb_path.write_text("REMARK   634 mock AF2-multimer prediction for smoke test\nEND\n")
    return scores_path


def run(settings, ctx: dict) -> list[Path]:
    logger = ctx["logger"]
    part = "part3_colabfold"
    cfg = settings.require(part)
    outdir = settings.stage_dir("3_refold") / "colabfold"
    fastas = gather_fastas(settings, cfg, ctx)
    if not fastas:
        logger.warning("[refold] no input FASTAs found; nothing to do")
        ctx["refold_scores"] = []
        return []
    logger.info("[refold] %d complexes to refold with AF2-multimer", len(fastas))

    if ctx["mock"] and "refold" in ctx["mock"]:
        scores = []
        for i, fasta in enumerate(fastas):
            scores.append(mock_scores_json(settings, cfg, fasta, outdir, i))
            logger.info("[refold][mock] wrote %s", scores[-1])
        ctx["refold_scores"] = scores
        pdbs = sorted(outdir.glob("*_rank_001_*.pdb"))
        ctx["refold_pdbs"] = pdbs
        ctx["manifest"].append({"stage": "refold", "mode": "mock",
                                "scores": [str(s) for s in scores],
                                "pdbs": [str(p) for p in pdbs]})
        return scores

    if not ctx["dry_run"]:
        ensure_setup(settings, part, logger, dry_run=ctx["dry_run"])

    cmds = []
    for fasta in fastas:
        cmd = build_command(settings, cfg, fasta, outdir)
        cmds.append(cmd)
        run_cmd(cmd, cwd=settings.root, env=device_env(settings.device),
                logger=logger, dry_run=ctx["dry_run"])
    ctx["manifest"].append({
        "stage": "refold", "mode": "dry-run" if ctx["dry_run"] else "run",
        "commands": cmds,
    })

    if ctx["dry_run"]:
        ctx["refold_scores"] = []
        ctx["refold_pdbs"] = []
        return []
    scores = sorted(outdir.glob("*_scores_rank_001_*.json"))
    pdbs = sorted(outdir.glob("*_rank_001_*.pdb"))
    ctx["refold_scores"] = scores
    ctx["refold_pdbs"] = pdbs
    logger.info("[refold] collected %d rank-1 score files and %d rank-1 PDBs (unrelaxed%s)",
                len(scores), len(pdbs), "/relaxed" if any("_relaxed_" in p.name for p in pdbs) else "")
    for s in scores:
        stem = s.name.split("_scores_")[0]
        if not list(outdir.glob(f"{stem}_*_rank_001_*.pdb")):
            logger.warning("[refold] no PDB found for %s", s.name)
    return scores
