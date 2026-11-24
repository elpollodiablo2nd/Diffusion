"""Evaluation: ipAE vs pLDDT for AF2-multimer refolded designs.

Metrics per design (rank-1 ColabFold model):
  - ipAE: mean predicted aligned error (PAE) over all cross-chain residue
    pairs of the target:binder complex (Angstrom).
  - pLDDT: mean per-residue pLDDT, normalized to 0-1. Scope "binder" averages
    over the binder chain(s) only, "complex" over all chains.
  - pass: pLDDT > plddt_min AND ipAE < ipae_max  (defaults 0.7 and 10 A).

Outputs in <workdir>/4_eval/: metrics.csv, summary.json, ipae_vs_plddt.png and
passing/ with the rank-1 PDBs of passing designs.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .util import read_fasta


def interface_pae(pae: list[list[float]], lengths: list[int]) -> float:
    """Mean PAE over residue pairs belonging to different chains."""
    mat = np.asarray(pae, dtype=float)
    n = mat.shape[0]
    if mat.shape != (n, n):
        raise ValueError(f"PAE matrix is not square: {mat.shape}")
    asym = np.zeros(n, dtype=int)
    start = 0
    for ci, length in enumerate(lengths):
        asym[start:start + length] = ci
        start += length
    if start != n:
        raise ValueError(f"Chain lengths {lengths} sum to {start}, PAE is {n}x{n}")
    cross = asym[:, None] != asym[None, :]
    if not cross.any():
        return float("nan")
    return float(mat[cross].mean())


def plddt_fraction(plddt, lengths: list[int], scope: str = "binder") -> float:
    """Mean pLDDT normalized to 0-1 over the requested scope."""
    if isinstance(plddt, (int, float)):
        return float(plddt) / 100.0
    values = np.asarray(plddt, dtype=float)
    if scope == "binder" and len(lengths) > 1:
        values = values[sum(lengths[:1]):]
    return float(values.mean() / 100.0)


def chain_lengths_from_fasta(fasta: Path) -> list[int]:
    return [len(seq) for _, seq in read_fasta(fasta)]


def evaluate(settings, ctx: dict) -> list[dict]:
    logger = ctx["logger"]
    eval_cfg = settings.get("eval", {})
    plddt_min = float(eval_cfg.get("plddt_min", 0.7))
    ipae_max = float(eval_cfg.get("ipae_max", 10.0))
    scope = str(eval_cfg.get("plddt_scope", "binder"))
    top_n = int(eval_cfg.get("top_n", 10))

    eval_dir = settings.stage_dir("4_eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    scores = ctx.get("refold_scores")
    if not scores:
        pattern = settings.stage_dir("3_refold") / "colabfold" / "*_scores_rank_001_*.json"
        scores = sorted(pattern.parent.glob(pattern.name))
    mpnn_rows = {r.get("fasta_stem"): r for r in ctx.get("mpnn_rows", [])}

    rows = []
    for scores_path in scores:
        stem = scores_path.stem
        for tag in ("_scores_rank_001_",):
            if tag in stem:
                stem = stem.split(tag)[0]
                break
        fasta = settings.stage_dir("3_refold") / "fastas" / f"{stem}.fasta"
        lengths = chain_lengths_from_fasta(fasta) if fasta.exists() else None
        with scores_path.open() as fh:
            data = json.load(fh)
        pae = data.get("pae")
        plddt = data.get("plddt")
        if lengths is None and isinstance(plddt, list):
            lengths = [len(plddt)]
        ipae = interface_pae(pae, lengths) if pae and lengths else None
        plddt_binder = plddt_fraction(plddt, lengths, "binder") if plddt is not None and lengths else None
        plddt_complex = plddt_fraction(plddt, lengths, "complex") if plddt is not None and lengths else None
        plddt_used = plddt_binder if scope == "binder" else plddt_complex
        passed = (plddt_used is not None and plddt_used > plddt_min
                  and ipae is not None and ipae < ipae_max)
        mpnn = mpnn_rows.get(stem, {})
        rows.append({
            "design": mpnn.get("design", stem),
            "seq_id": mpnn.get("seq_id", ""),
            "scores_file": str(scores_path),
            "plddt": round(plddt_used, 4) if plddt_used is not None else None,
            "plddt_binder": round(plddt_binder, 4) if plddt_binder is not None else None,
            "plddt_complex": round(plddt_complex, 4) if plddt_complex is not None else None,
            "ipae": round(ipae, 3) if ipae is not None else None,
            "max_pae": data.get("max_pae"),
            "ptm": data.get("ptm"),
            "iptm": data.get("iptm"),
            "mpnn_score": mpnn.get("score"),
            "mpnn_global_score": mpnn.get("global_score"),
            "seq_recovery": mpnn.get("seq_recovery"),
            "pass": bool(passed),
        })

    rows.sort(key=lambda r: (not r["pass"], r["ipae"] if r["ipae"] is not None else 1e9))
    write_metrics_csv(eval_dir / "metrics.csv", rows)
    passing = [r for r in rows if r["pass"]]
    copy_passing_pdbs(settings, eval_dir, rows)
    summary = {
        "n_total": len(rows),
        "n_pass": len(passing),
        "pass_rate": round(len(passing) / len(rows), 4) if rows else 0.0,
        "plddt_min": plddt_min,
        "ipae_max": ipae_max,
        "plddt_scope": scope,
        "criteria": f"pass = pLDDT({scope}) > {plddt_min} AND ipAE < {ipae_max} A",
        "passing": passing[:top_n],
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_ipae_vs_plddt(eval_dir / "ipae_vs_plddt.png", rows, plddt_min, ipae_max, top_n)
    ctx["eval_rows"] = rows
    ctx["manifest"].append({"stage": "eval", "n_total": len(rows), "n_pass": len(passing),
                            "summary": str(eval_dir / "summary.json")})
    logger.info("[eval] %d designs, %d pass (pLDDT>%s and ipAE<%s A) -> %s",
                len(rows), len(passing), plddt_min, ipae_max, eval_dir)
    return rows


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    fields = ["design", "seq_id", "plddt", "plddt_binder", "plddt_complex", "ipae",
              "max_pae", "ptm", "iptm", "mpnn_score", "mpnn_global_score",
              "seq_recovery", "pass", "scores_file"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def copy_passing_pdbs(settings, eval_dir: Path, rows: list[dict]) -> None:
    passing_dir = eval_dir / "passing"
    for row in rows:
        if not row["pass"]:
            continue
        scores_path = Path(row["scores_file"])
        for pdb in scores_path.parent.glob(f"{scores_path.name.split('_scores_')[0]}_unrelaxed_rank_001_*.pdb"):
            passing_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(pdb, passing_dir / pdb.name)
        for pdb in scores_path.parent.glob(f"{scores_path.name.split('_scores_')[0]}_relaxed_rank_001_*.pdb"):
            passing_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(pdb, passing_dir / pdb.name)


def plot_ipae_vs_plddt(path: Path, rows: list[dict], plddt_min: float,
                       ipae_max: float, top_n: int) -> None:
    xs = [r["ipae"] for r in rows if r["ipae"] is not None]
    ys = [r["plddt"] for r in rows if r["ipae"] is not None and r["plddt"] is not None]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvspan(0, ipae_max, ymin=plddt_min, ymax=1.0, color="green", alpha=0.12,
               label="pass region")
    ax.axhline(plddt_min, color="gray", ls="--", lw=1)
    ax.axvline(ipae_max, color="gray", ls="--", lw=1)
    for row in rows:
        if row["ipae"] is None or row["plddt"] is None:
            continue
        ax.scatter(row["ipae"], row["plddt"],
                   c="green" if row["pass"] else "red",
                   edgecolor="black", s=45, zorder=3)
    passing = [r for r in rows if r["pass"]][:top_n]
    for row in passing:
        ax.annotate(f"{row['design']}_s{row['seq_id']}", (row["ipae"], row["plddt"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("ipAE (A)  - lower is better")
    ax.set_ylabel("pLDDT (0-1)  - higher is better")
    n_pass = sum(1 for r in rows if r["pass"])
    ax.set_title(f"AF2-multimer refolding: ipAE vs pLDDT  ({n_pass}/{len(rows)} pass: "
                 f"pLDDT > {plddt_min}, ipAE < {ipae_max} A)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
