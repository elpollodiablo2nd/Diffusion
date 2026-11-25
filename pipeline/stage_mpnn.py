"""Part 2: ProteinMPNN inverse folding (sequence design) on diffused backbones.

Runs helper_scripts/parse_multiple_chains.py and protein_mpnn_run.py from the
cloned ProteinMPNN repo. Only the binder chain is designed; the target chain is
fixed. Output sequences are combined with the target sequence into per-design
multimer FASTAs consumed by part 3.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

from .setup import ensure_setup
from .util import (device_env, fasta_chain_lengths, pdb_chain_bfacts,
                   pdb_chain_ids, pdb_chain_residues, run_cmd, venv_python,
                   write_fasta)


def resolve_binder_chain(cfg: dict, ctx: dict, chains: list[str],
                         pdb: Path | None = None) -> str:
    """Pick the binder (designed) chain of a diffused complex PDB.

    Order of evidence:
      1. explicit `binder_chain` setting (not "auto")
      2. RFdiffusion marks diffused/binder residues with B-factor 0 in the
         output PDB -> the chain whose mean B-factor < 0.5 is the binder
      3. the single chain that is not the receptor chain from the contig layout
      4. the contig-layout binder chain if present
      5. first chain
    """
    setting = str(cfg.get("binder_chain", "auto"))
    if setting != "auto":
        return setting
    if pdb is not None and chains:
        bf = pdb_chain_bfacts(pdb)
        low = [c for c in chains if c in bf and bf[c] < 0.5]
        if len(low) == 1:
            return low[0]
    receptor = ctx.get("receptor_chain")
    others = [c for c in chains if c != receptor] if receptor else []
    if len(others) == 1:
        return others[0]
    for c in ctx.get("binder_chains", []):
        if c in chains:
            return c
    return chains[0] if chains else "A"


def gather_backbones(settings, cfg: dict, ctx: dict) -> list[Path]:
    if ctx.get("diffuse_outputs"):
        pdbs = list(ctx["diffuse_outputs"])
    else:
        pattern = cfg.get("input_glob") or str(settings.stage_dir("1_diffusion") / "outputs" / "*.pdb")
        p = Path(pattern)
        if not p.is_absolute():
            p = settings.root / p
        pdbs = sorted(p.parent.glob(p.name))
    max_bb = cfg.get("max_backbones")
    if max_bb:
        pdbs = pdbs[: int(max_bb)]
    return pdbs


def parse_mpnn_fasta(fasta: Path, num_seq: int) -> list[dict]:
    records = []
    header, seq = None, []
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header, seq = line[1:], []
        else:
            seq.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq)))
    samples = []
    for header, seq in records:
        if "sample=" not in header:
            continue
        score = re.search(r"score=([-\d.]+)", header)
        global_score = re.search(r"global_score=([-\d.]+)", header)
        recovery = re.search(r"seq_recovery=([-\d.]+)", header)
        samples.append({
            "seq": seq,
            "score": float(score.group(1)) if score else None,
            "global_score": float(global_score.group(1)) if global_score else None,
            "seq_recovery": float(recovery.group(1)) if recovery else None,
        })
        if len(samples) >= num_seq:
            break
    return samples


def run(settings, ctx: dict) -> list[dict]:
    logger = ctx["logger"]
    part = "part2_proteinmpnn"
    cfg = settings.require(part)
    workdir = settings.workdir
    stage = settings.stage_dir("2_mpnn")
    pdbs_dir = stage / "pdbs"
    fastas_dir = settings.stage_dir("3_refold") / "fastas"
    stage.mkdir(parents=True, exist_ok=True)
    fastas_dir.mkdir(parents=True, exist_ok=True)

    backbones = gather_backbones(settings, cfg, ctx)
    if not backbones:
        logger.warning("[mpnn] no input backbones found; nothing to do")
        ctx["mpnn_rows"] = []
        return []
    logger.info("[mpnn] %d backbones to design", len(backbones))

    repo_dir = settings.path(cfg["setup"]["repo_dir"])
    venv = settings.path(cfg["setup"]["venv"])
    parsed_jsonl = stage / "parsed.jsonl"
    chain_id_jsonl = stage / "chain_id.jsonl"
    out_folder = stage / "out"
    num_seq = int(cfg.get("num_seq_per_target", 4))

    if ctx["dry_run"]:
        stems = [p.stem for p in backbones]
        binder_chain = resolve_binder_chain(cfg, ctx, [], pdb=None)
        ctx["binder_chain"] = binder_chain
        chain_id = {stem: [[binder_chain], []] for stem in stems}
        logger.info("[mpnn][dry-run] chain_id.jsonl would contain:\n%s", json.dumps(chain_id))
        parse_cmd = [str(venv_python(venv)), str(repo_dir / "helper_scripts" / "parse_multiple_chains.py"),
                     "--input_path", str(pdbs_dir), "--output_path", str(parsed_jsonl)]
        mpnn_cmd = [str(venv_python(venv)), str(repo_dir / "protein_mpnn_run.py"),
                    "--jsonl_path", str(parsed_jsonl), "--chain_id_jsonl", str(chain_id_jsonl),
                    "--out_folder", str(out_folder), "--num_seq_per_target", str(num_seq),
                    "--sampling_temp", str(cfg.get("sampling_temp", "0.1")),
                    "--batch_size", str(int(cfg.get("batch_size", 1))),
                    "--model_name", str(cfg.get("model_name", "v_48_020")),
                    "--path_to_model_weights", str(repo_dir / "vanilla_model_weights"),
                    "--seed", str(settings.seed)]
        run_cmd(parse_cmd, cwd=repo_dir, env=device_env(settings.device), logger=logger, dry_run=True)
        run_cmd(mpnn_cmd, cwd=repo_dir, env=device_env(settings.device), logger=logger, dry_run=True)
        ctx["manifest"].append({"stage": "mpnn", "mode": "dry-run",
                                "parse_command": parse_cmd, "mpnn_command": mpnn_cmd})
        ctx["mpnn_rows"] = [
            {"design": stem, "seq_id": k, "score": None, "global_score": None,
             "seq_recovery": None, "binder_seq": "", "target_seq": "",
             "fasta": str(fastas_dir / f"{stem}_s{k}.fasta"), "fasta_stem": f"{stem}_s{k}",
             "binder_chain": binder_chain, "target_chain": ctx.get("receptor_chain", "B")}
            for stem in stems for k in range(num_seq)
        ]
        return ctx["mpnn_rows"]

    pdbs_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    used = set()
    for pdb in backbones:
        stem = pdb.stem
        if stem in used:
            stem = f"{stem}_{len(used)}"
        used.add(stem)
        dst = pdbs_dir / f"{stem}.pdb"
        shutil.copy(pdb, dst)
        copied.append((stem, dst))

    binder_chain = resolve_binder_chain(cfg, ctx, pdb_chain_ids(copied[0][1]), pdb=copied[0][1])
    ctx["binder_chain"] = binder_chain
    logger.info("[mpnn] designed (binder) chain: %s (chains in complex: %s)",
                binder_chain, pdb_chain_ids(copied[0][1]))

    if ctx["mock"] and "mpnn" in ctx["mock"]:
        rows = mock_designs(settings, cfg, ctx, copied, binder_chain, fastas_dir)
        ctx["mpnn_rows"] = rows
        ctx["manifest"].append({"stage": "mpnn", "mode": "mock", "rows": len(rows)})
        return rows

    ensure_setup(settings, part, logger, dry_run=ctx["dry_run"])

    parse_cmd = [
        str(venv_python(venv)), str(repo_dir / "helper_scripts" / "parse_multiple_chains.py"),
        "--input_path", str(pdbs_dir), "--output_path", str(parsed_jsonl),
    ]
    run_cmd(parse_cmd, cwd=repo_dir, env=device_env(settings.device),
            logger=logger, dry_run=ctx["dry_run"])

    chain_id = {}
    for stem, pdb in copied:
        chains = pdb_chain_ids(pdb)
        designed = [binder_chain] if binder_chain in chains else [chains[0]]
        fixed = sorted(c for c in chains if c not in designed)
        chain_id[stem] = [designed, fixed]
    # protein_mpnn_run.py keeps only the last line of this file, so write one JSON object
    chain_id_jsonl.write_text(json.dumps(chain_id) + "\n")

    mpnn_cmd = [
        str(venv_python(venv)), str(repo_dir / "protein_mpnn_run.py"),
        "--jsonl_path", str(parsed_jsonl),
        "--chain_id_jsonl", str(chain_id_jsonl),
        "--out_folder", str(out_folder),
        "--num_seq_per_target", str(num_seq),
        "--sampling_temp", str(cfg.get("sampling_temp", "0.1")),
        "--batch_size", str(int(cfg.get("batch_size", 1))),
        "--model_name", str(cfg.get("model_name", "v_48_020")),
        "--path_to_model_weights", str(repo_dir / "vanilla_model_weights"),
        "--seed", str(settings.seed),
    ]
    run_cmd(mpnn_cmd, cwd=repo_dir, env=device_env(settings.device),
            logger=logger, dry_run=ctx["dry_run"])
    ctx["manifest"].append({
        "stage": "mpnn", "mode": "run",
        "parse_command": parse_cmd, "mpnn_command": mpnn_cmd, "cwd": str(repo_dir),
    })

    rows = collect_rows(settings, cfg, ctx, copied, binder_chain, fastas_dir, out_folder)
    ctx["mpnn_rows"] = rows
    logger.info("[mpnn] wrote %d designed complexes", len(rows))
    return rows


def collect_rows(settings, cfg, ctx, copied, binder_chain, fastas_dir, out_folder) -> list[dict]:
    logger = ctx["logger"]
    receptor_chain = ctx.get("receptor_chain")
    num_seq = int(cfg.get("num_seq_per_target", 4))
    rows = []
    tsv = settings.stage_dir("2_mpnn") / "designs.tsv"
    with tsv.open("w") as fh:
        fh.write("design\tseq_id\tscore\tglobal_score\tseq_recovery\tbinder_seq\ttarget_seq\tfasta\n")
        for stem, pdb in copied:
            chains = pdb_chain_ids(pdb)
            designed = [binder_chain] if binder_chain in chains else [chains[0]]
            fixed = sorted(c for c in chains if c not in designed)
            order = sorted(designed) + sorted(fixed)
            fasta = out_folder / "seqs" / f"{stem}.fa"
            if not fasta.exists():
                logger.warning("[mpnn] no sequence output for %s", stem)
                continue
            chain_lens = {c: len(pdb_chain_residues(pdb, c)) for c in chains}
            target_chain = (receptor_chain if receptor_chain in chains and receptor_chain != binder_chain
                            else (fixed[0] if fixed else chains[0]))
            target_res = pdb_chain_residues(pdb, target_chain)
            target_seq = "".join(target_res[k] for k in sorted(target_res))
            designed_lengths = [chain_lens[c] for c in sorted(designed)]
            for k, sample in enumerate(parse_mpnn_fasta(fasta, num_seq)):
                seq = sample["seq"]
                if "/" in seq:
                    segments = seq.split("/")
                else:
                    # MPNN emits only the designed chains, concatenated without separators
                    segments, pos = [], 0
                    for length in designed_lengths:
                        segments.append(seq[pos:pos + length])
                        pos += length
                if len(segments) != len(designed) or sum(designed_lengths) != len(seq):
                    logger.warning("[mpnn] %s sample %d: seq length %d does not match designed lengths %s",
                                   stem, k, len(seq), designed_lengths)
                    continue
                chain_seq = dict(zip(sorted(designed), segments))
                binder_seq = chain_seq.get(binder_chain, "")
                fa = fastas_dir / f"{stem}_s{k}.fasta"
                # ColabFold multimer input: ONE fasta record, chains joined by ':'
                write_fasta(fa, [(fa.stem, f"{target_seq}:{binder_seq}")])
                row = {
                    "design": stem, "seq_id": k, "score": sample["score"],
                    "global_score": sample["global_score"],
                    "seq_recovery": sample["seq_recovery"],
                    "binder_seq": binder_seq, "target_seq": target_seq,
                    "fasta": str(fa), "fasta_stem": fa.stem,
                    "binder_chain": binder_chain, "target_chain": target_chain,
                }
                rows.append(row)
                fh.write("\t".join(str(row[c]) for c in
                          ("design", "seq_id", "score", "global_score", "seq_recovery",
                           "binder_seq", "target_seq", "fasta")) + "\n")
    return rows


def mock_designs(settings, cfg, ctx, copied, binder_chain, fastas_dir) -> list[dict]:
    receptor_chain = ctx.get("receptor_chain", "B")
    num_seq = int(cfg.get("num_seq_per_target", 4))
    rng = random.Random(settings.seed + 777)
    rows = []
    for stem, pdb in copied:
        chains = pdb_chain_ids(pdb)
        binder_res = pdb_chain_residues(pdb, binder_chain) if binder_chain in chains else {}
        receptor_chain_eff = next((c for c in chains if c != binder_chain), chains[0])
        target_res = pdb_chain_residues(pdb, receptor_chain_eff)
        target_seq = "".join(target_res[k] for k in sorted(target_res))
        binder_len = len(binder_res)
        for k in range(num_seq):
            binder_seq = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(binder_len))
            fa = fastas_dir / f"{stem}_s{k}.fasta"
            write_fasta(fa, [(fa.stem, f"{target_seq}:{binder_seq}")])
            rows.append({
                "design": stem, "seq_id": k, "score": None, "global_score": None,
                "seq_recovery": None, "binder_seq": binder_seq, "target_seq": target_seq,
                "fasta": str(fa), "fasta_stem": fa.stem,
                "binder_chain": binder_chain, "target_chain": receptor_chain_eff,
            })
    return rows
