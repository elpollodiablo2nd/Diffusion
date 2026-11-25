"""Shared helpers: logging, subprocess execution, device handling, PDB/FASTA IO."""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Optional

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def setup_logger(name: str, logfile: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        filehandler = logging.FileHandler(logfile)
        filehandler.setFormatter(fmt)
        logger.addHandler(filehandler)
    return logger


def device_env(device: str, base: Optional[dict] = None) -> dict:
    """Return an environment for the given device.

    cpu mode hides all CUDA devices so torch/JAX/DGL fall back to CPU.
    """
    env = dict(base if base is not None else os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("JAX_PLATFORMS", None)
    return env


def run_cmd(
    cmd: Iterable[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
    dry_run: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    log = logger or logging.getLogger("pipeline")
    log.info("$ %s", shlex.join(cmd))
    if cwd is not None:
        log.info("  (cwd: %s)", cwd)
    if dry_run:
        log.info("  [dry-run] command not executed")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env,
                          capture_output=True, text=True)
    if proc.stdout:
        log.info(proc.stdout.rstrip())
    if proc.returncode != 0:
        if proc.stderr:
            log.error(proc.stderr.rstrip()[-4000:])
        if check:
            raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    elif proc.stderr:
        log.info(proc.stderr.rstrip()[-2000:])
    return proc


def venv_python(venv: Path) -> Path:
    return Path(venv) / "bin" / "python"


def pdb_chain_ids(pdb: Path) -> list[str]:
    chains = []
    for line in Path(pdb).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            ch = line[21]
            if ch not in chains:
                chains.append(ch)
    return chains


def pdb_chain_residues(pdb: Path, chain: str) -> dict[int, str]:
    """Map residue number -> 1-letter AA for a chain (standard residues only)."""
    residues: dict[int, str] = {}
    for line in Path(pdb).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain:
            continue
        if line[16] not in (" ", "A"):
            continue
        try:
            resi = int(line[22:26])
        except ValueError:
            continue
        aa3 = line[17:20].strip()
        if aa3 in AA3TO1:
            residues[resi] = AA3TO1[aa3]
    return residues


def pdb_chain_sequence(pdb: Path, chain: str) -> str:
    residues = pdb_chain_residues(pdb, chain)
    return "".join(residues[k] for k in sorted(residues))


def pdb_chain_bfacts(pdb: Path) -> dict[str, float]:
    """Mean B-factor per chain. RFdiffusion marks diffused (binder) residues with B-factor 0."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for line in Path(pdb).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        ch = line[21]
        try:
            b = float(line[60:66])
        except ValueError:
            b = 1.0
        sums[ch] = sums.get(ch, 0.0) + b
        counts[ch] = counts.get(ch, 0) + 1
    return {ch: sums[ch] / counts[ch] for ch in sums}


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header, seq = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header, seq = line[1:].strip(), []
        else:
            seq.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def fasta_chain_lengths(path: Path) -> list[int]:
    """Chain lengths of a (multimer) FASTA. ColabFold splits each record's
    sequence on ':' into the chains of the complex."""
    lengths: list[int] = []
    for _, seq in read_fasta(path):
        lengths.extend(len(part) for part in seq.split(":"))
    return lengths


def write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n{seq}\n")


def parse_residue_range(spec: str) -> tuple[int, int]:
    spec = str(spec).strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
    else:
        lo = hi = spec
    return int(lo), int(hi)
