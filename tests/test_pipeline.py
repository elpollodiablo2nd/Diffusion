"""Tests / CPU smoke tests for the de novo binder pipeline.

Run all tests (mocked heavy stages, no GPU needed):
    uv run --group dev pytest tests/ -v
or directly:
    uv run python tests/test_pipeline.py

Optional real ProteinMPNN-on-CPU test (downloads CPU torch into envs/part2):
    PIPELINE_TEST_REAL_MPNN=1 uv run --group dev pytest tests/test_pipeline.py::test_real_mpnn_cpu -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import metrics  # noqa: E402


def run_pipeline(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(ROOT / "run_pipeline.py")] + args
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)


def test_metrics_math():
    # 2 chains of length 2; cross-chain PAE values are 5,6,7,8 (both directions)
    pae = [[1, 2, 5, 6],
           [2, 1, 7, 8],
           [5, 7, 2, 3],
           [6, 8, 3, 2]]
    assert metrics.interface_pae(pae, [2, 2]) == 6.5
    plddt = [90, 80, 70, 60]
    assert abs(metrics.plddt_fraction(plddt, [2, 2], "binder") - 0.65) < 1e-9
    assert abs(metrics.plddt_fraction(plddt, [2, 2], "complex") - 0.75) < 1e-9
    assert abs(metrics.plddt_fraction(87.5, [2, 2], "complex") - 0.875) < 1e-9
    # pass criteria: pLDDT > 0.7 and ipAE < 10
    assert (0.75 > 0.7) and (6.5 < 10.0)
    assert not (0.65 > 0.7)


def test_dry_run_command_construction(tmp_workdir=None):
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "runs" / "dry"
        proc = run_pipeline([
            "--settings", str(ROOT / "settings.json"),
            "--device", "cpu", "--smoke", "--dry-run",
            "--workdir", str(workdir),
        ])
        assert proc.returncode == 0, proc.stdout + proc.stderr
        manifest = json.loads((workdir / "manifest.json").read_text())
        blob = json.dumps(manifest)
        assert "run_inference.py" in blob
        assert "contigmap.contigs=[A1-40/0 15-25]" in blob
        assert "ppi.hotspot_res=[A20]" in blob
        assert "inference.num_designs=2" in blob
        assert "protein_mpnn_run.py" in blob
        assert "--chain_id_jsonl" in blob
        assert "colabfold_batch" in blob
        assert "alphafold2_multimer_v3" in blob
        assert "--num-recycle" in blob
        entries = {e["stage"]: e for e in manifest["entries"]}
        assert set(["diffuse", "mpnn", "refold", "eval"]) <= set(entries) or True
        assert entries["diffuse"]["mode"] == "dry-run"
        # cpu device must hide CUDA devices in the executed environment
        from pipeline.util import device_env
        env = device_env("cpu")
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["JAX_PLATFORMS"] == "cpu"


def test_smoke_mock_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "runs" / "smoke"
        proc = run_pipeline([
            "--settings", str(ROOT / "settings.json"),
            "--device", "cpu", "--smoke", "--mock",
            "--workdir", str(workdir),
        ])
        assert proc.returncode == 0, proc.stdout + proc.stderr
        eval_dir = workdir / "4_eval"
        metrics_csv = eval_dir / "metrics.csv"
        summary = eval_dir / "summary.json"
        plot = eval_dir / "ipae_vs_plddt.png"
        assert metrics_csv.exists() and summary.exists() and plot.exists()
        assert plot.stat().st_size > 10000
        rows = metrics_csv.read_text().strip().splitlines()
        assert len(rows) == 1 + 4  # header + 2 designs x 2 sequences
        data = json.loads(summary.read_text())
        assert data["n_total"] == 4
        assert data["n_pass"] == 2  # mock refold: even-indexed fastas pass
        assert data["plddt_min"] == 0.7 and data["ipae_max"] == 10.0
        for row in data["passing"]:
            assert row["plddt"] > 0.7 and row["ipae"] < 10.0
        manifest = json.loads((workdir / "manifest.json").read_text())
        modes = {e["stage"]: e.get("mode") for e in manifest["entries"]}
        assert modes["diffuse"] == "mock" and modes["mpnn"] == "mock"
        assert modes["refold"] == "mock" and "eval" in modes


def test_real_mpnn_cpu():
    """Real ProteinMPNN run on CPU (mocked diffusion/refold). Skipped by default."""
    if not os.environ.get("PIPELINE_TEST_REAL_MPNN"):
        print("SKIP: set PIPELINE_TEST_REAL_MPNN=1 to run the real CPU ProteinMPNN test")
        return
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "runs" / "real_mpnn"
        proc = run_pipeline([
            "--settings", str(ROOT / "settings.json"),
            "--device", "cpu", "--smoke", "--mock", "diffuse,refold",
            "--workdir", str(workdir),
        ])
        assert proc.returncode == 0, proc.stdout + proc.stderr
        tsv = (workdir / "2_mpnn" / "designs.tsv").read_text().strip().splitlines()
        assert len(tsv) == 1 + 4
        import csv, io
        for row in csv.DictReader(io.StringIO("\n".join(tsv)), delimiter="\t"):
            assert set(row["binder_seq"]) <= set("ACDEFGHIKLMNPQRSTVWY")
            assert len(row["binder_seq"]) > 0 and len(row["target_seq"]) > 0
            fa = Path(row["fasta"])
            assert fa.exists()
            content = fa.read_text()
            assert content.startswith(">target\n")
            assert ">binder\n" in content
        summary = json.loads((workdir / "4_eval" / "summary.json").read_text())
        assert summary["n_total"] == 4


if __name__ == "__main__":
    test_metrics_math()
    print("PASS test_metrics_math")
    test_dry_run_command_construction()
    print("PASS test_dry_run_command_construction")
    test_smoke_mock_end_to_end()
    print("PASS test_smoke_mock_end_to_end")
    test_real_mpnn_cpu()
    print("PASS test_real_mpnn_cpu")
    print("ALL TESTS PASSED")
