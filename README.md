# De novo binder design pipeline (RFdiffusion → ProteinMPNN → ColabFold AF2-multimer)

A three-part computational pipeline for protein de novo binder design against a
target crystal structure:

1. **Part 1 — `part1_diffusion/` (RFdiffusion):** diffuse backbone binders against the target
   (contig = fixed target segment + diffused binder segment, hotspot-conditioned).
2. **Part 2 — `part2_proteinmpnn/` (ProteinMPNN):** inverse folding — design binder sequences on the
   diffused backbones with the target chain fixed.
3. **Part 3 — `part3_colabfold/` (ColabFold):** refold each target+binder complex with
   AF2-multimer (`alphafold2_multimer_v3`).

Orchestrated by **`run_pipeline.py`**, configured by **`settings.json`**. Evaluation produces an
**ipAE vs pLDDT graph**; a design **passes if pLDDT > 0.7 and ipAE < 10 Å**.

All environments are created with [uv](https://docs.astral.sh/uv/); each part clones its
upstream repo into `vendor/` and builds its own venv in `envs/`.

## Layout

```
settings.json            # all settings (single source of truth)
run_pipeline.py          # orchestrator CLI
pipeline/                # python package used by the orchestrator
  config.py              #   settings loading, --smoke overrides, --device flag
  util.py                #   logging, subprocess, device env, PDB/FASTA helpers
  setup.py               #   per-part setup invocation + markers
  stage_diffuse.py       #   part 1
  stage_mpnn.py          #   part 2
  stage_refold.py        #   part 3
  metrics.py             #   eval: ipAE/pLDDT, pass/fail, plot
part1_diffusion/setup.sh # clone RFdiffusion + uv venv (torch/dgl +CUDA or CPU) + model ckpts
part2_proteinmpnn/setup.sh
part3_colabfold/setup.sh
tests/test_pipeline.py   # CPU smoke tests (mocked heavy stages; optional real MPNN-on-CPU)
data/example_target.pdb  # example target (2KL8, chain A, 79 residues)
```

Workdir layout (per run, e.g. `runs/example/`):

```
1_diffusion/outputs/design_*.pdb     # RFdiffusion backbone complexes (binder chain A, target chain B)
2_mpnn/pdbs, parsed.jsonl, chain_id.jsonl, out/seqs/*.fa, designs.tsv
3_refold/fastas/<design>_s<k>.fasta  # ONE record, chains joined by ':' -> seq1:seq2 (ColabFold multimer input)
3_refold/colabfold/*_scores_rank_001_*.json, *_unrelaxed_rank_001_*.pdb
4_eval/metrics.csv, summary.json, ipae_vs_plddt.png, passing/
manifest.json, pipeline.log
```

## Quickstart

Requires: `uv`, `git`, and on the GPU machine an NVIDIA driver with CUDA ≥ 12.1
(RTX 4090 works with the default `cu121` wheels).

```bash
# 1) CPU smoke test (no GPU, heavy stages mocked): validates orchestration + eval end to end
uv run python run_pipeline.py --settings settings.json --device cpu --smoke --mock

# 2) Dry run on any machine: prints every command that would be executed (GPU path)
uv run python run_pipeline.py --settings settings.json --device gpu --dry-run

# 3) Real run on the RTX 4090 (clones repos, builds venvs, downloads RFdiffusion
#    checkpoints and AF2 params on first use, then runs all stages):
uv run python run_pipeline.py --settings settings.json --device gpu
```

Run individual stages with `--stages setup,diffuse,mpnn,refold,eval` (or `all`).

### Flags

| flag | meaning |
|---|---|
| `--device gpu\|cpu` | `gpu` (default from settings) uses CUDA; `cpu` sets `CUDA_VISIBLE_DEVICES=""` / `JAX_PLATFORMS=cpu` so everything runs on CPU (smoke tests) |
| `--smoke` | applies the `smoke` overrides in `settings.json` (2 designs, tiny contig, 2 MPNN seqs, 1 recycle) |
| `--dry-run` | print all commands without executing |
| `--mock [diffuse,mpnn,refold]` | replace heavy stages with deterministic synthetic outputs (CPU testing); bare `--mock` mocks all three |
| `--workdir DIR` | override `paths.workdir` |
| `--force-setup` | re-run part setups |

## settings.json

- `paths.target_pdb` — target crystal structure; `part1_diffusion.target_chain` / `target_residues`
  select the fixed target segment (e.g. `A` + `1-79` → contig segment `A1-79/0`).
- `part1_diffusion.binder_length` — diffused binder length range, e.g. `50-80` → contig `50-80`.
  Alternatively set `part1_diffusion.contigs` explicitly, e.g. `["A1-150/0 70-100"]`.
- `part1_diffusion.hotspot_res` — target hotspot residues, e.g. `["A31","A35","A39"]`.
- `part1_diffusion.num_designs`, `diffusion_steps` (`diffuser.T`), `noise_scale_ca/frame`,
  `extra_args` (extra RFdiffusion hydra overrides).
- `part2_proteinmpnn` — `num_seq_per_target`, `sampling_temp`, `batch_size`, `model_name`
  (`v_48_020`), `binder_chain` (`auto` = detected from the diffused PDB: RFdiffusion marks
  binder residues with B-factor 0, so the chain with mean B-factor < 0.5 is the binder;
  set explicitly to override).
- `part3_colabfold` — `model_type` (default `alphafold2_multimer_v3`), `model_order`,
  `num_models`, `num_recycle`, `num_seeds`, `pair_mode`, `msa_mode` (default
  `single_sequence`: **local execution only, no MSA server**), `rank_by`, `skip_output`,
  `extra_args` (extra `colabfold_batch` flags). Each design FASTA is a single record with
  chains joined by `:` (`SEQ_target:SEQ_binder`), which ColabFold parses as a multimer
  complex. Refolding writes rank-1 PDBs (`*_unrelaxed_rank_001_*.pdb`, plus
  `*_relaxed_rank_001_*.pdb` with `--amber`) alongside the scores JSONs; passing PDBs are
  copied to `4_eval/passing/`.
- `eval` — `plddt_min` (0.7), `ipae_max` (10.0 Å), `plddt_scope` (`binder` or `complex`), `top_n`.
- `smoke` — overrides applied with `--smoke`.

Each part's `setup` block controls cloning/venv: `repo_url`, `repo_dir`, `venv`, `python`,
and for GPU `cuda_version` (default `cu121`), `torch_version` (`2.4.1`), `dgl_version` (`2.4.1`);
`part3_colabfold.setup.jax_version` / `jax_cuda` control the CUDA jaxlib.

## Evaluation metrics

For each design the rank-1 AF2-multimer scores JSON is parsed:

- **ipAE** = mean PAE over all cross-chain (target↔binder) residue pairs (Å).
- **pLDDT** = mean per-residue pLDDT normalized to 0–1, over the binder chain
  (`plddt_scope: binder`) or the whole complex.
- **pass** = `pLDDT > 0.7` **and** `ipAE < 10 Å`.

Outputs in `<workdir>/4_eval/`: `metrics.csv` (all designs), `summary.json`
(pass counts + top passing designs), `ipae_vs_plddt.png` (scatter with shaded pass
region and threshold lines), and `passing/` with the rank-1 PDBs of passing designs.

## Tests

```bash
uv run --group dev pytest tests/ -v          # mocked heavy stages, CPU only
# optional: real ProteinMPNN on CPU (downloads CPU torch into envs/part2)
PIPELINE_TEST_REAL_MPNN=1 uv run --group dev pytest tests/test_pipeline.py::test_real_mpnn_cpu -v
```

## Notes / troubleshooting

- RFdiffusion falls back to CPU automatically when no GPU is visible; `--device cpu` enforces that.
- RFdiffusion checkpoints are downloaded by `part1_diffusion/setup.sh` from the RosettaCommons
  server into `vendor/RFdiffusion/models/` (the complex/PPI model `Complex_base_ckpt.pt` is used
  automatically when hotspots are provided). Re-run with `--force-setup` to redo.
- ColabFold downloads AF2-multimer v3 parameters automatically on the first `colabfold_batch` run.
- MSA generation is **local only** by default (`msa_mode: single_sequence` — no MSA server is
  contacted). Change `part3_colabfold.msa_mode` to e.g. `mmseqs2_uniref_env` if you want server MSAs.
- `vendor/`, `envs/`, `runs/`, `.venv/` are gitignored — only pipeline code and settings are tracked.
