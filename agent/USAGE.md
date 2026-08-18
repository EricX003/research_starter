# Documentation For Usage & Operations
This document should map out current state of the codebase, guidelines for usage, and pointers on how to reproduce core functionalities. A human or agent should be able to read this document and cleanly be able to set up everything including data preprocessing, training, evaluation, ablation studies, and visualization. This document should be regularly pruned to ensure it gives recommendations that describe the code as it exists now rather than as it once did.

Below the line:
- **Harness gaps**, first and brief: what is referenced but not yet implemented, so a fresh clone knows what will fail. Delete rows as they are filled, then the section.
- **Environment**: conda env, every env var the configs read and what breaks without it, marker files the entrypoint requires, where datasets live on this cluster.
- **Quickstart** in increasing cost: a CPU smoke test that exercises the full path, single GPU, multi GPU. Copy-pasteable as written.
- **Config composition**: the entrypoint config, the data contract it enforces, package syntax for wiring datamodules into train/val/test, group vs leaf overrides, and any ordering that leaks into behaviour.
- **Data**: one row per datamodule — expected input layout, what it is for, which is the default. Then the preprocessing to reach that layout, and per-module behaviour that silently changes what the model sees (filters, dedup, resampling, how a streaming epoch is defined).
- **SLURM**: partitions and which to prefer, the allocation to charge, a working sbatch script, and the launcher/config invariants that must agree.
- **Outputs**: run directory layout, what lands where, how loggers are selected.
- **Evaluation**: how to reproduce a reported number, and how the reported protocol differs from any cheaper in-training proxy. Conflating the two makes SOTA.md meaningless.
- **Sweeps and visualization**: multirun usage, one experiment config per claim, and how tables and figures are produced (provtab for LaTeX).
- **Gotchas**: the sharp edges that cost an hour — defaults that are not what you want, values that must be set per experiment, prompts that hang batch jobs, anything pinned at import time.
- **Where things go**: the directory map, so new code and scratch work land correctly without asking.

Maintenance:
- Update in the same turn as the change that invalidated it; a wrong instruction here gets followed.
- Document only what has actually been run. Mark untested commands as untested, or leave them out.
- Cite `path:line` for claims about behaviour, and re-verify pointers when the target file changes.
- Write down every path a result depends on — `agent/`, `logs/`, `audit/`, `*.json`, `*.parquet`, `*.ckpt` are gitignored.
- Delete stale instructions rather than annotating them. Exception: a retired procedure that produced an artifact still in use goes to `agent_legacy/usage/` with a note on what replaced it.
- Keep it readable start to finish. A section outgrowing that should push detail into a docstring or a runnable `scripts/` example, leaving a pointer here.

##### MODIFY UNDER THIS LINE #####
