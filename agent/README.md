# Agent Working Directory
Agent-facing state: what an agent needs to pick up work mid-project and act correctly. Not paper documentation, not a changelog. In every file here the preamble above `##### MODIFY UNDER THIS LINE #####` is the spec and stays fixed; project content goes below it.

Read order:
- **ACTIVE.md** — asks, open directions, blockers, retirements. Read first, update last.
- **DESIGN.md** — why the research should work, and what would falsify it. Read before changing approach.
- **USAGE.md** — setup, running, evaluation, reproduction. Read before touching code or launching.
- **SOTA.md** — the best ≤5 designs with provenance, plus baselines. Write to it the turn a qualifying run lands.
- **litreview/** — distilled prior work, one file per question.

Conventions:
- `agent/`, `agent_legacy/`, `archive/`, `audit/`, `*.json`, `*.parquet`, `*.ckpt` are gitignored. Nothing here is recoverable from git, so write down any path a result depends on.
- Prune, don't append. Delete what is stale; move stale-but-useful content to the matching `agent_legacy/` subfolder with a verdict and leave a one-line pointer. No verdict means deletion.
- Lit-review subagents write into `litreview/` directly. All other subagents write full output to `audit/` and report only a pointer.
- Before ending a turn that changed code or a direction's state, fix what these files got wrong.

##### MODIFY UNDER THIS LINE #####
