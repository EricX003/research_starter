# State of the Art Board
The best five designs this project has produced, ranked, with the baseline numbers that give them context. The one place to answer "what is our best result, and is it actually better than what existed". A leaderboard, not a notebook: work in progress lives in ACTIVE.md, evicted designs in `agent_legacy/sota/`.

Below the line:
- **Metric contract**, written before any number is admitted: exact definition, eval split and size, protocol (for retrieval, single- vs multi-caption ground truth, gallery size, query direction), the code path that computes it, and which direction is better. Numbers from different contracts never share a table.
- **Baselines**, split into author-reported and reproduced-here — the gap between the two is itself a finding. Reproduced ones need full provenance and a pointer into `baselines/`. Baselines are context and are never evicted to make room for a design.
- **At most five ranked entries**: rank, short name, the one-line design idea, primary metric, and deltas against the relevant baseline and the previous best.
- **Provenance per entry**, non-negotiable: commit SHA, launch command or experiment config, run dir, wandb id, seed(s), eval script, hardware and wall-clock, date. An entry that cannot be re-run from its own row does not belong — `agent/` and every artifact type are gitignored, so the row is the record.
- **Variance** where known: seeds run, spread, and whether the top gap exceeds it. Say so explicitly when a ranking sits inside the noise floor.
- **Eviction log**: what was displaced, by what, when, and where its detail went.

Maintenance:
- Update in the turn a run clears the current fifth place. An unrecorded winning run is a lost result.
- Admission is gated on the checklist, not enthusiasm: the contract must already cover the number, provenance must be complete, and the eval must have used the reported protocol rather than the cheaper in-training proxy. Otherwise it stays in ACTIVE.md.
- A sixth qualifier displaces the weakest: write it to `agent_legacy/sota/YYYY-MM-DD_<slug>.md` with provenance and what superseded it, then log the eviction. The board never exceeds five.
- If the metric contract changes, every number on the board is invalid. Archive the whole board under the old contract, state the new one, and rebuild. Never mix.
- Paper tables are collated from these rows with provtab, so keep them clean enough to be the source.

##### MODIFY UNDER THIS LINE #####
