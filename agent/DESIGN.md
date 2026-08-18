# Design Intuition
Why the research should work, in the terms used to convince a skeptical colleague at a whiteboard. USAGE.md says how to run the code; this says what the code is trying to be true about. Intuition first — equations and hyperparameters belong in `configs/experiment/` and the paper draft.

Below the line:
- **Three-sentence pitch** — problem, mechanism, consequence, no hedging. If it takes more, the design is not designed yet; write the three sentences first and let the rest elaborate them.
- **The problem** — setting, observed failure, evidence it is not a tuning artifact, and why prior work does not already fix it. A problem no existing method fails at is not a problem statement.
- **Core intuition** — the idea as one image or analogy; the causal story for why it helps; and what it predicts that the baseline does not. That last item is what separates "it worked" from "it worked for the reason claimed".
- **Why this might be wrong** — per claim, the cheapest falsifying observation and whether it has been run. Plus confounds that could produce the same gain (parameter count, effective batch size, data seen, eval protocol) and the most boring explanation for a win, with the ablation that rules it out.
- **Load-bearing decisions** — only choices where a different answer would move a number: alternatives rejected, why, and when to revisit. Everything else is a config value. This is what stops a future agent re-litigating a settled question.
- **Evaluation logic** — why the metrics can detect the mechanism, what they are blind to, which diagnostics test it directly, and which protocol details break comparability if they drift. The numbers themselves go in SOTA.md.
- **Idea → code** — where each component lives. A wrong pointer costs more than a missing one.
- **Open questions** — ranked by what a wrong guess costs. Promote one to a direction in ACTIVE.md rather than answering it here.

Maintenance:
- Rewrite in place, never append; superseded versions go to `agent_legacy/design/YYYY-MM-DD_<slug>.md`. This file states the current belief, not its history.
- A mechanism added without its failure modes is an unfinished edit.
- Directions graduate here from ACTIVE.md once they survive their success criterion.

##### MODIFY UNDER THIS LINE #####
