# Literature Review Index
Distilled prior work: enough to check a claim about the literature without re-reading the papers, and to stop repeated subagents rediscovering the same things. A synthesis store, not a bibliography — titles belong in the `.bib`.

Below the line, and in this folder:
- **One file per topic, not per paper**: `<topic-slug>.md`, where a topic is a question the project actually needs answered, so the file can end in a position rather than a pile of summaries. A paper relevant to several topics gets a line in each.
- **This file holds the index**: slug, the question it answers, last refresh date, confidence. Flag a file older than the current design direction rather than trusting it.
- **Each topic file**: the question as it matters to us; a paper table (venue, year, one-line claim, what we take or reject); a synthesis that takes a position; **what this means for us**, tied to a DESIGN.md section or an ACTIVE.md direction; the gap nobody has filled, which is where our contribution has to sit; and search coverage — what was searched, where, and what was deliberately not read.

Maintenance:
- Lit review is subagent work, and lit-review subagents write here directly — the topic file, plus raw per-paper notes and search transcripts worth keeping as `<topic-slug>.notes.md`. They report only a concise completion message and the paths. 
- Record what a paper is for us, not what its abstract says.
- Quoted numbers are that paper's, under that paper's protocol. Tag them reported-not-reproduced, and never move one into SOTA.md without reproducing it in `baselines/` or labelling it there.
- Refresh a topic when its question becomes load-bearing, not on a schedule. Retire superseded reviews to `agent_legacy/litreview/` with a note on what replaced them, and drop the index row.

##### MODIFY UNDER THIS LINE #####
