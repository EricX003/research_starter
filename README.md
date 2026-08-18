# Start here!

This README will establish general principles and information for working on research from this templated starting point. Update it from below the end of my guidelines with project-specific documentation. 

General information:
- You are on a SLURM-based HPC cluster with cpu, cpu_amd, gpu (8 x H100 nodes), and gpu_a100 (2 x A100 nodes) partitions. You should generally prefer the use of A100s as they are cheaper. You should generally use the vis-lang conda environment. Some datasets can be found at: /u/ericx003/data

How to use this starting directory:

General guidelines:
- All subagents should be dispatched specifically as Opus subagents. Do not use Fable subagents without express permission from the user. Subagents should generally be preferred for atomic tasks like literature review, surgical code changes, and red-team analysis. 
- All subagents must redirect primary output to audit/. Subgagents are to only report a concise completion message to the main conversation and indicate to the main agent that the full output is to be found in audit/. This keeps the conversation and context clean. 
- Ensure all jobs use the compute allocation specified by configs/paths/default.yaml.
- All work related to baseline implementation should go in baselines/
- All minor evaluations, smoke-tests, visualizations, etc. can go into scripts/. This directory should be regularly flushed to archive/ as things become stale. 

Documentation Guidelines:
- Documentation should be regularly pruned and maintained. Before ending a turn, while code runs after significant changes, and upon request you should thoroughly go through documentation and ensure it is concise, complete, and up-to-date. Old and incorrect documentation should be deleted or moved to agent_legacy if it is historically useful. 
- Code should be commented sparingly without excessive header comments. Header comments should be at most 3 sentences with usage examples.
- Inline comments should be used only when absolutely neccessary, and should generally not describe small bugfixes. Describing bugfixes is confusing for a human or agent that hasn't been completely in the loop. Inline comments should clarify logically potentially confusing or load-bearing choices. 
- LaTeX results tables should be collated and displayed with provtab. Refer to the provtab/ directory for how to use the provtab tool. 


##### MODIFY UNDER THIS LINE #####