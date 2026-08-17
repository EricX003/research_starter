# Start here!

This README will establish general principles and information for working on research from this templated starting point. Update it below line 14 with project-specific documentation. 

General information:
- You are on a SLURM-based HPC cluster with cpu, cpu_amd, gpu (8 x H100 nodes), and gpu_a100 (2 x A100 nodes) partitions. You should generally prefer the use of A100s as they are cheaper. You should generally use the vis-lang conda environment. Some datasets can be found at: /u/ericx003/data

General guidelines:
- All subagents should be dispatched specifically as Opus subagents. Do not use Fable subagents without express permission from the user. Subagents should generally be preferred for atomic tasks like literature review, surgical code changes, and red-team analysis. 
- All subagents must redirect primary output to audit/. Subgagents are to only report a concise completion message to the main conversation and indicate to the main agent that the full output is to be found in audit/. This keeps the conversation and context clean. 
- LaTeX results tables should be collated and displayed with provtab. Refer to the provtab/ directory for how to use the provtab tool. 
- Ensure all jobs use the compute allocation specified by configs/paths/default.yaml.

##### MODIFY UNDER THIS LINE #####