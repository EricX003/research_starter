On most HPC clusters, you will automatically be generated a small “home directory” in addition to a larger partition on one or more “storage directories.” The home directory is designed to hold a small amount of private data (e.g. small codebases and configuration files), and typically have a limit on the order of 10GB. For convenience, you can soft-link commonly accessed data into your home directory to give the illusion that everything is in your home while not consuming all of your home directory storage. The following commands will set up your cache and conda directories to reside in storage while being accessible from home. 

```bash
STORAGE=/project/osprey/scratch/e.xing #obv change this depending on your ldap
ln -s $STORAGE $HOME/storage
ln -s $STORAGE/.cache $HOME/.cache
ln -s $STORAGE/.conda $HOME/conda
```

After this, you can run the following to install conda:

```bash
PREFIX="$HOME"

mkdir -p $PREFIX/conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $PREFIX/conda/conda.sh
bash $PREFIX/conda/conda.sh -b -u -p $PREFIX/conda/
rm -rf $PREFIX/conda/conda.sh

#chmod 700 $PREFIX/conda/bin/conda
#chmod 700 $PREFIX/conda/bin/python
$PREFIX/conda/bin/conda init bash
$PREFIX/conda/bin/conda init zsh
exec $SHELL
conda init
```

This script also works to download conda onto any of the standalone machines (crow, raven, and albatross).

Conda is a commonly used package manager used across all kinds of python workflows. You can think of it like venv+pip, as it manages both environmental isolation and package installation. Once you’ve installed conda, your default conda environment will be one called “base.” It’s generally ill-advised to install packages to base, and instead to create new environments:

```bash
conda create -n my-project python=3.11
```

Once the environment has been created, you can activate it and begin installing any pacakges:

```bash
conda install pip
pip install numpy torch …
```

You can share conda environments with friends and family by running:

```bash
conda env export > FILENAME.yml
```

And create environments from these files

```bash
conda env create -f FILENAME.yml
```

Find more documentation on common conda activities here.

EIT, like most HPC clusters, uses a number of “login nodes” to direct computational loads onto the appropriate compute nodes. Since login nodes serve the traffic of all users, you should always refrain from running anything computationally expensive on the login node. To use compute on EIT, you must queue a job using the SLURM job scheduler. The SLURM software manages the requests of all cluster users, and allocates compute to jobs once resources are available. 

Our lab’s partition is called `condo-jacobsn`, and jobs on this partition should use the `engr-lab-jacobsn` account. It contains one node called `osprey` with 16 CPU cores, 375GB of allocatable memory, and three NVIDIA A40 GPUs. This node is a dedicated condo purchased by the big boss himself many moons ago. You can check the current state of the partition yourself:

```bash
sinfo -p condo-jacobsn
scontrol show node osprey
```

EIT also has shared research GPU partitions that are useful when `osprey` is busy or when you need a different kind of GPU. `general-gpu` currently contains A40, A6000, A100, A100 SXM4, H100, and RTX 2080 GPUs, and permits jobs up to seven days. `general-gpu-long` contains the same GPUs except for the H100 and permits jobs up to twenty-one days (please do not exercise this it's not that serious).

```bash
sinfo -p general-gpu,general-gpu-long
sinfo -p general-gpu,general-gpu-long -N -o "%P %N %G %T"
```

Use `--partition=general-gpu` or `--partition=general-gpu-long` in place of `--partition=condo-jacobsn`.

```bash
#SBATCH --partition=general-gpu
#SBATCH --account=engr-lab-jacobsn
#SBATCH --gpus=1
```

If the program requires a particular GPU model, include its SLURM resource name. For example, the following requests one A100 SXM4:

```bash
#SBATCH --partition=general-gpu
#SBATCH --account=engr-lab-jacobsn
#SBATCH --gpus=a100-sxm4:1
```

Other currently valid model names include `a40`, `a6000`, `a100`, `h100`, and `rtx2080`. The H100 is currently only in `general-gpu`, so an H100 job must have a time limit of seven days or less.

Interactive jobs allow you to access compute resources through a shell CLI. This makes them easy to use because running programs becomes interactive. The following command requests one A40 GPU, four CPU cores, 32GB of memory, and two hours of compute time on the lab condo:

```bash
srun \
    --partition=condo-jacobsn \
    --account=engr-lab-jacobsn \
    --gpus=a40:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=2:00:00 \
    --pty bash -l
```

This command may wait if the requested resources are already in use. Once it starts, your shell is running on the compute node and you can activate a conda environment, run Python, or use `nvidia-smi` to inspect your assigned GPU. Run `exit` when you are finished so that SLURM can release the resources for some other job or to collect dust. 

Interactive jobs are useful for debugging, but sometimes you want a job that runs some pre-validated program and terminates after that workload is completed. Put the following in a file such as `train.sh:

```bash
#!/bin/bash
#SBATCH --job-name=my-project
#SBATCH --partition=condo-jacobsn
#SBATCH --account=engr-lab-jacobsn
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

source "$HOME/conda/etc/profile.d/conda.sh"
conda activate my-project

python train.py
```

Lines beginning with `#SBATCH` describe the resources needed by the job. They go after the shebang line. Submit the script from the directory containing your code:

```bash
sbatch < train.sh
```

This command should tell you that your job has been queued. SLURM substitutes the job name for `%x` and the job ID for `%j`, so the example above will write its terminal output to a file such as `my-project-12345.out`. Batch jobs keep running after you disconnect from the cluster.

These commands cover most day-to-day job management:

```bash
squeue --me
scontrol show job JOBID
scancel JOBID
sacct -j JOBID --format=JobID,State,Elapsed,AllocTRES,ExitCode
```

`squeue --me` lists your queued and running jobs. `scontrol` shows the complete request and, for a pending job, the reason that it has not started. `scancel` stops a job, and `sacct` shows information about current and completed jobs. Replace `JOBID` with the number printed by `sbatch`.

`--cpus-per-task` controls CPU cores, `--mem` controls system memory (RAM), `--gpus` controls the number of GPUs, and `--time` sets the maximum runtime. A job that exceeds its memory or time limit will be stopped. 

Training big vision models (which most of you all will certainly do) requires special considerations to the CPU resources compared to other training workflows. This is because images are really big and clunky to process compared to other modalities like text. Before image data can be sent to the GPU for training, it has to be read (I/O limited) and processed (compute limited). If preparing the next batch takes longer than training on the current one, the GPU will sit idle. Generally you should try to reduce these cheap bottlenecks to maximize the utilization of the GPU. 

The `num_workers` argument controls the number of DataLoader worker processes. With `num_workers=0`, the training process loads every batch itself. A positive value creates that many additional processes to read samples, apply dataset transforms, combine samples into batches, and prefetch those batches (and do whatever other weird things you want by hooking into the dataloading code). 

Setting `pin_memory=True` is always a good optimization on CUDA-enabled GPUs as it puts tensors into efficiently indexed CPU memory for fast transfer.

For a single-GPU job with one training process, a reasonable starting point is to request one CPU core for the training process plus one core for each DataLoader worker. For example:

```bash
#SBATCH --cpus-per-task=5
```

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
)
```

Occasionally there's some black magic that happens but in general you should increase the number of workers until the training throughput stops improving. Depending on the I/O bottleneck (can only process data as fast as it can be read) this is between 12-20.

Some image transforms and numerical libraries create their own CPU threads inside each worker. This can accidentally turn four workers into many more than four active threads. For workloads where this happens, limit common numerical libraries to one thread per process otherwise you get excessive overhead from round-robining the threads:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
...
```

You can also derive the worker count from the resources assigned by SLURM so that the code and batch script remain consistent across various CPU constraints, although it's generally a stupid idea to train on a job that has insufficient CPU resources because CPUs are cheap and GPUs are expensive:

```python
import os

allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
num_workers = max(0, allocated_cpus - 1)

loader = DataLoader(
    dataset,
    batch_size=X,
    num_workers=num_workers,
)
```

This calculation assumes one training process. Distributed training across many GPUs normally creates one training process and one DataLoader per GPU, so its total worker count and CPU requirements must account for all of those processes. You should probably consult your mentoring PhD student for guidance on this before you launch any large multi-node or multi-GPU runs. 
