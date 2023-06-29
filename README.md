# GPU energy usage counter for AMD/ROCm

Reads the current GPU energy counters for AMD GPU cards using the 
[ROCm SMI library](https://github.com/RadeonOpenCompute/rocm_smi_lib/). 

## Installation

To compile just run `make`.

**NOTE:** on LUMI it is pre-installed in `/appl/local/csc/soft/ai/bin/gpu-energy`.

## Usage

Print current counter values for all visible devices:

```bash
gpu-energy
```

Save counters to a temporary file for later use:

```bash
gpu-energy --save [filename]
```

if no filename is given, it will try to figure out a good name based
on the Slurm environment.

Print energy usage difference since last save:

```bash
gpu-energy --diff [filename]
```

Typical usage in a Slurm script:


```bash
gpu-energy --save

# run job here

gpu-energy --diff
```

Multi node job:

```bash
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 gpu-energy --save

# run job here

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 gpu-energy --diff
```

If you're using a module (like CSC's `pytorch`) that sets the
`SLURM_MPI_TYPE` environment variable, you need to run it like this
(otherwise it will not detect MPI and will not calculate the energy
sum over nodes).

```bash
srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 gpu-energy --save

# run job here

srun --mpi=cray_shasta --ntasks=$SLURM_NNODES --ntasks-per-node=1 gpu-energy --diff
```
