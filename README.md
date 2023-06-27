# GPU energy usage counter for AMD/ROCm

Reads the current GPU energy counters for AMD GPU cards using the [ROCm SMI library](https://github.com/RadeonOpenCompute/rocm_smi_lib/). 

## Installation

To compile just run `make`.

## Usage

Print current counter values for all visible devices:

```bash
gpu-energy
```

Save counters to a temporary file for later use:

```bash
gpu-energy --save [filename]
```

if no filename is given, it will try to figure out a good name based on the Slurm environment.

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
srun gpu-energy --save

# run job here

srun gpu-energy --diff
```
