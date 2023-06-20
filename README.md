# GPU energy usage counter for AMD/ROCm

Using the [ROCm SMI library](https://github.com/RadeonOpenCompute/rocm_smi_lib/).

To compile just run `make`.

Usage example:

```bash
GPUENERGY=$(gpu-energy)

# run job here

gpu-energy $GPUENERGY
```
