# FFTE Parameter Header

## Description

Defines compile-time constants used throughout the FFTE (Fast Fourier Transform in the East) library. These parameters control array sizes, blocking factors, and hardware-specific tuning.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAXNPU` | 65536 | Maximum supported number of processors |
| `NDA2` | 65536 | Maximum supported 2-D transform length per dimension |
| `NDA3` | 4096 | Maximum supported 3-D transform length per dimension |
| `NBLK` | 16 | Cache blocking factor for loop tiling |
| `NB` | 128 | Blocking parameter for NVIDIA GPU kernels |
| `NP` | 8 | Padding added to work arrays to avoid cache-line conflicts |
| `L2SIZE` | 8388608 | Assumed L2 cache size in bytes (8 MB) |

### Notes

- `NBLK` controls the tile size in blocked transpose and FFT operations. The optimal value depends on the L2 cache line size and associativity.
- `NP` pads work arrays (e.g., `B(NY+NP, *)`) so that consecutive columns do not alias to the same cache set.
- `L2SIZE` is used to decide whether a 1-D FFT is small enough to compute directly or must be decomposed using the six-step algorithm.
- These values should be tuned for the target hardware. The defaults are suitable for modern x86 processors.
