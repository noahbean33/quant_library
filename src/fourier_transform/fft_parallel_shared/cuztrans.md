# GPU Blocked Matrix Transposition (CUDA Fortran)

## Overview

CUDA Fortran routines for transposing complex matrices on NVIDIA GPUs. Contains two routines:

1. **`ZTRANS`** — Blocked transpose of an `NX × NY` matrix into `NY × NX`, using `NB × NB` tile blocks with shared-memory staging (`C_d`) for coalesced memory access.
2. **`MZTRANS`** — Multi-row (batched) transpose of an `NS × NX × NY` array into `NS × NY × NX`, transposing the last two dimensions while keeping the first (row) dimension intact.

### ZTRANS Pseudocode

```
function ZTRANS(A_d, B_d, NX, NY):
    // A_d: NX × NY device array (input)
    // B_d: NY × NX device array (output)
    // NB: tile block size (from param.h)

    for ii = 0 to NX-1 step NB:
        for jj = 0 to NY-1 step NB:
            // Phase 1: Load tile into shared memory C_d
            GPU kernel:
                for i = ii to min(ii+NB-1, NX-1):
                    for j = jj to min(jj+NB-1, NY-1):
                        C_d[j-jj, i-ii] = A_d[i, j]

            // Phase 2: Write transposed tile from shared memory
            GPU kernel:
                for i = ii to min(ii+NB-1, NX-1):
                    for j = jj to min(jj+NB-1, NY-1):
                        B_d[j, i] = C_d[j-jj, i-ii]
```

### MZTRANS Pseudocode

```
function MZTRANS(A_d, B_d, NS, NX, NY):
    // A_d: NS × NX × NY device array
    // B_d: NS × NY × NX device array

    GPU kernel (3D):
        for i = 0 to NX-1:
            for j = 0 to NY-1:
                for k = 0 to NS-1:
                    B_d[k, j, i] = A_d[k, i, j]
```

### Notes

- `ZTRANS` uses a two-phase approach with shared memory (`C_d`) to avoid uncoalesced global memory writes — the tile is loaded row-wise, then written column-wise from shared memory.
- The tile size `NB` is defined in `param.h` and tuned for GPU shared memory size (typically 16 or 32).
- `MZTRANS` is simpler since the leading dimension `NS` stays contiguous — a direct 3-D kernel suffices.
- Both routines use `!$cuf kernel do` directives for automatic CUDA kernel generation.
- These are the fundamental data-movement primitives used by `cuzfft1d`, `cuzfft2d`, and `cuzfft3d`.

## Source

Original Fortran: `cuztrans.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
