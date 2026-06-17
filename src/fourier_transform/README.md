# FFT Algorithm Collection

A comprehensive collection of Fast Fourier Transform (FFT) algorithms implemented in C++ with accompanying Markdown documentation. The repository covers a wide range of FFT techniques, from basic textbook algorithms to high-performance, parallel, and distributed variants.

## Repository Structure

| Folder | Description |
|--------|-------------|
| `fft_basic/` | Fundamental FFT algorithms — Cooley-Tukey, decimation-in-frequency, Stockham, and real FFT |
| `fft_mixed_raddix/` | Mixed-radix FFT supporting radices 2, 3, 4, 5, and 8, including butterfly kernels, factorization utilities, and multi-column FFT routines |
| `fft_split_raddix/` | Extended split-radix FFT using L-shaped butterflies for reduced arithmetic operations |
| `fft_high_performance/` | Cache-optimized FFT routines — six-step FFT, blocked variants, vectorizable kernels, and SSE3-accelerated implementations |
| `fft_multidimensional/` | 2-D and 3-D FFTs (complex-to-complex, real-to-complex, complex-to-real) using row-column decomposition with cache blocking |
| `fft_parallel_shared/` | Shared-memory parallel FFTs using OpenMP and CUDA |
| `fft_parallel_distributed/` | Distributed-memory parallel FFTs using MPI, including autotuning and computation-communication overlap |

## File Convention

Each algorithm is provided as a pair of files:

- **`.cpp`** — Clean C++ implementation using `std::complex<double>` and `std::vector`
- **`.md`** — Documentation with algorithm description, pseudocode, and notes
