// GPU blocked matrix transposition (CUDA C++ equivalent of cuztrans.f)
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.
// This C++ version uses CUDA runtime API.

#include <complex>
#include <cuda_runtime.h>

using Complex = std::complex<double>;

// Reinterpret std::complex<double> as cuDoubleComplex for device code
struct cuComplex2 { double x, y; };

static constexpr int NB = 16;  // tile block size (from param.h)

// CUDA kernel: blocked transpose phase 1 — load tile into shared memory
__global__
void ztrans_kernel(const cuComplex2* __restrict__ A_d,
                   cuComplex2* __restrict__ B_d,
                   int NX, int NY) {
    __shared__ cuComplex2 tile[NB][NB];

    int ii = blockIdx.x * NB;
    int jj = blockIdx.y * NB;
    int i  = ii + threadIdx.x;
    int j  = jj + threadIdx.y;

    // Load tile from A_d(NX, NY) into shared memory
    if (i < NX && j < NY) {
        tile[threadIdx.y][threadIdx.x] = A_d[j * NX + i];
    }
    __syncthreads();

    // Write transposed tile to B_d(NY, NX)
    if (i < NX && j < NY) {
        B_d[i * NY + j] = tile[threadIdx.y][threadIdx.x];
    }
}

// Host wrapper: transpose NX × NY → NY × NX
void ztrans(const Complex* A_d, Complex* B_d, int NX, int NY) {
    dim3 block(NB, NB);
    dim3 grid((NX + NB - 1) / NB, (NY + NB - 1) / NB);
    ztrans_kernel<<<grid, block>>>(
        reinterpret_cast<const cuComplex2*>(A_d),
        reinterpret_cast<cuComplex2*>(B_d),
        NX, NY);
    cudaDeviceSynchronize();
}

// CUDA kernel: multi-row transpose A_d(NS, NX, NY) → B_d(NS, NY, NX)
__global__
void mztrans_kernel(const cuComplex2* __restrict__ A_d,
                    cuComplex2* __restrict__ B_d,
                    int NS, int NX, int NY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NS * NX * NY;
    if (idx >= total) return;

    int k = idx % NS;
    int i = (idx / NS) % NX;
    int j = idx / (NS * NX);

    // A_d[k + NS*(i + NX*j)] → B_d[k + NS*(j + NY*i)]
    B_d[k + NS * (j + NY * i)] = A_d[k + NS * (i + NX * j)];
}

// Host wrapper: multi-row transpose
void mztrans(const Complex* A_d, Complex* B_d, int NS, int NX, int NY) {
    int total = NS * NX * NY;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mztrans_kernel<<<blocks, threads>>>(
        reinterpret_cast<const cuComplex2*>(A_d),
        reinterpret_cast<cuComplex2*>(B_d),
        NS, NX, NY);
    cudaDeviceSynchronize();
}
