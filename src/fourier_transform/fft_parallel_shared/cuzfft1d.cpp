// 1-D Complex FFT for NVIDIA GPUs (CUDA C++ equivalent of cuzfft1d.f)
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.
// This C++ version uses cuFFT and CUDA runtime API.

#include <cmath>
#include <complex>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

using Complex = std::complex<double>;

// Forward declarations for GPU transpose routines (from cuztrans.cpp)
void ztrans(const Complex* A_d, Complex* B_d, int NX, int NY);

// Factor N into NX × NY with NX ≈ √N
static void get_nx_ny(int N, int& NX, int& NY) {
    NX = static_cast<int>(std::sqrt(static_cast<double>(N)));
    while (N % NX != 0) --NX;
    NY = N / NX;
}

// CUDA kernel: conjugate all elements
__global__
void conjugate_kernel(cufftDoubleComplex* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].y = -data[idx].y;
    }
}

// CUDA kernel: conjugate and scale
__global__
void conjugate_scale_kernel(cufftDoubleComplex* data, int n, double scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x =  data[idx].x * scale;
        data[idx].y = -data[idx].y * scale;
    }
}

// CUDA kernel: blocked transpose with twiddle factor multiplication
__global__
void ztransmul_kernel(const cufftDoubleComplex* __restrict__ A_d,
                      cufftDoubleComplex* __restrict__ B_d,
                      int NX, int NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NX && j < NY) {
        double angle = -2.0 * M_PI * static_cast<double>(i) * static_cast<double>(j)
                       / (static_cast<double>(NX) * static_cast<double>(NY));
        double tw_r = cos(angle);
        double tw_i = sin(angle);
        cufftDoubleComplex val = A_d[j * NX + i];  // A_d(i, j) col-major → row-major
        B_d[i * NY + j].x = val.x * tw_r - val.y * tw_i;
        B_d[i * NY + j].y = val.x * tw_i + val.y * tw_r;
    }
}

class CuZFFT1D {
public:
    void init(int N) {
        n_ = N;
        get_nx_ny(N, nx_, ny_);
        cufftPlan1d(&plan_x_, nx_, CUFFT_Z2Z, ny_);
        cufftPlan1d(&plan_y_, ny_, CUFFT_Z2Z, nx_);
        cudaMalloc(&a_d_, sizeof(cufftDoubleComplex) * N);
        cudaMalloc(&b_d_, sizeof(cufftDoubleComplex) * N);
    }

    void destroy() {
        cufftDestroy(plan_x_);
        cufftDestroy(plan_y_);
        cudaFree(a_d_);
        cudaFree(b_d_);
    }

    // iopt: -1 = forward, +1 = inverse
    void execute(Complex* a, int iopt) {
        int threads = 256;
        int blocks = (n_ + threads - 1) / threads;

        cudaMemcpy(a_d_, a, sizeof(Complex) * n_, cudaMemcpyHostToDevice);

        if (iopt == 1) {
            conjugate_kernel<<<blocks, threads>>>(a_d_, n_);
        }

        // Six-step FFT on GPU
        ztrans(reinterpret_cast<Complex*>(a_d_),
               reinterpret_cast<Complex*>(b_d_), nx_, ny_);
        cufftExecZ2Z(plan_y_, b_d_, b_d_, CUFFT_FORWARD);

        dim3 tblock(16, 16);
        dim3 tgrid((ny_ + 15) / 16, (nx_ + 15) / 16);
        ztransmul_kernel<<<tgrid, tblock>>>(b_d_, a_d_, ny_, nx_);

        cufftExecZ2Z(plan_x_, a_d_, a_d_, CUFFT_FORWARD);
        ztrans(reinterpret_cast<Complex*>(a_d_),
               reinterpret_cast<Complex*>(b_d_), nx_, ny_);

        if (iopt == 1) {
            double dn = 1.0 / static_cast<double>(n_);
            conjugate_scale_kernel<<<blocks, threads>>>(b_d_, n_, dn);
        }

        cudaMemcpy(a, b_d_, sizeof(Complex) * n_, cudaMemcpyDeviceToHost);
    }

private:
    int n_, nx_, ny_;
    cufftHandle plan_x_, plan_y_;
    cufftDoubleComplex *a_d_, *b_d_;
};
