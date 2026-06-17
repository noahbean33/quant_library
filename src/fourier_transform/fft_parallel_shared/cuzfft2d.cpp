// 2-D Complex FFT for NVIDIA GPUs (CUDA C++ equivalent of cuzfft2d.f)
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.

#include <cmath>
#include <complex>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

using Complex = std::complex<double>;

// Forward declarations (from cuztrans.cpp)
void ztrans(const Complex* A_d, Complex* B_d, int NX, int NY);

// CUDA kernels
__global__
void conjugate_kernel_2d(cufftDoubleComplex* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx].y = -data[idx].y;
}

__global__
void conjugate_scale_kernel_2d(cufftDoubleComplex* data, int n, double scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x =  data[idx].x * scale;
        data[idx].y = -data[idx].y * scale;
    }
}

class CuZFFT2D {
public:
    void init(int NX, int NY) {
        nx_ = NX; ny_ = NY;
        n_ = NX * NY;
        cufftPlan1d(&plan_x_, NX, CUFFT_Z2Z, NY);
        cufftPlan1d(&plan_y_, NY, CUFFT_Z2Z, NX);
        cudaMalloc(&a_d_, sizeof(cufftDoubleComplex) * n_);
        cudaMalloc(&b_d_, sizeof(cufftDoubleComplex) * n_);
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
            conjugate_kernel_2d<<<blocks, threads>>>(a_d_, n_);
        }

        // Transpose NX×NY → NY×NX, then FFT along Y
        ztrans(reinterpret_cast<Complex*>(a_d_),
               reinterpret_cast<Complex*>(b_d_), nx_, ny_);
        cufftExecZ2Z(plan_y_, b_d_, b_d_, CUFFT_FORWARD);

        // Transpose NY×NX → NX×NY, then FFT along X
        ztrans(reinterpret_cast<Complex*>(b_d_),
               reinterpret_cast<Complex*>(a_d_), ny_, nx_);
        cufftExecZ2Z(plan_x_, a_d_, a_d_, CUFFT_FORWARD);

        if (iopt == 1) {
            double dn = 1.0 / (static_cast<double>(nx_) * static_cast<double>(ny_));
            conjugate_scale_kernel_2d<<<blocks, threads>>>(a_d_, n_, dn);
        }

        cudaMemcpy(a, a_d_, sizeof(Complex) * n_, cudaMemcpyDeviceToHost);
    }

private:
    int nx_, ny_, n_;
    cufftHandle plan_x_, plan_y_;
    cufftDoubleComplex *a_d_, *b_d_;
};
