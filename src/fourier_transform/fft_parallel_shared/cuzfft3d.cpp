// 3-D Complex FFT for NVIDIA GPUs (CUDA C++ equivalent of cuzfft3d.f)
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
void conjugate_kernel_3d(cufftDoubleComplex* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx].y = -data[idx].y;
}

__global__
void conjugate_scale_kernel_3d(cufftDoubleComplex* data, int n, double scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x =  data[idx].x * scale;
        data[idx].y = -data[idx].y * scale;
    }
}

class CuZFFT3D {
public:
    void init(int NX, int NY, int NZ) {
        nx_ = NX; ny_ = NY; nz_ = NZ;
        n_ = NX * NY * NZ;
        cufftPlan1d(&plan_x_, NX, CUFFT_Z2Z, NY * NZ);
        cufftPlan1d(&plan_y_, NY, CUFFT_Z2Z, NZ * NX);
        cufftPlan1d(&plan_z_, NZ, CUFFT_Z2Z, NX * NY);
        cudaMalloc(&a_d_, sizeof(cufftDoubleComplex) * n_);
        cudaMalloc(&b_d_, sizeof(cufftDoubleComplex) * n_);
    }

    void destroy() {
        cufftDestroy(plan_x_);
        cufftDestroy(plan_y_);
        cufftDestroy(plan_z_);
        cudaFree(a_d_);
        cudaFree(b_d_);
    }

    // iopt: -1 = forward, +1 = inverse
    void execute(Complex* a, int iopt) {
        int threads = 256;
        int blocks = (n_ + threads - 1) / threads;

        cudaMemcpy(a_d_, a, sizeof(Complex) * n_, cudaMemcpyHostToDevice);

        if (iopt == 1) {
            conjugate_kernel_3d<<<blocks, threads>>>(a_d_, n_);
        }

        // FFT along Z: transpose (NX*NY, NZ), FFT, ...
        ztrans(reinterpret_cast<Complex*>(a_d_),
               reinterpret_cast<Complex*>(b_d_), nx_ * ny_, nz_);
        cufftExecZ2Z(plan_z_, b_d_, b_d_, CUFFT_FORWARD);

        // FFT along Y: transpose (NZ*NX, NY), FFT, ...
        ztrans(reinterpret_cast<Complex*>(b_d_),
               reinterpret_cast<Complex*>(a_d_), nz_ * nx_, ny_);
        cufftExecZ2Z(plan_y_, a_d_, a_d_, CUFFT_FORWARD);

        // FFT along X: transpose (NY*NZ, NX), FFT into output
        ztrans(reinterpret_cast<Complex*>(a_d_),
               reinterpret_cast<Complex*>(b_d_), ny_ * nz_, nx_);
        cufftExecZ2Z(plan_x_, b_d_, a_d_, CUFFT_FORWARD);

        if (iopt == 1) {
            double dn = 1.0 / (static_cast<double>(nx_) *
                               static_cast<double>(ny_) *
                               static_cast<double>(nz_));
            conjugate_scale_kernel_3d<<<blocks, threads>>>(a_d_, n_, dn);
        }

        cudaMemcpy(a, a_d_, sizeof(Complex) * n_, cudaMemcpyDeviceToHost);
    }

private:
    int nx_, ny_, nz_, n_;
    cufftHandle plan_x_, plan_y_, plan_z_;
    cufftDoubleComplex *a_d_, *b_d_;
};
