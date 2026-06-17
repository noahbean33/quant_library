// Parallel 2-D complex FFT for NVIDIA GPUs (CUDA + MPI) — C++ equivalent of pcuzfft2d.f
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cufft.h>

using Complex = std::complex<double>;

void ztrans(const Complex* A_d, Complex* B_d, int NX, int NY);

class PCUZFFT2D {
public:
    void init(int NX, int NY, MPI_Comm comm, int npu) {
        nx_ = NX; ny_ = NY; npu_ = npu; comm_ = comm;
        nx_hat_ = NX / npu; ny_hat_ = NY / npu;
        int local_size = NX * ny_hat_;

        cufftPlan1d(&plan_x_, NX, CUFFT_Z2Z, ny_hat_);
        cufftPlan1d(&plan_y_, NY, CUFFT_Z2Z, nx_hat_);

        cudaMalloc(&a_d_, sizeof(cufftDoubleComplex) * local_size);
        cudaMalloc(&b_d_, sizeof(cufftDoubleComplex) * local_size);
    }

    void destroy() {
        cufftDestroy(plan_x_);
        cufftDestroy(plan_y_);
        cudaFree(a_d_);
        cudaFree(b_d_);
    }

    void execute(Complex* a, Complex* b, int iopt) {
        int local_size = nx_ * ny_hat_;

        // Copy to GPU, FFT along X
        cudaMemcpy(a_d_, a, sizeof(Complex) * local_size, cudaMemcpyHostToDevice);
        ztrans(reinterpret_cast<Complex*>(a_d_),
               reinterpret_cast<Complex*>(b_d_), nx_, ny_hat_);
        cufftExecZ2Z(plan_x_, a_d_, a_d_, CUFFT_FORWARD);

        // Copy to host for all-to-all
        std::vector<Complex> sendbuf(local_size), recvbuf(local_size);
        cudaMemcpy(sendbuf.data(), a_d_, sizeof(Complex) * local_size,
                   cudaMemcpyDeviceToHost);

        int chunk = nx_hat_ * ny_hat_;
        MPI_Alltoall(sendbuf.data(), chunk * 2, MPI_DOUBLE,
                     recvbuf.data(), chunk * 2, MPI_DOUBLE, comm_);

        // Copy to GPU, FFT along Y
        cudaMemcpy(a_d_, recvbuf.data(), sizeof(Complex) * local_size,
                   cudaMemcpyHostToDevice);
        cufftExecZ2Z(plan_y_, a_d_, b_d_, CUFFT_FORWARD);

        // Copy result to host
        cudaMemcpy(b, b_d_, sizeof(Complex) * local_size, cudaMemcpyDeviceToHost);
    }

private:
    int nx_, ny_, npu_, nx_hat_, ny_hat_;
    MPI_Comm comm_;
    cufftHandle plan_x_, plan_y_;
    cufftDoubleComplex *a_d_, *b_d_;
};
