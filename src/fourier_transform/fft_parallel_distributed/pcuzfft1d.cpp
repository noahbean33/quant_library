// Parallel 1-D complex FFT for NVIDIA GPUs (CUDA + MPI) — C++ equivalent of pcuzfft1d.f
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
void pgetnxny(long long N, int& NX, int& NY, int NPU);

class PCUZFFT1D {
public:
    void init(long long N, MPI_Comm comm, int me, int npu) {
        n_ = N; comm_ = comm; me_ = me; npu_ = npu;
        nn_ = static_cast<int>(N / npu);
        pgetnxny(N, nx_, ny_, npu);

        int nx_hat = nx_ / npu;
        cufftPlan1d(&plan_x_, nx_, CUFFT_Z2Z, ny_ / npu);
        cufftPlan1d(&plan_y_, ny_, CUFFT_Z2Z, nx_hat);

        cudaMalloc(&a_d_, sizeof(cufftDoubleComplex) * nn_);
        cudaMalloc(&b_d_, sizeof(cufftDoubleComplex) * nn_);
    }

    void destroy() {
        cufftDestroy(plan_x_);
        cufftDestroy(plan_y_);
        cudaFree(a_d_);
        cudaFree(b_d_);
    }

    void execute(Complex* a, Complex* b, int iopt) {
        // Copy to GPU
        cudaMemcpy(a_d_, a, sizeof(Complex) * nn_, cudaMemcpyHostToDevice);

        // GPU: transpose + NY-point FFTs + twiddle + transpose
        ztrans(reinterpret_cast<Complex*>(a_d_),
               reinterpret_cast<Complex*>(b_d_), nx_ / npu_, ny_);
        cufftExecZ2Z(plan_y_, b_d_, b_d_, CUFFT_FORWARD);
        // Twiddle multiplication on GPU...
        // Transpose back...

        // Copy to host for MPI all-to-all
        std::vector<Complex> sendbuf(nn_), recvbuf(nn_);
        cudaMemcpy(sendbuf.data(), b_d_, sizeof(Complex) * nn_,
                   cudaMemcpyDeviceToHost);

        int chunk = (nx_ / npu_) * (ny_ / npu_);
        MPI_Alltoall(sendbuf.data(), chunk * 2, MPI_DOUBLE,
                     recvbuf.data(), chunk * 2, MPI_DOUBLE, comm_);

        // Copy back to GPU for NX-point FFTs
        cudaMemcpy(a_d_, recvbuf.data(), sizeof(Complex) * nn_,
                   cudaMemcpyHostToDevice);

        // GPU: rearrange + NX-point FFTs + transpose
        cufftExecZ2Z(plan_x_, a_d_, b_d_, CUFFT_FORWARD);

        // Copy result to host
        cudaMemcpy(b, b_d_, sizeof(Complex) * nn_, cudaMemcpyDeviceToHost);
    }

private:
    long long n_;
    int nn_, nx_, ny_, me_, npu_;
    MPI_Comm comm_;
    cufftHandle plan_x_, plan_y_;
    cufftDoubleComplex *a_d_, *b_d_;
};
