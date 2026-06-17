// Parallel 3-D complex FFT for NVIDIA GPUs (CUDA + MPI, 1-D decomposition)
// C++ equivalent of pcuzfft3d.f
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

class PCUZFFT3D {
public:
    void init(int NX, int NY, int NZ, MPI_Comm comm, int npu) {
        nx_ = NX; ny_ = NY; nz_ = NZ; npu_ = npu; comm_ = comm;
        nz_hat_ = NZ / npu; nx_hat_ = NX / npu;
        int local_size = NX * NY * nz_hat_;

        cufftPlan1d(&plan_x_, NX, CUFFT_Z2Z, NY * nz_hat_);
        cufftPlan1d(&plan_y_, NY, CUFFT_Z2Z, NX * nz_hat_);
        cufftPlan1d(&plan_z_, NZ, CUFFT_Z2Z, nx_hat_ * NY);

        cudaMalloc(&a_d_, sizeof(cufftDoubleComplex) * local_size);
        cudaMalloc(&b_d_, sizeof(cufftDoubleComplex) * local_size);
    }

    void destroy() {
        cufftDestroy(plan_x_);
        cufftDestroy(plan_y_);
        cufftDestroy(plan_z_);
        cudaFree(a_d_);
        cudaFree(b_d_);
    }

    void execute(Complex* a, Complex* b, int iopt) {
        int local_size = nx_ * ny_ * nz_hat_;

        // Copy to GPU
        cudaMemcpy(a_d_, a, sizeof(Complex) * local_size, cudaMemcpyHostToDevice);

        // GPU: FFTs along X
        ztrans(reinterpret_cast<Complex*>(a_d_),
               reinterpret_cast<Complex*>(b_d_), ny_ * nz_hat_, nx_);
        cufftExecZ2Z(plan_x_, b_d_, b_d_, CUFFT_FORWARD);

        // GPU: FFTs along Y
        ztrans(reinterpret_cast<Complex*>(b_d_),
               reinterpret_cast<Complex*>(a_d_), nx_ * nz_hat_, ny_);
        cufftExecZ2Z(plan_y_, a_d_, a_d_, CUFFT_FORWARD);

        // Copy to host for all-to-all
        std::vector<Complex> sendbuf(local_size), recvbuf(local_size);
        cudaMemcpy(sendbuf.data(), a_d_, sizeof(Complex) * local_size,
                   cudaMemcpyDeviceToHost);

        int chunk = nx_hat_ * ny_ * nz_hat_;
        MPI_Alltoall(sendbuf.data(), chunk * 2, MPI_DOUBLE,
                     recvbuf.data(), chunk * 2, MPI_DOUBLE, comm_);

        // Copy to GPU for Z FFTs
        int z_local = nz_ * nx_hat_ * ny_;
        cudaMemcpy(a_d_, recvbuf.data(), sizeof(Complex) * z_local,
                   cudaMemcpyHostToDevice);

        // GPU: FFTs along Z
        ztrans(reinterpret_cast<Complex*>(a_d_),
               reinterpret_cast<Complex*>(b_d_), nx_hat_ * ny_, nz_);
        cufftExecZ2Z(plan_z_, b_d_, b_d_, CUFFT_FORWARD);

        // Copy result to host
        cudaMemcpy(b, b_d_, sizeof(Complex) * z_local, cudaMemcpyDeviceToHost);
    }

private:
    int nx_, ny_, nz_, npu_, nz_hat_, nx_hat_;
    MPI_Comm comm_;
    cufftHandle plan_x_, plan_y_, plan_z_;
    cufftDoubleComplex *a_d_, *b_d_;
};
