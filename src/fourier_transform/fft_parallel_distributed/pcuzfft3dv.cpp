// Parallel 3-D complex FFT for NVIDIA GPUs (CUDA + MPI, 2-D decomposition)
// C++ equivalent of pcuzfft3dv.f
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

class PCUZFFT3DV {
public:
    void init(int NX, int NY, int NZ,
              MPI_Comm comm_y, MPI_Comm comm_z,
              int npuy, int npuz) {
        nx_ = NX; ny_ = NY; nz_ = NZ;
        npuy_ = npuy; npuz_ = npuz;
        comm_y_ = comm_y; comm_z_ = comm_z;
        ny_hat_ = NY / npuy; nz_hat_ = NZ / npuz;
        int local_size = NX * ny_hat_ * nz_hat_;

        cufftPlan1d(&plan_x_, NX, CUFFT_Z2Z, ny_hat_ * nz_hat_);
        cufftPlan1d(&plan_y_, NY, CUFFT_Z2Z, NX * nz_hat_ / npuy);
        cufftPlan1d(&plan_z_, NZ, CUFFT_Z2Z, NX * ny_hat_ / npuz);

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
        int local_size = nx_ * ny_hat_ * nz_hat_;

        // Copy to GPU
        cudaMemcpy(a_d_, a, sizeof(Complex) * local_size, cudaMemcpyHostToDevice);

        // GPU: FFTs along X (fully local)
        cufftExecZ2Z(plan_x_, a_d_, a_d_, CUFFT_FORWARD);

        // Copy to host for Y all-to-all
        std::vector<Complex> sendbuf(local_size), recvbuf(local_size);
        cudaMemcpy(sendbuf.data(), a_d_, sizeof(Complex) * local_size,
                   cudaMemcpyDeviceToHost);

        int chunk_y = local_size / npuy_;
        MPI_Alltoall(sendbuf.data(), chunk_y * 2, MPI_DOUBLE,
                     recvbuf.data(), chunk_y * 2, MPI_DOUBLE, comm_y_);

        // Copy to GPU for Y FFTs
        cudaMemcpy(a_d_, recvbuf.data(), sizeof(Complex) * local_size,
                   cudaMemcpyHostToDevice);
        cufftExecZ2Z(plan_y_, a_d_, a_d_, CUFFT_FORWARD);

        // Copy to host for Z all-to-all
        cudaMemcpy(sendbuf.data(), a_d_, sizeof(Complex) * local_size,
                   cudaMemcpyDeviceToHost);

        int chunk_z = local_size / npuz_;
        MPI_Alltoall(sendbuf.data(), chunk_z * 2, MPI_DOUBLE,
                     recvbuf.data(), chunk_z * 2, MPI_DOUBLE, comm_z_);

        // Copy to GPU for Z FFTs
        cudaMemcpy(a_d_, recvbuf.data(), sizeof(Complex) * local_size,
                   cudaMemcpyHostToDevice);
        cufftExecZ2Z(plan_z_, a_d_, b_d_, CUFFT_FORWARD);

        // Copy result to host
        cudaMemcpy(b, b_d_, sizeof(Complex) * local_size, cudaMemcpyDeviceToHost);
    }

private:
    int nx_, ny_, nz_, npuy_, npuz_, ny_hat_, nz_hat_;
    MPI_Comm comm_y_, comm_z_;
    cufftHandle plan_x_, plan_y_, plan_z_;
    cufftDoubleComplex *a_d_, *b_d_;
};
