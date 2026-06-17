// Parallel 3-D real-to-complex FFT (MPI, 2-D decomposition) — C++ equivalent of pdzfft3dv.f
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

void fft(std::vector<Complex>& col, int n);
void rfft(const std::vector<double>& in, std::vector<Complex>& out, int n);

class PDZFFT3DV {
public:
    void init(int NX, int NY, int NZ,
              MPI_Comm comm_y, MPI_Comm comm_z,
              int mey, int npuy, int npuz) {
        nx_ = NX; ny_ = NY; nz_ = NZ;
        npuy_ = npuy; npuz_ = npuz; mey_ = mey;
        comm_y_ = comm_y; comm_z_ = comm_z;
        ny_hat_ = NY / npuy;
        nz_hat_ = NZ / npuz;
        nx_half_ = NX / 2;
    }

    void execute(void* a, Complex* b, int iopt) {
        double* a_real = static_cast<double*>(a);

        // Step 1: Local real-to-complex FFTs along X
        for (int k = 0; k < nz_hat_; ++k) {
            for (int j = 0; j < ny_hat_; ++j) {
                std::vector<double> row(nx_);
                for (int i = 0; i < nx_; ++i)
                    row[i] = a_real[(i * ny_hat_ + j) * nz_hat_ + k];
                std::vector<Complex> crow(nx_half_ + 1);
                rfft(row, crow, nx_);
                for (int i = 0; i <= nx_half_; ++i)
                    b[(i * ny_hat_ + j) * nz_hat_ + k] = crow[i];
            }
        }

        // Step 2: All-to-all along Y communicator, then local Y FFTs
        int chunk_y = (nx_half_ + 1) * ny_hat_ * nz_hat_ / npuy_;
        std::vector<Complex> recvbuf(chunk_y * npuy_);
        MPI_Alltoall(b, chunk_y * 2, MPI_DOUBLE,
                     recvbuf.data(), chunk_y * 2, MPI_DOUBLE, comm_y_);
        // Unpack, assemble full NY, compute NY-point FFTs...

        // Step 3: All-to-all along Z communicator, then local Z FFTs
        int chunk_z = chunk_y;  // proportional
        std::vector<Complex> recvbuf2(chunk_z * npuz_);
        MPI_Alltoall(recvbuf.data(), chunk_z * 2, MPI_DOUBLE,
                     recvbuf2.data(), chunk_z * 2, MPI_DOUBLE, comm_z_);
        // Unpack, assemble full NZ, compute NZ-point FFTs...

        // Final output placed in a (in-place) or b depending on iopt
    }

private:
    int nx_, ny_, nz_, npuy_, npuz_, mey_;
    int ny_hat_, nz_hat_, nx_half_;
    MPI_Comm comm_y_, comm_z_;
};
