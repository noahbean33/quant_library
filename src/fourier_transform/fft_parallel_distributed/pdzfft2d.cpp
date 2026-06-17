// Parallel 2-D real-to-complex FFT (MPI) — C++ equivalent of pdzfft2d.f
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

// Forward declarations
void fft(std::vector<Complex>& col, int n);
void rfft(const std::vector<double>& in, std::vector<Complex>& out, int n);

class PDZFFT2D {
public:
    void init(int NX, int NY, MPI_Comm comm, int me, int npu) {
        nx_ = NX; ny_ = NY; npu_ = npu; me_ = me; comm_ = comm;
        ny_hat_ = NY / npu;
        nx_half_ = NX / 2;
        nx_half_hat_ = nx_half_ / npu;
    }

    // iopt: -1 forward (same dist), -2 forward (transposed output)
    void execute(const double* a, Complex* b, int iopt) {
        int real_local = nx_ * ny_hat_;
        int cplx_local = (nx_half_ + 1) * ny_hat_;

        // Step 1: Local real-to-complex FFTs along X
        std::vector<Complex> work(cplx_local);
        for (int j = 0; j < ny_hat_; ++j) {
            std::vector<double> row(nx_);
            for (int i = 0; i < nx_; ++i) row[i] = a[i * ny_hat_ + j];

            std::vector<Complex> crow(nx_half_ + 1);
            rfft(row, crow, nx_);

            for (int i = 0; i <= nx_half_; ++i)
                work[i * ny_hat_ + j] = crow[i];
        }

        // Steps 2-3: Pack and all-to-all to redistribute Y
        int chunk = nx_half_hat_ * ny_hat_;
        std::vector<Complex> sendbuf(cplx_local), recvbuf(cplx_local);

        for (int p = 0; p < npu_; ++p)
            for (int j = 0; j < ny_hat_; ++j)
                for (int i = 0; i < nx_half_hat_; ++i)
                    sendbuf[p * chunk + j * nx_half_hat_ + i] =
                        work[(p * nx_half_hat_ + i) * ny_hat_ + j];

        MPI_Alltoall(sendbuf.data(), chunk * 2, MPI_DOUBLE,
                     recvbuf.data(), chunk * 2, MPI_DOUBLE, comm_);

        // Step 4: Unpack and local NY-point FFTs
        int out_local = nx_half_hat_ * ny_;
        std::vector<Complex> phase2(out_local);
        for (int p = 0; p < npu_; ++p)
            for (int j = 0; j < ny_hat_; ++j)
                for (int i = 0; i < nx_half_hat_; ++i)
                    phase2[i * ny_ + p * ny_hat_ + j] =
                        recvbuf[p * chunk + j * nx_half_hat_ + i];

        for (int i = 0; i < nx_half_hat_; ++i) {
            std::vector<Complex> col(ny_);
            for (int j = 0; j < ny_; ++j) col[j] = phase2[i * ny_ + j];
            fft(col, ny_);
            for (int j = 0; j < ny_; ++j) phase2[i * ny_ + j] = col[j];
        }

        // Step 5: Output
        std::copy(phase2.begin(), phase2.begin() + out_local, b);
    }

private:
    int nx_, ny_, npu_, me_, ny_hat_, nx_half_, nx_half_hat_;
    MPI_Comm comm_;
};
