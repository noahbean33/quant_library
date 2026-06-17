// Parallel 3-D complex FFT (MPI, 1-D decomposition) — C++ equivalent of pzfft3d.f
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

void fft(std::vector<Complex>& col, int n);

static inline int idx3(int i, int j, int k, int d2, int d3) {
    return (i * d2 + j) * d3 + k;
}

class PZFFT3D {
public:
    void init(int NX, int NY, int NZ, MPI_Comm comm, int npu) {
        nx_ = NX; ny_ = NY; nz_ = NZ; npu_ = npu; comm_ = comm;
        nz_hat_ = NZ / npu;
        nx_hat_ = NX / npu;
    }

    void execute(Complex* a, Complex* b, int iopt) {
        int local_in = nx_ * ny_ * nz_hat_;
        bool inverse = (iopt > 0);
        double dn = 1.0 / (static_cast<double>(nx_) * static_cast<double>(ny_) *
                           static_cast<double>(nz_));

        std::vector<Complex> data(a, a + local_in);
        if (inverse) for (auto& v : data) v = std::conj(v);

        // Step 1: FFTs along X (local)
        for (int k = 0; k < nz_hat_; ++k) {
            for (int j = 0; j < ny_; ++j) {
                std::vector<Complex> col(nx_);
                for (int i = 0; i < nx_; ++i)
                    col[i] = data[idx3(i, j, k, ny_, nz_hat_)];
                fft(col, nx_);
                for (int i = 0; i < nx_; ++i)
                    data[idx3(i, j, k, ny_, nz_hat_)] = col[i];
            }
        }

        // Step 2-3: FFTs along Y (local, with transposition for cache)
        for (int k = 0; k < nz_hat_; ++k) {
            for (int i = 0; i < nx_; ++i) {
                std::vector<Complex> col(ny_);
                for (int j = 0; j < ny_; ++j)
                    col[j] = data[idx3(i, j, k, ny_, nz_hat_)];
                fft(col, ny_);
                for (int j = 0; j < ny_; ++j)
                    data[idx3(i, j, k, ny_, nz_hat_)] = col[j];
            }
        }

        // Step 4: Pack for all-to-all (redistribute Z)
        int chunk = nx_hat_ * ny_ * nz_hat_;
        std::vector<Complex> sendbuf(local_in), recvbuf(local_in);
        for (int p = 0; p < npu_; ++p)
            for (int i = 0; i < nx_hat_; ++i)
                for (int j = 0; j < ny_; ++j)
                    for (int k = 0; k < nz_hat_; ++k)
                        sendbuf[p * chunk + idx3(i, j, k, ny_, nz_hat_)] =
                            data[idx3(p * nx_hat_ + i, j, k, ny_, nz_hat_)];

        // Step 5: All-to-all
        MPI_Alltoall(sendbuf.data(), chunk * 2, MPI_DOUBLE,
                     recvbuf.data(), chunk * 2, MPI_DOUBLE, comm_);

        // Step 6: Unpack — assemble full NZ, layout (NZ, NX_hat, NY)
        int local_out = nz_ * nx_hat_ * ny_;
        std::vector<Complex> phase2(local_out);
        for (int p = 0; p < npu_; ++p)
            for (int i = 0; i < nx_hat_; ++i)
                for (int j = 0; j < ny_; ++j)
                    for (int k = 0; k < nz_hat_; ++k)
                        phase2[idx3(p * nz_hat_ + k, i, j, nx_hat_, ny_)] =
                            recvbuf[p * chunk + idx3(i, j, k, ny_, nz_hat_)];

        // Step 7: FFTs along Z
        for (int i = 0; i < nx_hat_; ++i) {
            for (int j = 0; j < ny_; ++j) {
                std::vector<Complex> col(nz_);
                for (int k = 0; k < nz_; ++k)
                    col[k] = phase2[idx3(k, i, j, nx_hat_, ny_)];
                fft(col, nz_);
                for (int k = 0; k < nz_; ++k)
                    phase2[idx3(k, i, j, nx_hat_, ny_)] = col[k];
            }
        }

        // Output
        int out_size = nx_ * ny_ * nz_hat_;
        if (inverse) {
            for (int i = 0; i < out_size; ++i)
                b[i] = std::conj(phase2[i]) * dn;
        } else {
            std::copy(phase2.begin(), phase2.begin() + out_size, b);
        }
    }

private:
    int nx_, ny_, nz_, npu_, nz_hat_, nx_hat_;
    MPI_Comm comm_;
};
