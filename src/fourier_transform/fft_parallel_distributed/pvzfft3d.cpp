// Parallel 3-D complex FFT for vector machines (MPI) — C++ equivalent of pvzfft3d.f
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.
// Same parallel algorithm as pzfft3d.cpp with vector-optimized local FFT kernels.

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

void vzfft1d(std::vector<Complex>& col, int n);

static inline int idx3(int i, int j, int k, int d2, int d3) {
    return (i * d2 + j) * d3 + k;
}

class PVZFFT3D {
public:
    void init(int NX, int NY, int NZ, MPI_Comm comm, int npu) {
        nx_ = NX; ny_ = NY; nz_ = NZ; npu_ = npu; comm_ = comm;
        nz_hat_ = NZ / npu; nx_hat_ = NX / npu;
    }

    void execute(Complex* a, Complex* b, int iopt) {
        int local_in = nx_ * ny_ * nz_hat_;
        bool inverse = (iopt > 0);
        double dn = 1.0 / (static_cast<double>(nx_) * static_cast<double>(ny_) *
                           static_cast<double>(nz_));

        std::vector<Complex> data(a, a + local_in);
        if (inverse) for (auto& v : data) v = std::conj(v);

        // FFTs along X (local, vector-optimized)
        for (int k = 0; k < nz_hat_; ++k)
            for (int j = 0; j < ny_; ++j) {
                std::vector<Complex> col(nx_);
                for (int i = 0; i < nx_; ++i)
                    col[i] = data[idx3(i, j, k, ny_, nz_hat_)];
                vzfft1d(col, nx_);
                for (int i = 0; i < nx_; ++i)
                    data[idx3(i, j, k, ny_, nz_hat_)] = col[i];
            }

        // FFTs along Y (local, vector-optimized)
        for (int k = 0; k < nz_hat_; ++k)
            for (int i = 0; i < nx_; ++i) {
                std::vector<Complex> col(ny_);
                for (int j = 0; j < ny_; ++j)
                    col[j] = data[idx3(i, j, k, ny_, nz_hat_)];
                vzfft1d(col, ny_);
                for (int j = 0; j < ny_; ++j)
                    data[idx3(i, j, k, ny_, nz_hat_)] = col[j];
            }

        // All-to-all to redistribute Z
        int chunk = nx_hat_ * ny_ * nz_hat_;
        std::vector<Complex> sendbuf(local_in), recvbuf(local_in);
        for (int p = 0; p < npu_; ++p)
            for (int i = 0; i < nx_hat_; ++i)
                for (int j = 0; j < ny_; ++j)
                    for (int k = 0; k < nz_hat_; ++k)
                        sendbuf[p * chunk + idx3(i, j, k, ny_, nz_hat_)] =
                            data[idx3(p * nx_hat_ + i, j, k, ny_, nz_hat_)];

        MPI_Alltoall(sendbuf.data(), chunk * 2, MPI_DOUBLE,
                     recvbuf.data(), chunk * 2, MPI_DOUBLE, comm_);

        // Unpack + FFTs along Z (vector-optimized)
        int local_out = nz_ * nx_hat_ * ny_;
        std::vector<Complex> phase2(local_out);
        for (int p = 0; p < npu_; ++p)
            for (int i = 0; i < nx_hat_; ++i)
                for (int j = 0; j < ny_; ++j)
                    for (int k = 0; k < nz_hat_; ++k)
                        phase2[idx3(p * nz_hat_ + k, i, j, nx_hat_, ny_)] =
                            recvbuf[p * chunk + idx3(i, j, k, ny_, nz_hat_)];

        for (int i = 0; i < nx_hat_; ++i)
            for (int j = 0; j < ny_; ++j) {
                std::vector<Complex> col(nz_);
                for (int k = 0; k < nz_; ++k)
                    col[k] = phase2[idx3(k, i, j, nx_hat_, ny_)];
                vzfft1d(col, nz_);
                for (int k = 0; k < nz_; ++k)
                    phase2[idx3(k, i, j, nx_hat_, ny_)] = col[k];
            }

        int out_size = local_out;
        if (inverse)
            for (int i = 0; i < out_size; ++i) b[i] = std::conj(phase2[i]) * dn;
        else
            std::copy(phase2.begin(), phase2.begin() + out_size, b);
    }

private:
    int nx_, ny_, nz_, npu_, nz_hat_, nx_hat_;
    MPI_Comm comm_;
};
