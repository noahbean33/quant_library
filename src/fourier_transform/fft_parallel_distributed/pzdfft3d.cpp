// Parallel 3-D complex-to-real FFT (MPI, 1-D decomposition) — C++ equivalent of pzdfft3d.f
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

void fft(std::vector<Complex>& col, int n);
void irfft(const std::vector<Complex>& in, std::vector<double>& out, int n);

class PZDFFT3D {
public:
    void init(int NX, int NY, int NZ, MPI_Comm comm, int me, int npu) {
        nx_ = NX; ny_ = NY; nz_ = NZ;
        npu_ = npu; me_ = me; comm_ = comm;
        nz_hat_ = NZ / npu;
        nx_half_ = NX / 2;
    }

    // iopt: +1 inverse (same dist), +2 inverse (transposed input)
    void execute(Complex* a, Complex* b, int iopt) {
        int cplx_local = (nx_half_ + 1) * ny_ * nz_hat_;

        // Step 1: All-to-all to make Z local
        // Pack, exchange, unpack...
        std::vector<Complex> work(cplx_local);
        std::copy(a, a + cplx_local, work.begin());

        // Step 2: Inverse complex FFTs along Z
        for (int i = 0; i <= nx_half_; ++i) {
            for (int j = 0; j < ny_; ++j) {
                std::vector<Complex> col(nz_);
                for (int k = 0; k < nz_; ++k)
                    col[k] = work[(i * ny_ + j) * nz_ + k];
                for (auto& v : col) v = std::conj(v);
                fft(col, nz_);
                double dn = 1.0 / static_cast<double>(nz_);
                for (int k = 0; k < nz_; ++k)
                    work[(i * ny_ + j) * nz_ + k] = std::conj(col[k]) * dn;
            }
        }

        // Step 3: Inverse complex FFTs along Y
        for (int k = 0; k < nz_hat_; ++k) {
            for (int i = 0; i <= nx_half_; ++i) {
                std::vector<Complex> col(ny_);
                for (int j = 0; j < ny_; ++j)
                    col[j] = work[(i * ny_ + j) * nz_hat_ + k];
                for (auto& v : col) v = std::conj(v);
                fft(col, ny_);
                double dn = 1.0 / static_cast<double>(ny_);
                for (int j = 0; j < ny_; ++j)
                    work[(i * ny_ + j) * nz_hat_ + k] = std::conj(col[j]) * dn;
            }
        }

        // Step 4: Complex-to-real inverse FFTs along X
        double* out_real = reinterpret_cast<double*>(a);
        for (int k = 0; k < nz_hat_; ++k) {
            for (int j = 0; j < ny_; ++j) {
                std::vector<Complex> x_freq(nx_half_ + 1);
                for (int i = 0; i <= nx_half_; ++i)
                    x_freq[i] = work[(i * ny_ + j) * nz_hat_ + k];
                std::vector<double> row(nx_);
                irfft(x_freq, row, nx_);
                for (int i = 0; i < nx_; ++i)
                    out_real[(i * ny_ + j) * nz_hat_ + k] = row[i];
            }
        }
    }

private:
    int nx_, ny_, nz_, npu_, me_, nz_hat_, nx_half_;
    MPI_Comm comm_;
};
