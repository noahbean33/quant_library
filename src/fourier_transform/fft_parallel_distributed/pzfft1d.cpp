// Parallel 1-D complex FFT (MPI) — C++ equivalent of pzfft1d.f
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

static const double PI = 4.0 * std::atan(1.0);

// Forward declarations
void fft(std::vector<Complex>& col, int n);
void pgetnxny(long long N, int& NX, int& NY, int NPU);

class PZFFT1D {
public:
    void init(long long N, int icomm, int me, int npu) {
        n_ = N; comm_ = icomm; me_ = me; npu_ = npu;
        nn_ = static_cast<int>(N / npu);
        pgetnxny(N, nx_, ny_, npu);

        // Precompute twiddle factors
        w_.resize(nn_);
        int nx_hat = nx_ / npu;
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_hat; ++i) {
                int i_global = me * nx_hat + i;
                double angle = -2.0 * PI * static_cast<double>(i_global) *
                               static_cast<double>(j) / static_cast<double>(N);
                w_[j * nx_hat + i] = Complex(std::cos(angle), std::sin(angle));
            }
        }
    }

    // iopt: -1 = forward, +1 = inverse
    void execute(Complex* a, Complex* b, int iopt) {
        std::vector<Complex> a_vec(a, a + nn_);
        std::vector<Complex> b_vec(nn_);
        std::vector<Complex> work(nn_);

        double dn = 1.0 / static_cast<double>(n_);
        int nx_hat = nx_ / npu_;
        int ny_hat = ny_ / npu_;

        // Conjugate for inverse
        if (iopt == 1) {
            for (int i = 0; i < nn_; ++i)
                a_vec[i] = std::conj(a_vec[i]);
        }

        // Local NY-point FFTs
        for (int i = 0; i < nx_hat; ++i) {
            std::vector<Complex> col(ny_);
            for (int j = 0; j < ny_; ++j)
                col[j] = a_vec[j * nx_hat + i];
            fft(col, ny_);
            for (int j = 0; j < ny_; ++j)
                a_vec[j * nx_hat + i] = col[j];
        }

        // Twiddle factor multiplication
        for (int i = 0; i < nn_; ++i)
            a_vec[i] *= w_[i];

        // Pack for all-to-all
        int chunk = nx_hat * ny_hat;
        for (int p = 0; p < npu_; ++p)
            for (int i = 0; i < nx_hat; ++i)
                for (int j = 0; j < ny_hat; ++j)
                    work[p * chunk + i * ny_hat + j] =
                        a_vec[(p * ny_hat + j) * nx_hat + i];

        // All-to-all
        MPI_Alltoall(work.data(), chunk * 2, MPI_DOUBLE,
                     b_vec.data(), chunk * 2, MPI_DOUBLE,
                     MPI_Comm_f2c(comm_));

        // Unpack and local NX-point FFTs
        for (int p = 0; p < npu_; ++p)
            for (int i = 0; i < nx_hat; ++i)
                for (int j = 0; j < ny_hat; ++j)
                    work[(p * nx_hat + i) * ny_hat + j] =
                        b_vec[p * chunk + i * ny_hat + j];

        for (int j = 0; j < ny_hat; ++j) {
            std::vector<Complex> col(nx_);
            for (int i = 0; i < nx_; ++i)
                col[i] = work[i * ny_hat + j];
            fft(col, nx_);
            for (int i = 0; i < nx_; ++i)
                work[i * ny_hat + j] = col[i];
        }

        // Conjugate and scale for inverse
        if (iopt == 1) {
            for (int i = 0; i < nn_; ++i)
                b[i] = std::conj(work[i]) * dn;
        } else {
            std::copy(work.begin(), work.end(), b);
        }
    }

private:
    long long n_;
    int nn_, nx_, ny_, comm_, me_, npu_;
    std::vector<Complex> w_;
};
