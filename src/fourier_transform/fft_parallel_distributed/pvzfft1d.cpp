// Parallel 1-D complex FFT for vector machines (MPI) — C++ equivalent of pvzfft1d.f
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.
// This version uses the same parallel algorithm as pzfft1d.cpp but with
// vector-optimized local FFT kernels.

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

// Vector-optimized FFT kernel (replaces scalar fft() for vector machines)
void vzfft1d(std::vector<Complex>& col, int n);
void pgetnxny(long long N, int& NX, int& NY, int NPU);

class PVZFFT1D {
public:
    void init(long long N, MPI_Comm comm, int me, int npu) {
        n_ = N; comm_ = comm; me_ = me; npu_ = npu;
        nn_ = static_cast<int>(N / npu);
        pgetnxny(N, nx_, ny_, npu);

        w_.resize(nn_);
        int nx_hat = nx_ / npu;
        double pi2 = 8.0 * std::atan(1.0);
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_hat; ++i) {
                int ig = me * nx_hat + i;
                double angle = -pi2 * static_cast<double>(ig) *
                               static_cast<double>(j) / static_cast<double>(N);
                w_[j * nx_hat + i] = Complex(std::cos(angle), std::sin(angle));
            }
    }

    void execute(Complex* a, Complex* b, int iopt) {
        std::vector<Complex> a_vec(a, a + nn_);
        std::vector<Complex> work(nn_), buf(nn_);

        double dn = 1.0 / static_cast<double>(n_);
        int nx_hat = nx_ / npu_;
        int ny_hat = ny_ / npu_;

        if (iopt == 1)
            for (auto& v : a_vec) v = std::conj(v);

        // Local NY-point vector FFTs
        for (int i = 0; i < nx_hat; ++i) {
            std::vector<Complex> col(ny_);
            for (int j = 0; j < ny_; ++j) col[j] = a_vec[j * nx_hat + i];
            vzfft1d(col, ny_);
            for (int j = 0; j < ny_; ++j) a_vec[j * nx_hat + i] = col[j];
        }

        // Twiddle
        for (int i = 0; i < nn_; ++i) a_vec[i] *= w_[i];

        // All-to-all
        int chunk = nx_hat * ny_hat;
        for (int p = 0; p < npu_; ++p)
            for (int i = 0; i < nx_hat; ++i)
                for (int j = 0; j < ny_hat; ++j)
                    work[p * chunk + i * ny_hat + j] =
                        a_vec[(p * ny_hat + j) * nx_hat + i];

        MPI_Alltoall(work.data(), chunk * 2, MPI_DOUBLE,
                     buf.data(), chunk * 2, MPI_DOUBLE, comm_);

        // Unpack + local NX-point vector FFTs
        for (int p = 0; p < npu_; ++p)
            for (int i = 0; i < nx_hat; ++i)
                for (int j = 0; j < ny_hat; ++j)
                    work[(p * nx_hat + i) * ny_hat + j] =
                        buf[p * chunk + i * ny_hat + j];

        for (int j = 0; j < ny_hat; ++j) {
            std::vector<Complex> col(nx_);
            for (int i = 0; i < nx_; ++i) col[i] = work[i * ny_hat + j];
            vzfft1d(col, nx_);
            for (int i = 0; i < nx_; ++i) work[i * ny_hat + j] = col[i];
        }

        if (iopt == 1)
            for (int i = 0; i < nn_; ++i) b[i] = std::conj(work[i]) * dn;
        else
            std::copy(work.begin(), work.end(), b);
    }

private:
    long long n_;
    int nn_, nx_, ny_, me_, npu_;
    MPI_Comm comm_;
    std::vector<Complex> w_;
};
