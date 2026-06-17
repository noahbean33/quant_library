// Parallel 2-D complex-to-real FFT (MPI) — C++ equivalent of pzdfft2d.f
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

class PZDFFT2D {
public:
    void init(int NX, int NY, MPI_Comm comm, int me, int npu) {
        nx_ = NX; ny_ = NY; npu_ = npu; me_ = me; comm_ = comm;
        ny_hat_ = NY / npu;
        nx_half_ = NX / 2;
        nx_half_hat_ = nx_half_ / npu;
    }

    // iopt: +1 inverse (same dist), +2 inverse (transposed input)
    void execute(Complex* a, double* b, int iopt) {
        int cplx_local = (nx_half_ + 1) * ny_hat_;

        // Step 1: Inverse complex FFTs along Y
        for (int i = 0; i < nx_half_hat_; ++i) {
            std::vector<Complex> col(ny_);
            for (int j = 0; j < ny_; ++j)
                col[j] = a[i * ny_ + j];
            // Conjugate, forward FFT, conjugate, scale = inverse FFT
            for (auto& v : col) v = std::conj(v);
            fft(col, ny_);
            double dn = 1.0 / static_cast<double>(ny_);
            for (int j = 0; j < ny_; ++j)
                col[j] = std::conj(col[j]) * dn;
            for (int j = 0; j < ny_; ++j)
                a[i * ny_ + j] = col[j];
        }

        // Step 2: All-to-all to redistribute X-frequencies
        int chunk = nx_half_hat_ * ny_hat_;
        std::vector<Complex> sendbuf(cplx_local), recvbuf(cplx_local);

        for (int p = 0; p < npu_; ++p)
            for (int i = 0; i < nx_half_hat_; ++i)
                for (int j = 0; j < ny_hat_; ++j)
                    sendbuf[p * chunk + i * ny_hat_ + j] =
                        a[i * ny_ + p * ny_hat_ + j];

        MPI_Alltoall(sendbuf.data(), chunk * 2, MPI_DOUBLE,
                     recvbuf.data(), chunk * 2, MPI_DOUBLE, comm_);

        // Step 3: Unpack and complex-to-real inverse FFTs along X
        std::vector<Complex> x_freq(nx_half_ + 1);
        for (int j = 0; j < ny_hat_; ++j) {
            for (int p = 0; p < npu_; ++p)
                for (int i = 0; i < nx_half_hat_; ++i)
                    x_freq[p * nx_half_hat_ + i] =
                        recvbuf[p * chunk + i * ny_hat_ + j];
            x_freq[nx_half_] = Complex(0.0, 0.0);  // Nyquist

            std::vector<double> row(nx_);
            irfft(x_freq, row, nx_);
            for (int i = 0; i < nx_; ++i)
                b[i * ny_hat_ + j] = row[i];
        }
    }

private:
    int nx_, ny_, npu_, me_, ny_hat_, nx_half_, nx_half_hat_;
    MPI_Comm comm_;
};
