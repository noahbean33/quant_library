// Parallel 2-D complex FFT for vector machines (MPI) — C++ equivalent of pvzfft2d.f
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.
// Same parallel algorithm as pzfft2d.cpp with vector-optimized local FFT kernels.

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

void vzfft1d(std::vector<Complex>& col, int n);

class PVZFFT2D {
public:
    void init(int NX, int NY, MPI_Comm comm, int npu) {
        nx_ = NX; ny_ = NY; npu_ = npu; comm_ = comm;
        nx_hat_ = NX / npu; ny_hat_ = NY / npu;
    }

    void execute(Complex* a, Complex* b, int iopt) {
        int local_size = nx_ * ny_hat_;
        bool inverse = (iopt > 0);
        double dn = 1.0 / (static_cast<double>(nx_) * static_cast<double>(ny_));

        std::vector<Complex> data(a, a + local_size);
        if (inverse) for (auto& v : data) v = std::conj(v);

        // Local NX-point vector FFTs along X
        for (int j = 0; j < ny_hat_; ++j) {
            std::vector<Complex> col(nx_);
            for (int i = 0; i < nx_; ++i) col[i] = data[i * ny_hat_ + j];
            vzfft1d(col, nx_);
            for (int i = 0; i < nx_; ++i) data[i * ny_hat_ + j] = col[i];
        }

        // All-to-all to redistribute Y
        int chunk = nx_hat_ * ny_hat_;
        std::vector<Complex> sendbuf(local_size), recvbuf(local_size);
        for (int p = 0; p < npu_; ++p)
            for (int j = 0; j < ny_hat_; ++j)
                for (int i = 0; i < nx_hat_; ++i)
                    sendbuf[p * chunk + j * nx_hat_ + i] =
                        data[(p * nx_hat_ + i) * ny_hat_ + j];

        MPI_Alltoall(sendbuf.data(), chunk * 2, MPI_DOUBLE,
                     recvbuf.data(), chunk * 2, MPI_DOUBLE, comm_);

        // Unpack + local NY-point vector FFTs along Y
        std::vector<Complex> work(nx_hat_ * ny_);
        for (int p = 0; p < npu_; ++p)
            for (int j = 0; j < ny_hat_; ++j)
                for (int i = 0; i < nx_hat_; ++i)
                    work[i * ny_ + p * ny_hat_ + j] =
                        recvbuf[p * chunk + j * nx_hat_ + i];

        for (int i = 0; i < nx_hat_; ++i) {
            std::vector<Complex> col(ny_);
            for (int j = 0; j < ny_; ++j) col[j] = work[i * ny_ + j];
            vzfft1d(col, ny_);
            for (int j = 0; j < ny_; ++j) work[i * ny_ + j] = col[j];
        }

        if (inverse)
            for (int i = 0; i < local_size; ++i) b[i] = std::conj(work[i]) * dn;
        else
            std::copy(work.begin(), work.begin() + local_size, b);
    }

private:
    int nx_, ny_, npu_, nx_hat_, ny_hat_;
    MPI_Comm comm_;
};
