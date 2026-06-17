// Parallel 2-D complex FFT (MPI) — C++ equivalent of pzfft2d.f
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

class PZFFT2D {
public:
    void init(int NX, int NY, MPI_Comm comm, int npu) {
        nx_ = NX; ny_ = NY; npu_ = npu; comm_ = comm;
        nx_hat_ = NX / npu;
        ny_hat_ = NY / npu;
    }

    // iopt: -1/-2 forward, +1/+2 inverse
    void execute(Complex* a, Complex* b, int iopt) {
        int local_size = nx_ * ny_hat_;
        std::vector<Complex> work(local_size);
        std::vector<Complex> buf(local_size);

        bool inverse = (iopt > 0);
        double dn = 1.0 / (static_cast<double>(nx_) * static_cast<double>(ny_));

        // Copy input
        std::vector<Complex> a_vec(a, a + local_size);
        if (inverse) {
            for (auto& v : a_vec) v = std::conj(v);
        }

        // Step 1: NY_hat individual NX-point FFTs along X
        for (int j = 0; j < ny_hat_; ++j) {
            std::vector<Complex> col(nx_);
            for (int i = 0; i < nx_; ++i) col[i] = a_vec[i * ny_hat_ + j];
            fft(col, nx_);
            for (int i = 0; i < nx_; ++i) a_vec[i * ny_hat_ + j] = col[i];
        }

        // Step 2: Pack for all-to-all
        int chunk = nx_hat_ * ny_hat_;
        for (int p = 0; p < npu_; ++p)
            for (int j = 0; j < ny_hat_; ++j)
                for (int i = 0; i < nx_hat_; ++i)
                    work[p * chunk + j * nx_hat_ + i] =
                        a_vec[(p * nx_hat_ + i) * ny_hat_ + j];

        // Step 3: All-to-all
        MPI_Alltoall(work.data(), chunk * 2, MPI_DOUBLE,
                     buf.data(), chunk * 2, MPI_DOUBLE, comm_);

        // Step 4: Unpack — assemble full NY
        for (int p = 0; p < npu_; ++p)
            for (int j = 0; j < ny_hat_; ++j)
                for (int i = 0; i < nx_hat_; ++i)
                    work[i * ny_ + p * ny_hat_ + j] =
                        buf[p * chunk + j * nx_hat_ + i];

        // Step 5: NX_hat individual NY-point FFTs along Y
        for (int i = 0; i < nx_hat_; ++i) {
            std::vector<Complex> col(ny_);
            for (int j = 0; j < ny_; ++j) col[j] = work[i * ny_ + j];
            fft(col, ny_);
            for (int j = 0; j < ny_; ++j) work[i * ny_ + j] = col[j];
        }

        // Output handling depends on iopt
        if (inverse) {
            for (int i = 0; i < local_size; ++i)
                b[i] = std::conj(work[i]) * dn;
        } else {
            std::copy(work.begin(), work.begin() + local_size, b);
        }
    }

private:
    int nx_, ny_, npu_, nx_hat_, ny_hat_;
    MPI_Comm comm_;
};
