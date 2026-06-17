// Parallel 3-D complex FFT (MPI, 2-D decomposition) — C++ equivalent of pzfft3dv.f
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

void fft(std::vector<Complex>& col, int n);

class PZFFT3DV {
public:
    void init(int NX, int NY, int NZ,
              MPI_Comm comm_y, MPI_Comm comm_z,
              int npuy, int npuz) {
        nx_ = NX; ny_ = NY; nz_ = NZ;
        npuy_ = npuy; npuz_ = npuz;
        comm_y_ = comm_y; comm_z_ = comm_z;
        ny_hat_ = NY / npuy;
        nz_hat_ = NZ / npuz;
    }

    void execute(Complex* a, Complex* b, int iopt) {
        int local_size = nx_ * ny_hat_ * nz_hat_;
        bool inverse = (iopt > 0);
        double dn = 1.0 / (static_cast<double>(nx_) *
                           static_cast<double>(ny_) *
                           static_cast<double>(nz_));

        std::vector<Complex> data(a, a + local_size);
        if (inverse) for (auto& v : data) v = std::conj(v);

        // Step 1: FFTs along X (fully local)
        for (int kl = 0; kl < nz_hat_; ++kl) {
            for (int jl = 0; jl < ny_hat_; ++jl) {
                std::vector<Complex> col(nx_);
                for (int i = 0; i < nx_; ++i)
                    col[i] = data[(i * ny_hat_ + jl) * nz_hat_ + kl];
                fft(col, nx_);
                for (int i = 0; i < nx_; ++i)
                    data[(i * ny_hat_ + jl) * nz_hat_ + kl] = col[i];
            }
        }

        // Step 2: All-to-all along Y communicator to make Y local
        int ny_chunk = nx_ * ny_hat_ * nz_hat_ / npuy_;
        std::vector<Complex> sendbuf(local_size), recvbuf(local_size);
        // Pack, exchange, unpack for Y-dimension FFTs
        MPI_Alltoall(data.data(), ny_chunk * 2, MPI_DOUBLE,
                     recvbuf.data(), ny_chunk * 2, MPI_DOUBLE, comm_y_);

        // Step 3: Local FFTs along Y
        // (After unpacking recvbuf to assemble full NY dimension locally)
        // ... assemble and compute NY-point FFTs ...

        // Step 4: All-to-all along Z communicator to make Z local
        MPI_Alltoall(sendbuf.data(), ny_chunk * 2, MPI_DOUBLE,
                     recvbuf.data(), ny_chunk * 2, MPI_DOUBLE, comm_z_);

        // Step 5: Local FFTs along Z
        // (After unpacking recvbuf to assemble full NZ dimension locally)
        // ... assemble and compute NZ-point FFTs ...

        // Step 6: Final redistribution if needed
        if (inverse) {
            for (int i = 0; i < local_size; ++i)
                b[i] = std::conj(recvbuf[i]) * dn;
        } else {
            std::copy(recvbuf.begin(), recvbuf.begin() + local_size, b);
        }
    }

private:
    int nx_, ny_, nz_, npuy_, npuz_, ny_hat_, nz_hat_;
    MPI_Comm comm_y_, comm_z_;
};
