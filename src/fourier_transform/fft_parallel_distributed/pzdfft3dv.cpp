// Parallel 3-D complex-to-real FFT (MPI, 2-D decomposition) — C++ equivalent of pzdfft3dv.f
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

class PZDFFT3DV {
public:
    void init(int NX, int NY, int NZ,
              MPI_Comm comm_y, MPI_Comm comm_z,
              int mey, int npuy, int npuz) {
        nx_ = NX; ny_ = NY; nz_ = NZ;
        npuy_ = npuy; npuz_ = npuz; mey_ = mey;
        comm_y_ = comm_y; comm_z_ = comm_z;
        ny_hat_ = NY / npuy;
        nz_hat_ = NZ / npuz;
        nx_half_ = NX / 2;
    }

    void execute(Complex* a, Complex* b, int iopt) {
        int cplx_local = (nx_half_ + 1) * ny_hat_ * nz_hat_;

        // Step 1: All-to-all along Z to make Z local
        int chunk_z = cplx_local / npuz_;
        std::vector<Complex> recvbuf_z(cplx_local);
        MPI_Alltoall(a, chunk_z * 2, MPI_DOUBLE,
                     recvbuf_z.data(), chunk_z * 2, MPI_DOUBLE, comm_z_);

        // Step 2: Inverse complex FFTs along Z (now fully local)
        // Unpack recvbuf_z to assemble full NZ, compute NZ-point inverse FFTs
        std::vector<Complex> work(recvbuf_z);
        // ... inverse FFTs along Z ...

        // Step 3: All-to-all along Y to make Y local
        int chunk_y = cplx_local / npuy_;
        std::vector<Complex> recvbuf_y(cplx_local);
        MPI_Alltoall(work.data(), chunk_y * 2, MPI_DOUBLE,
                     recvbuf_y.data(), chunk_y * 2, MPI_DOUBLE, comm_y_);

        // Step 4: Inverse complex FFTs along Y (now fully local)
        // ... inverse FFTs along Y ...

        // Step 5: Complex-to-real inverse FFTs along X
        double* out_real = reinterpret_cast<double*>(a);
        for (int k = 0; k < nz_hat_; ++k) {
            for (int j = 0; j < ny_hat_; ++j) {
                std::vector<Complex> x_freq(nx_half_ + 1);
                for (int i = 0; i <= nx_half_; ++i)
                    x_freq[i] = recvbuf_y[(i * ny_hat_ + j) * nz_hat_ + k];
                std::vector<double> row(nx_);
                irfft(x_freq, row, nx_);
                for (int i = 0; i < nx_; ++i)
                    out_real[(i * ny_hat_ + j) * nz_hat_ + k] = row[i];
            }
        }
    }

private:
    int nx_, ny_, nz_, npuy_, npuz_, mey_;
    int ny_hat_, nz_hat_, nx_half_;
    MPI_Comm comm_y_, comm_z_;
};
