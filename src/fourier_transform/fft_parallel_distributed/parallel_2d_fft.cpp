// Parallel two-dimensional FFT in distributed-memory parallel computers (MPI)

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

// Forward declaration: a local 1-D FFT routine
void fft(std::vector<Complex>& col, int n);

void parallel_2d_fft(std::vector<Complex>& x_local,
                     std::vector<Complex>& y_local,
                     int N1, int N2, MPI_Comm comm) {
    int P, rank;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &rank);

    int N1_hat = N1 / P;
    int N2_hat = N2 / P;
    int local_size = N1 * N2_hat;  // = N1_hat * N2

    std::vector<Complex> work(local_size);
    std::vector<Complex> buf(local_size);

    // Step 1: N2_hat individual N1-point FFTs (along dim 1)
    // x_local is (N1, N2_hat) row-major: x_local[i*N2_hat + j]
    for (int j = 0; j < N2_hat; ++j) {
        std::vector<Complex> col(N1);
        for (int i = 0; i < N1; ++i) col[i] = x_local[i * N2_hat + j];
        fft(col, N1);
        for (int i = 0; i < N1; ++i) x_local[i * N2_hat + j] = col[i];
    }

    // Step 2: Transposition + pack for all-to-all
    // Decompose K1 into (K1_tilde, P1) where K1_tilde = N1_hat
    int chunk = N1_hat * N2_hat;
    for (int p = 0; p < P; ++p) {
        for (int j2h = 0; j2h < N2_hat; ++j2h) {
            for (int k1t = 0; k1t < N1_hat; ++k1t) {
                int k1 = p * N1_hat + k1t;
                work[p * chunk + j2h * N1_hat + k1t] =
                    x_local[k1 * N2_hat + j2h];
            }
        }
    }

    // Step 3: All-to-all communication
    MPI_Alltoall(work.data(), chunk * 2, MPI_DOUBLE,
                 buf.data(), chunk * 2, MPI_DOUBLE, comm);

    // Step 4: Rearrangement — assemble full J2 from (J2_tilde, K1_hat, P2)
    // Result layout: (J2, K1_hat) where J2 = p*N2_hat + j2t
    for (int p = 0; p < P; ++p) {
        for (int j2t = 0; j2t < N2_hat; ++j2t) {
            int j2 = p * N2_hat + j2t;
            for (int k1h = 0; k1h < N1_hat; ++k1h) {
                work[j2 * N1_hat + k1h] =
                    buf[p * chunk + j2t * N1_hat + k1h];
            }
        }
    }

    // Step 5: N1_hat individual N2-point FFTs (along dim 2)
    for (int k1h = 0; k1h < N1_hat; ++k1h) {
        std::vector<Complex> col(N2);
        for (int j2 = 0; j2 < N2; ++j2)
            col[j2] = work[j2 * N1_hat + k1h];
        fft(col, N2);
        for (int j2 = 0; j2 < N2; ++j2)
            work[j2 * N1_hat + k1h] = col[j2];
    }

    // Step 6: Transposition + pack for second all-to-all
    // Decompose K2 into (K2_tilde, P2) where K2_tilde = N2_hat
    for (int p = 0; p < P; ++p) {
        for (int k1h = 0; k1h < N1_hat; ++k1h) {
            for (int k2t = 0; k2t < N2_hat; ++k2t) {
                int k2 = p * N2_hat + k2t;
                buf[p * chunk + k1h * N2_hat + k2t] =
                    work[k2 * N1_hat + k1h];
            }
        }
    }

    // Step 7: All-to-all communication
    MPI_Alltoall(buf.data(), chunk * 2, MPI_DOUBLE,
                 work.data(), chunk * 2, MPI_DOUBLE, comm);

    // Step 8: Rearrangement → final output y_local(K1, K2_hat)
    y_local.resize(local_size);
    for (int p = 0; p < P; ++p) {
        for (int k1t = 0; k1t < N1_hat; ++k1t) {
            int k1 = p * N1_hat + k1t;
            for (int k2h = 0; k2h < N2_hat; ++k2h) {
                y_local[k1 * N2_hat + k2h] =
                    work[p * chunk + k1t * N2_hat + k2h];
            }
        }
    }
}
