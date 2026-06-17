// Parallel one-dimensional FFT using cyclic distribution (MPI)

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

static const double PI = 4.0 * std::atan(1.0);

// Forward declaration: a local 1-D FFT routine
void fft(std::vector<Complex>& col, int n);

void parallel_1d_fft_cyclic(std::vector<Complex>& x_local,
                            std::vector<Complex>& y_local,
                            int N1, int N2, MPI_Comm comm) {
    int P, rank;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &rank);

    int N = N1 * N2;
    int N1_hat = N1 / P;
    int N2_hat = N2 / P;
    int local_size = N / P;  // = N1_hat * N2

    std::vector<Complex> work(local_size);
    std::vector<Complex> buf(local_size);

    // Step 1: Local transposition (N1_hat × N2) → (N2 × N1_hat)
    for (int j1h = 0; j1h < N1_hat; ++j1h)
        for (int j2 = 0; j2 < N2; ++j2)
            work[j2 * N1_hat + j1h] = x_local[j1h * N2 + j2];

    // Step 2: N1_hat individual N2-point FFTs
    for (int j1h = 0; j1h < N1_hat; ++j1h) {
        std::vector<Complex> col(N2);
        for (int j2 = 0; j2 < N2; ++j2)
            col[j2] = work[j2 * N1_hat + j1h];
        fft(col, N2);
        for (int j2 = 0; j2 < N2; ++j2)
            work[j2 * N1_hat + j1h] = col[j2];
    }

    // Step 3: Twiddle factor multiplication + transposition
    // Cyclic: global J1 = J1_hat * P + rank
    for (int j1h = 0; j1h < N1_hat; ++j1h) {
        int j1_global = j1h * P + rank;
        for (int k2 = 0; k2 < N2; ++k2) {
            double angle = -2.0 * PI * static_cast<double>(j1_global * k2)
                           / static_cast<double>(N);
            Complex tw(std::cos(angle), std::sin(angle));
            buf[j1h * N2 + k2] = work[k2 * N1_hat + j1h] * tw;
        }
    }

    // Step 4: Rearrangement into (J1_hat, K2_tilde, P2) for all-to-all
    for (int p = 0; p < P; ++p) {
        for (int k2t = 0; k2t < N2_hat; ++k2t) {
            int k2 = p * N2_hat + k2t;
            for (int j1h = 0; j1h < N1_hat; ++j1h) {
                work[p * (N1_hat * N2_hat) + k2t * N1_hat + j1h] =
                    buf[j1h * N2 + k2];
            }
        }
    }

    // Step 5: All-to-all communication (single exchange)
    int chunk = N1_hat * N2_hat;
    MPI_Alltoall(work.data(), chunk * 2, MPI_DOUBLE,
                 buf.data(), chunk * 2, MPI_DOUBLE, comm);

    // Step 6: Transposition — assemble full J1 from (J1_tilde, K2_hat, P1)
    // Global J1 = P1 * J1_tilde + ... ; reconstruct (J1, K2_hat)
    for (int p = 0; p < P; ++p) {
        for (int j1t = 0; j1t < N1_hat; ++j1t) {
            int j1 = p * N1_hat + j1t;
            for (int k2h = 0; k2h < N2_hat; ++k2h) {
                work[j1 * N2_hat + k2h] =
                    buf[p * chunk + j1t * N2_hat + k2h];
            }
        }
    }

    // Step 7: N2_hat individual N1-point FFTs
    for (int k2h = 0; k2h < N2_hat; ++k2h) {
        std::vector<Complex> col(N1);
        for (int j1 = 0; j1 < N1; ++j1)
            col[j1] = work[j1 * N2_hat + k2h];
        fft(col, N1);
        for (int j1 = 0; j1 < N1; ++j1)
            work[j1 * N2_hat + k2h] = col[j1];
    }

    // Step 8: Transposition → final output y_local(K2_hat, K1)
    y_local.resize(local_size);
    for (int k1 = 0; k1 < N1; ++k1)
        for (int k2h = 0; k2h < N2_hat; ++k2h)
            y_local[k2h * N1 + k1] = work[k1 * N2_hat + k2h];
}
