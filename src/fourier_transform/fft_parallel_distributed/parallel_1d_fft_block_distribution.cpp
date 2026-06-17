// Parallel one-dimensional FFT using block distribution (MPI)

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

static const double PI = 4.0 * std::atan(1.0);

// Forward declaration: a local 1-D FFT routine
void fft(std::vector<Complex>& col, int n);

// Local blocked transpose: src(rows x cols) -> dst(cols x rows)
static void local_transpose(const std::vector<Complex>& src,
                            std::vector<Complex>& dst,
                            int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst[j * rows + i] = src[i * cols + j];
}

void parallel_1d_fft_block(std::vector<Complex>& x_local,
                           std::vector<Complex>& y_local,
                           int N1, int N2, MPI_Comm comm) {
    int P, rank;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &rank);

    int N = N1 * N2;
    int N1_hat = N1 / P;
    int N2_hat = N2 / P;
    int local_size = N / P;  // = N1 * N2_hat = N1_hat * N2

    std::vector<Complex> work(local_size);
    std::vector<Complex> buf(local_size);

    // Step 1: Rearrangement for all-to-all
    // x_local is (N1, N2_hat), rearrange into (N1_tilde, N2_hat, P1)
    // where N1_tilde = N1/P = N1_hat
    for (int p = 0; p < P; ++p) {
        for (int j2 = 0; j2 < N2_hat; ++j2) {
            for (int j1t = 0; j1t < N1_hat; ++j1t) {
                int j1 = p * N1_hat + j1t;
                work[p * (N1_hat * N2_hat) + j2 * N1_hat + j1t] =
                    x_local[j1 * N2_hat + j2];
            }
        }
    }

    // Step 2: All-to-all communication
    int chunk = N1_hat * N2_hat;
    MPI_Alltoall(work.data(), chunk * 2, MPI_DOUBLE,
                 buf.data(), chunk * 2, MPI_DOUBLE, comm);

    // Step 3: Transposition to (J2, J1_hat)
    // buf is (N1_hat, N2_tilde, P2) per process; assemble full N2 dim
    for (int p = 0; p < P; ++p) {
        for (int j2t = 0; j2t < N2_hat; ++j2t) {
            int j2 = p * N2_hat + j2t;
            for (int j1h = 0; j1h < N1_hat; ++j1h) {
                work[j2 * N1_hat + j1h] =
                    buf[p * chunk + j2t * N1_hat + j1h];
            }
        }
    }

    // Step 4: N1_hat individual N2-point FFTs (columns of work viewed as N2 x N1_hat)
    for (int j1h = 0; j1h < N1_hat; ++j1h) {
        std::vector<Complex> col(N2);
        for (int j2 = 0; j2 < N2; ++j2)
            col[j2] = work[j2 * N1_hat + j1h];
        fft(col, N2);
        for (int j2 = 0; j2 < N2; ++j2)
            work[j2 * N1_hat + j1h] = col[j2];
    }

    // Step 5: Twiddle factor multiplication + rearrangement
    int j1_base = rank * N1_hat;
    for (int p = 0; p < P; ++p) {
        for (int j1h = 0; j1h < N1_hat; ++j1h) {
            int j1_global = j1_base + j1h;
            for (int k2t = 0; k2t < N2_hat; ++k2t) {
                int k2 = p * N2_hat + k2t;
                double angle = -2.0 * PI * static_cast<double>(j1_global * k2)
                               / static_cast<double>(N);
                Complex tw(std::cos(angle), std::sin(angle));
                buf[p * chunk + j1h * N2_hat + k2t] =
                    work[k2 * N1_hat + j1h] * tw;
            }
        }
    }

    // Step 6: All-to-all communication
    MPI_Alltoall(buf.data(), chunk * 2, MPI_DOUBLE,
                 work.data(), chunk * 2, MPI_DOUBLE, comm);

    // Step 7: Transposition to (J1, K2_hat)
    for (int p = 0; p < P; ++p) {
        for (int j1t = 0; j1t < N1_hat; ++j1t) {
            int j1 = p * N1_hat + j1t;
            for (int k2h = 0; k2h < N2_hat; ++k2h) {
                buf[j1 * N2_hat + k2h] =
                    work[p * chunk + j1t * N2_hat + k2h];
            }
        }
    }

    // Step 8: N2_hat individual N1-point FFTs
    for (int k2h = 0; k2h < N2_hat; ++k2h) {
        std::vector<Complex> col(N1);
        for (int j1 = 0; j1 < N1; ++j1)
            col[j1] = buf[j1 * N2_hat + k2h];
        fft(col, N1);
        for (int j1 = 0; j1 < N1; ++j1)
            buf[j1 * N2_hat + k2h] = col[j1];
    }

    // Step 9: Rearrangement for final all-to-all
    for (int p = 0; p < P; ++p) {
        for (int k2h = 0; k2h < N2_hat; ++k2h) {
            for (int k1t = 0; k1t < N1_hat; ++k1t) {
                int k1 = p * N1_hat + k1t;
                work[p * chunk + k2h * N1_hat + k1t] =
                    buf[k1 * N2_hat + k2h];
            }
        }
    }

    // Step 10: All-to-all communication
    MPI_Alltoall(work.data(), chunk * 2, MPI_DOUBLE,
                 buf.data(), chunk * 2, MPI_DOUBLE, comm);

    // Step 11: Transposition → final output y_local(K2, K1_hat)
    y_local.resize(local_size);
    for (int p = 0; p < P; ++p) {
        for (int k2t = 0; k2t < N2_hat; ++k2t) {
            int k2 = p * N2_hat + k2t;
            for (int k1h = 0; k1h < N1_hat; ++k1h) {
                y_local[k2 * N1_hat + k1h] =
                    buf[p * chunk + k2t * N1_hat + k1h];
            }
        }
    }
}
