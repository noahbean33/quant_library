// Parallel three-dimensional FFT in distributed-memory parallel computers (MPI)

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <mpi.h>

using Complex = std::complex<double>;

// Forward declaration: a local 1-D FFT routine
void fft(std::vector<Complex>& col, int n);

// Helper: 3-D index in row-major (d1 x d2 x d3)
static inline int idx3(int i, int j, int k, int d2, int d3) {
    return (i * d2 + j) * d3 + k;
}

void parallel_3d_fft(std::vector<Complex>& x_local,
                     std::vector<Complex>& y_local,
                     int N1, int N2, int N3, MPI_Comm comm) {
    int P, rank;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &rank);

    int N1_hat = N1 / P;
    int N3_hat = N3 / P;
    int local_size = N1 * N2 * N3_hat;  // initial local size

    std::vector<Complex> work(local_size);
    std::vector<Complex> buf(local_size);

    // x_local is (N1, N2, N3_hat) row-major

    // Step 1: N2*N3_hat individual N1-point FFTs (along dim 1)
    for (int k = 0; k < N3_hat; ++k) {
        for (int j = 0; j < N2; ++j) {
            std::vector<Complex> col(N1);
            for (int i = 0; i < N1; ++i)
                col[i] = x_local[idx3(i, j, k, N2, N3_hat)];
            fft(col, N1);
            for (int i = 0; i < N1; ++i)
                x_local[idx3(i, j, k, N2, N3_hat)] = col[i];
        }
    }

    // Step 2: Local transposition x2(J2, K1, J3_hat) = x1(K1, J2, J3_hat)
    for (int k = 0; k < N3_hat; ++k)
        for (int k1 = 0; k1 < N1; ++k1)
            for (int j2 = 0; j2 < N2; ++j2)
                work[idx3(j2, k1, k, N1, N3_hat)] =
                    x_local[idx3(k1, j2, k, N2, N3_hat)];

    // Step 3: N1*N3_hat individual N2-point FFTs (along dim 2, now dim 1 of work)
    for (int k = 0; k < N3_hat; ++k) {
        for (int k1 = 0; k1 < N1; ++k1) {
            std::vector<Complex> col(N2);
            for (int j2 = 0; j2 < N2; ++j2)
                col[j2] = work[idx3(j2, k1, k, N1, N3_hat)];
            fft(col, N2);
            for (int j2 = 0; j2 < N2; ++j2)
                work[idx3(j2, k1, k, N1, N3_hat)] = col[j2];
        }
    }
    // work now holds x3(K2, K1, J3_hat) in (N2, N1, N3_hat) layout

    // Step 4: Rearrangement for all-to-all
    // Decompose K1 into (K1_tilde, P1): K1 = p*N1_hat + k1t
    // Pack into (K1_tilde, K2, J3_hat) chunks of size N1_hat*N2*N3_hat per process
    int chunk = N1_hat * N2 * N3_hat;
    for (int p = 0; p < P; ++p) {
        for (int k2 = 0; k2 < N2; ++k2) {
            for (int k1t = 0; k1t < N1_hat; ++k1t) {
                int k1 = p * N1_hat + k1t;
                for (int j3h = 0; j3h < N3_hat; ++j3h) {
                    buf[p * chunk + (k1t * N2 + k2) * N3_hat + j3h] =
                        work[idx3(k2, k1, j3h, N1, N3_hat)];
                }
            }
        }
    }

    // Step 5: All-to-all communication
    MPI_Alltoall(buf.data(), chunk * 2, MPI_DOUBLE,
                 work.data(), chunk * 2, MPI_DOUBLE, comm);

    // Step 6: Transposition — assemble full J3 and rearrange to (J3, K1_hat, K2)
    // Received: P chunks of (N1_hat, N2, N3_hat), each from different process
    // Assemble J3 = p*N3_hat + j3t
    // Output layout: (N3, N1_hat, N2) = (J3, K1_hat, K2)
    int local_size2 = N3 * N1_hat * N2;
    std::vector<Complex> phase2(local_size2);
    for (int p = 0; p < P; ++p) {
        for (int k1h = 0; k1h < N1_hat; ++k1h) {
            for (int k2 = 0; k2 < N2; ++k2) {
                for (int j3t = 0; j3t < N3_hat; ++j3t) {
                    int j3 = p * N3_hat + j3t;
                    phase2[idx3(j3, k1h, k2, N1_hat, N2)] =
                        work[p * chunk + (k1h * N2 + k2) * N3_hat + j3t];
                }
            }
        }
    }

    // Step 7: N1_hat*N2 individual N3-point FFTs (along dim 3, now dim 1)
    for (int k2 = 0; k2 < N2; ++k2) {
        for (int k1h = 0; k1h < N1_hat; ++k1h) {
            std::vector<Complex> col(N3);
            for (int j3 = 0; j3 < N3; ++j3)
                col[j3] = phase2[idx3(j3, k1h, k2, N1_hat, N2)];
            fft(col, N3);
            for (int j3 = 0; j3 < N3; ++j3)
                phase2[idx3(j3, k1h, k2, N1_hat, N2)] = col[j3];
        }
    }

    // Step 8: Transposition + pack for second all-to-all
    // phase2 is (K3, K1_hat, K2). Decompose K3 into (K3_tilde, P3)
    // Pack as (K1_hat, K2, K3_tilde) chunks
    for (int p = 0; p < P; ++p) {
        for (int k1h = 0; k1h < N1_hat; ++k1h) {
            for (int k2 = 0; k2 < N2; ++k2) {
                for (int k3t = 0; k3t < N3_hat; ++k3t) {
                    int k3 = p * N3_hat + k3t;
                    buf[p * chunk + (k1h * N2 + k2) * N3_hat + k3t] =
                        phase2[idx3(k3, k1h, k2, N1_hat, N2)];
                }
            }
        }
    }

    // Step 9: All-to-all communication
    MPI_Alltoall(buf.data(), chunk * 2, MPI_DOUBLE,
                 work.data(), chunk * 2, MPI_DOUBLE, comm);

    // Step 10: Rearrangement → final output y_local(K1, K2, K3_hat)
    y_local.resize(N1 * N2 * N3_hat);
    for (int p = 0; p < P; ++p) {
        for (int k1t = 0; k1t < N1_hat; ++k1t) {
            int k1 = p * N1_hat + k1t;
            for (int k2 = 0; k2 < N2; ++k2) {
                for (int k3h = 0; k3h < N3_hat; ++k3h) {
                    y_local[idx3(k1, k2, k3h, N2, N3_hat)] =
                        work[p * chunk + (k1t * N2 + k2) * N3_hat + k3h];
                }
            }
        }
    }
}
