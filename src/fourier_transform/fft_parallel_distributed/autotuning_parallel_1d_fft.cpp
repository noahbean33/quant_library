// Automatic tuning of parallel one-dimensional FFT (MPI)

#include <cmath>
#include <cfloat>
#include <algorithm>
#include <mpi.h>

// Forward declaration: the parallel 1-D FFT with tunable parameters
void parallel_1d_fft(int N1, int N2, int NB, int NDIV, MPI_Comm comm);

static int gcd(int a, int b) {
    while (b != 0) { int t = b; b = a % b; a = t; }
    return a;
}

struct FFTParams {
    int N1, N2, NB, NDIV;
};

FFTParams autotune_parallel_1d_fft(int N, int P, int iter_num,
                                   MPI_Comm comm) {
    double min_time = DBL_MAX;
    FFTParams best = {0, 0, 0, 0};

    int log2_P = static_cast<int>(std::log2(static_cast<double>(P)));
    int log2_sqrtN = static_cast<int>(std::log2(std::sqrt(static_cast<double>(N))));

    for (int k = 2; k <= 6; ++k) {
        int NB = 1 << k;  // 4, 8, 16, 32, 64

        for (int j = log2_P; j <= log2_sqrtN; ++j) {
            int N1 = 1 << j;
            int N2 = N / N1;

            if (N1 % P != 0 || N2 % P != 0) continue;

            for (int i = 1; i <= 16; ++i) {
                int NDIV = gcd(i, gcd(N1 / P, N2 / P));

                MPI_Barrier(comm);
                double start = MPI_Wtime();

                for (int count = 0; count < iter_num; ++count) {
                    parallel_1d_fft(N1, N2, NB, NDIV, comm);
                }

                MPI_Barrier(comm);
                double end = MPI_Wtime();

                if (end - start < min_time) {
                    min_time = end - start;
                    best.N1   = N1;
                    best.N2   = N2;
                    best.NB   = NB;
                    best.NDIV = NDIV;
                }
            }
        }
    }

    return best;
}
