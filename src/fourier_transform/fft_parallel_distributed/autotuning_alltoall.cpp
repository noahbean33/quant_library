// Automatic tuning of all-to-all communication (MPI)

#include <cmath>
#include <cfloat>
#include <vector>
#include <complex>
#include <mpi.h>

using Complex = std::complex<double>;

// Forward declaration: two-step all-to-all using 2-D process grid
void two_step_alltoall(const Complex* sendbuf, Complex* recvbuf,
                       int count, int Px, int Py, MPI_Comm comm);

struct AlltoallParams {
    int Px, Py;
};

AlltoallParams autotune_alltoall(int P, Complex* sendbuf, Complex* recvbuf,
                                int count, int iter_num, MPI_Comm comm) {
    double min_time = DBL_MAX;
    AlltoallParams best = {1, P};

    int log2_P = static_cast<int>(std::log2(static_cast<double>(P)));

    for (int i = 0; i <= log2_P; ++i) {
        int Px = 1 << i;
        int Py = P / Px;

        MPI_Barrier(comm);
        double start = MPI_Wtime();

        for (int c = 0; c < iter_num; ++c) {
            if (Px == 1 || Py == 1) {
                MPI_Alltoall(sendbuf, count * 2, MPI_DOUBLE,
                             recvbuf, count * 2, MPI_DOUBLE, comm);
            } else {
                two_step_alltoall(sendbuf, recvbuf, count, Px, Py, comm);
            }
        }

        MPI_Barrier(comm);
        double end = MPI_Wtime();

        if (end - start < min_time) {
            min_time = end - start;
            best.Px = Px;
            best.Py = Py;
        }
    }

    return best;
}
