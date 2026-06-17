// Computation-communication overlap pattern (MPI + OpenMP)

#include <vector>
#include <complex>
#include <mpi.h>

using Complex = std::complex<double>;

// Forward declarations for application-specific routines
void compute_independent(std::vector<Complex>& data, int i);
void compute_with_comm_result(std::vector<Complex>& data,
                              const std::vector<Complex>& recv_buf, int i);

void overlap_comm_compute(std::vector<Complex>& data,
                          std::vector<Complex>& send_buf,
                          std::vector<Complex>& recv_buf,
                          int n, int count, MPI_Comm comm) {
    #pragma omp parallel
    {
        // Master thread performs MPI communication
        #pragma omp master
        {
            MPI_Alltoall(send_buf.data(), count * 2, MPI_DOUBLE,
                         recv_buf.data(), count * 2, MPI_DOUBLE, comm);
        }

        // All threads: independent computation (dynamic schedule
        // to handle master thread being busy with communication)
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            compute_independent(data, i);
        }

        // Implicit barrier ensures communication is complete

        // All threads: computation using the result of communication
        #pragma omp for
        for (int i = 0; i < n; ++i) {
            compute_with_comm_result(data, recv_buf, i);
        }
    }
}
