// OpenMP implementation of a recursive three-step FFT algorithm

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

static const double PI = 4.0 * std::atan(1.0);
static constexpr int CACHE_SIZE = 1024;  // base-case threshold (tunable)

// Forward declarations
void fft(std::vector<Complex>& a, std::vector<Complex>& temp,
         const std::vector<Complex>& w, int n);

void recursive_three_step_fft(std::vector<Complex>& a,
                              std::vector<Complex>& temp,
                              const std::vector<Complex>& w, int n);

void openmp_parallel_fft(std::vector<Complex>& a,
                         std::vector<Complex>& temp,
                         const std::vector<Complex>& w, int n) {
    if (n <= CACHE_SIZE) {
        fft(a, temp, w, n);
        return;
    }

    int half = n / 2;

    #pragma omp parallel
    {
        // Step 1: Butterfly + twiddle factor multiplication
        #pragma omp for
        for (int i = 0; i < half; ++i) {
            Complex c0 = a[i];
            Complex c1 = a[i + half];
            temp[i]        = c0 + c1;
            temp[i + half] = (c0 - c1) * w[i];
        }

        // Step 2: Two recursive n/2-point FFTs
        #pragma omp for
        for (int j = 0; j < 2; ++j) {
            int offset = j * half;
            std::vector<Complex> sub(temp.begin() + offset,
                                     temp.begin() + offset + half);
            std::vector<Complex> sub_temp(half);
            std::vector<Complex> sub_w(w.begin() + half, w.end());
            recursive_three_step_fft(sub, sub_temp, sub_w, half);
            for (int i = 0; i < half; ++i) {
                temp[offset + i] = sub[i];
            }
        }

        // Step 3: Interleave (transposition)
        #pragma omp for
        for (int i = 0; i < half; ++i) {
            a[2 * i]     = temp[i];
            a[2 * i + 1] = temp[i + half];
        }
    }
}
