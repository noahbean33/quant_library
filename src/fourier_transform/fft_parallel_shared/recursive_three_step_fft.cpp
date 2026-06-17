// Recursive three-step FFT algorithm

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

static const double PI = 4.0 * std::atan(1.0);
static constexpr int CACHE_SIZE = 1024;  // base-case threshold (tunable)

// Forward declaration: a standard in-cache FFT
void fft(std::vector<Complex>& a, std::vector<Complex>& temp,
         const std::vector<Complex>& w, int n);

void recursive_three_step_fft(std::vector<Complex>& a,
                              std::vector<Complex>& temp,
                              const std::vector<Complex>& w, int n) {
    if (n <= CACHE_SIZE) {
        fft(a, temp, w, n);
        return;
    }

    int half = n / 2;

    // Step 1: Butterfly + twiddle factor multiplication
    for (int i = 0; i < half; ++i) {
        Complex c0 = a[i];
        Complex c1 = a[i + half];
        temp[i]        = c0 + c1;
        temp[i + half] = (c0 - c1) * w[i];
    }

    // Step 2: Two recursive n/2-point FFTs
    // Build sub-arrays and sub-twiddle views
    std::vector<Complex> sub0(temp.begin(), temp.begin() + half);
    std::vector<Complex> sub1(temp.begin() + half, temp.begin() + n);
    std::vector<Complex> sub_temp(half);
    std::vector<Complex> sub_w(w.begin() + half, w.end());

    recursive_three_step_fft(sub0, sub_temp, sub_w, half);
    recursive_three_step_fft(sub1, sub_temp, sub_w, half);

    // Copy results back into temp
    for (int i = 0; i < half; ++i) {
        temp[i]        = sub0[i];
        temp[i + half] = sub1[i];
    }

    // Step 3: Transposition (interleave even/odd halves)
    for (int i = 0; i < half; ++i) {
        a[2 * i]     = temp[i];
        a[2 * i + 1] = temp[i + half];
    }
}
