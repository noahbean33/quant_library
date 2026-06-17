// Complex-to-real FFT routine

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

// Forward declaration: assumes a complex FFT routine is available
void fft(std::vector<Complex>& x, int n, int m);

void rfft(std::vector<Complex>& a, int n) {
    const double pi = 4.0 * std::atan(1.0);
    const double px = -2.0 * pi / static_cast<double>(n);
    const int half = n / 2;
    const int quarter = n / 4;

    // Compute log2(n/2) for the complex FFT call
    int m = 0;
    for (int tmp = half; tmp > 1; tmp >>= 1) {
        ++m;
    }

    // Step 1: n/2-point complex FFT
    fft(a, half, m);

    // Step 2: Unscramble DC and Nyquist components
    double temp = a[0].real() - a[0].imag();
    a[0] = Complex(a[0].real() + a[0].imag(), 0.0);
    a[quarter] = Complex(a[quarter].real(), -a[quarter].imag());
    a[half] = Complex(temp, 0.0);

    // Step 3: Unscramble remaining frequency bins
    for (int i = 1; i < quarter; ++i) {
        double w = px * static_cast<double>(i);

        double ar = 0.5 * (a[i].real() - a[half - i].real());
        double ai = 0.5 * (a[i].imag() + a[half - i].imag());

        double wr = 1.0 - std::sin(w);
        double wi = std::cos(w);

        temp = ar * wi + ai * wr;
        ar   = ar * wr - ai * wi;
        ai   = temp;

        a[i]        = Complex(a[i].real() - ar, a[i].imag() - ai);
        a[half - i] = Complex(a[half - i].real() + ar, a[half - i].imag() - ai);
    }
}
