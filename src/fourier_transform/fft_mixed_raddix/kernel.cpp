// Radix-2, 3, 4, 5 and 8 FFT Kernel Routines
// Low-level butterfly operations for the mixed-radix FFT

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

static const double C31 = 0.86602540378443865;  // sin(pi/3) = sqrt(3)/2
static const double C32 = 0.5;
static const double C51 = 0.95105651629515357;  // sin(2*pi/5)
static const double C52 = 0.61803398874989485;  // (sqrt(5)-1)/2 * sin(2*pi/5) / sin(4*pi/5)
static const double C53 = 0.55901699437494742;  // (sqrt(5)-1)/4
static const double C54 = 0.25;
static const double C81 = 0.70710678118654752;  // 1/sqrt(2)

// ============================================================
// Radix-2 butterfly
// ============================================================
void fft2(std::vector<Complex>& a, std::vector<Complex>& b, int m) {
    for (int i = 0; i < m; ++i) {
        Complex c0 = a[i];
        Complex c1 = a[i + m];
        b[i]     = c0 + c1;
        b[i + m] = c0 - c1;
    }
}

// ============================================================
// Radix-3 butterfly (stride m == 1)
// ============================================================
void fft3a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l) {
    for (int j = 0; j < l; ++j) {
        Complex w1 = w[j];
        Complex w2 = w[l + j];
        Complex c0 = a[j];
        Complex c1 = a[j + l];
        Complex c2 = a[j + 2 * l];
        Complex d0 = c1 + c2;
        Complex d1 = c0 - C32 * d0;
        Complex d2 = Complex(0.0, -1.0) * C31 * (c1 - c2);
        b[3 * j]     = c0 + d0;
        b[3 * j + 1] = w1 * (d1 + d2);
        b[3 * j + 2] = w2 * (d1 - d2);
    }
}

// Radix-3 butterfly (general stride m > 1)
void fft3b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l) {
    // First group (j == 0): no twiddle factors
    for (int i = 0; i < m; ++i) {
        Complex c0 = a[i];
        Complex c1 = a[i + m * l];
        Complex c2 = a[i + 2 * m * l];
        Complex d0 = c1 + c2;
        Complex d1 = c0 - C32 * d0;
        Complex d2 = Complex(0.0, -1.0) * C31 * (c1 - c2);
        b[i]         = c0 + d0;
        b[i + m]     = d1 + d2;
        b[i + 2 * m] = d1 - d2;
    }
    // Remaining groups (j >= 1): apply twiddle factors
    for (int j = 1; j < l; ++j) {
        Complex w1 = w[j];
        Complex w2 = w[l + j];
        for (int i = 0; i < m; ++i) {
            Complex c0 = a[i + j * m];
            Complex c1 = a[i + j * m + m * l];
            Complex c2 = a[i + j * m + 2 * m * l];
            Complex d0 = c1 + c2;
            Complex d1 = c0 - C32 * d0;
            Complex d2 = Complex(0.0, -1.0) * C31 * (c1 - c2);
            b[i + 3 * j * m]         = c0 + d0;
            b[i + 3 * j * m + m]     = w1 * (d1 + d2);
            b[i + 3 * j * m + 2 * m] = w2 * (d1 - d2);
        }
    }
}

// ============================================================
// Radix-4 butterfly (stride m == 1)
// ============================================================
void fft4a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l) {
    for (int j = 0; j < l; ++j) {
        Complex w1 = w[j];
        Complex w2 = w[l + j];
        Complex w3 = w[2 * l + j];
        Complex c0 = a[j];
        Complex c1 = a[j + l];
        Complex c2 = a[j + 2 * l];
        Complex c3 = a[j + 3 * l];
        Complex d0 = c0 + c2;
        Complex d1 = c0 - c2;
        Complex d2 = c1 + c3;
        Complex d3 = Complex(0.0, -1.0) * (c1 - c3);
        b[4 * j]     = d0 + d2;
        b[4 * j + 1] = w1 * (d1 + d3);
        b[4 * j + 2] = w2 * (d0 - d2);
        b[4 * j + 3] = w3 * (d1 - d3);
    }
}

// Radix-4 butterfly (general stride m > 1)
void fft4b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l) {
    // First group (j == 0): no twiddle factors
    for (int i = 0; i < m; ++i) {
        Complex c0 = a[i];
        Complex c1 = a[i + m * l];
        Complex c2 = a[i + 2 * m * l];
        Complex c3 = a[i + 3 * m * l];
        Complex d0 = c0 + c2;
        Complex d1 = c0 - c2;
        Complex d2 = c1 + c3;
        Complex d3 = Complex(0.0, -1.0) * (c1 - c3);
        b[i]             = d0 + d2;
        b[i + m]         = d1 + d3;
        b[i + 2 * m]     = d0 - d2;
        b[i + 3 * m]     = d1 - d3;
    }
    // Remaining groups
    for (int j = 1; j < l; ++j) {
        Complex w1 = w[j];
        Complex w2 = w[l + j];
        Complex w3 = w[2 * l + j];
        for (int i = 0; i < m; ++i) {
            Complex c0 = a[i + j * m];
            Complex c1 = a[i + j * m + m * l];
            Complex c2 = a[i + j * m + 2 * m * l];
            Complex c3 = a[i + j * m + 3 * m * l];
            Complex d0 = c0 + c2;
            Complex d1 = c0 - c2;
            Complex d2 = c1 + c3;
            Complex d3 = Complex(0.0, -1.0) * (c1 - c3);
            b[i + 4 * j * m]         = d0 + d2;
            b[i + 4 * j * m + m]     = w1 * (d1 + d3);
            b[i + 4 * j * m + 2 * m] = w2 * (d0 - d2);
            b[i + 4 * j * m + 3 * m] = w3 * (d1 - d3);
        }
    }
}

// ============================================================
// Radix-5 butterfly (stride m == 1)
// ============================================================
void fft5a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l) {
    for (int j = 0; j < l; ++j) {
        Complex w1 = w[j];
        Complex w2 = w[l + j];
        Complex w3 = w[2 * l + j];
        Complex w4 = w[3 * l + j];
        Complex c0 = a[j];
        Complex c1 = a[j + l];
        Complex c2 = a[j + 2 * l];
        Complex c3 = a[j + 3 * l];
        Complex c4 = a[j + 4 * l];
        Complex d0 = c1 + c4;
        Complex d1 = c2 + c3;
        Complex d2 = C51 * (c1 - c4);
        Complex d3 = C51 * (c2 - c3);
        Complex d4 = d0 + d1;
        Complex d5 = C53 * (d0 - d1);
        Complex d6 = c0 - C54 * d4;
        Complex d7 = d6 + d5;
        Complex d8 = d6 - d5;
        Complex d9  = Complex(0.0, -1.0) * (d2 + C52 * d3);
        Complex d10 = Complex(0.0, -1.0) * (C52 * d2 - d3);
        b[5 * j]     = c0 + d4;
        b[5 * j + 1] = w1 * (d7 + d9);
        b[5 * j + 2] = w2 * (d8 + d10);
        b[5 * j + 3] = w3 * (d8 - d10);
        b[5 * j + 4] = w4 * (d7 - d9);
    }
}

// Radix-5 butterfly (general stride m > 1)
void fft5b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l) {
    // First group (j == 0)
    for (int i = 0; i < m; ++i) {
        Complex c0 = a[i];
        Complex c1 = a[i + m * l];
        Complex c2 = a[i + 2 * m * l];
        Complex c3 = a[i + 3 * m * l];
        Complex c4 = a[i + 4 * m * l];
        Complex d0 = c1 + c4;
        Complex d1 = c2 + c3;
        Complex d2 = C51 * (c1 - c4);
        Complex d3 = C51 * (c2 - c3);
        Complex d4 = d0 + d1;
        Complex d5 = C53 * (d0 - d1);
        Complex d6 = c0 - C54 * d4;
        Complex d7 = d6 + d5;
        Complex d8 = d6 - d5;
        Complex d9  = Complex(0.0, -1.0) * (d2 + C52 * d3);
        Complex d10 = Complex(0.0, -1.0) * (C52 * d2 - d3);
        b[i]             = c0 + d4;
        b[i + m]         = d7 + d9;
        b[i + 2 * m]     = d8 + d10;
        b[i + 3 * m]     = d8 - d10;
        b[i + 4 * m]     = d7 - d9;
    }
    // Remaining groups
    for (int j = 1; j < l; ++j) {
        Complex w1 = w[j];
        Complex w2 = w[l + j];
        Complex w3 = w[2 * l + j];
        Complex w4 = w[3 * l + j];
        for (int i = 0; i < m; ++i) {
            Complex c0 = a[i + j * m];
            Complex c1 = a[i + j * m + m * l];
            Complex c2 = a[i + j * m + 2 * m * l];
            Complex c3 = a[i + j * m + 3 * m * l];
            Complex c4 = a[i + j * m + 4 * m * l];
            Complex d0 = c1 + c4;
            Complex d1 = c2 + c3;
            Complex d2 = C51 * (c1 - c4);
            Complex d3 = C51 * (c2 - c3);
            Complex d4 = d0 + d1;
            Complex d5 = C53 * (d0 - d1);
            Complex d6 = c0 - C54 * d4;
            Complex d7 = d6 + d5;
            Complex d8 = d6 - d5;
            Complex d9  = Complex(0.0, -1.0) * (d2 + C52 * d3);
            Complex d10 = Complex(0.0, -1.0) * (C52 * d2 - d3);
            b[i + 5 * j * m]         = c0 + d4;
            b[i + 5 * j * m + m]     = w1 * (d7 + d9);
            b[i + 5 * j * m + 2 * m] = w2 * (d8 + d10);
            b[i + 5 * j * m + 3 * m] = w3 * (d8 - d10);
            b[i + 5 * j * m + 4 * m] = w4 * (d7 - d9);
        }
    }
}

// ============================================================
// Radix-8 butterfly (stride m == 1)
// ============================================================
void fft8a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l) {
    for (int j = 0; j < l; ++j) {
        Complex w1 = w[j];
        Complex w2 = w[l + j];
        Complex w3 = w[2 * l + j];
        Complex w4 = w[3 * l + j];
        Complex w5 = w[4 * l + j];
        Complex w6 = w[5 * l + j];
        Complex w7 = w[6 * l + j];
        Complex c0 = a[j];
        Complex c1 = a[j + l];
        Complex c2 = a[j + 2 * l];
        Complex c3 = a[j + 3 * l];
        Complex c4 = a[j + 4 * l];
        Complex c5 = a[j + 5 * l];
        Complex c6 = a[j + 6 * l];
        Complex c7 = a[j + 7 * l];
        Complex d0 = c0 + c4;
        Complex d1 = c0 - c4;
        Complex d2 = c2 + c6;
        Complex d3 = Complex(0.0, -1.0) * (c2 - c6);
        Complex d4 = c1 + c5;
        Complex d5 = c1 - c5;
        Complex d6 = c3 + c7;
        Complex d7 = c3 - c7;
        Complex e0 = d0 + d2;
        Complex e1 = d0 - d2;
        Complex e2 = d4 + d6;
        Complex e3 = Complex(0.0, -1.0) * (d4 - d6);
        Complex e4 = C81 * (d5 - d7);
        Complex e5 = Complex(0.0, -1.0) * C81 * (d5 + d7);
        Complex e6 = d1 + e4;
        Complex e7 = d1 - e4;
        Complex e8 = d3 + e5;
        Complex e9 = d3 - e5;
        b[8 * j]     = e0 + e2;
        b[8 * j + 1] = w1 * (e6 + e8);
        b[8 * j + 2] = w2 * (e1 + e3);
        b[8 * j + 3] = w3 * (e7 - e9);
        b[8 * j + 4] = w4 * (e0 - e2);
        b[8 * j + 5] = w5 * (e7 + e9);
        b[8 * j + 6] = w6 * (e1 - e3);
        b[8 * j + 7] = w7 * (e6 - e8);
    }
}

// Radix-8 butterfly (general stride m > 1)
void fft8b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l) {
    // First group (j == 0)
    for (int i = 0; i < m; ++i) {
        Complex c0 = a[i];
        Complex c1 = a[i + m * l];
        Complex c2 = a[i + 2 * m * l];
        Complex c3 = a[i + 3 * m * l];
        Complex c4 = a[i + 4 * m * l];
        Complex c5 = a[i + 5 * m * l];
        Complex c6 = a[i + 6 * m * l];
        Complex c7 = a[i + 7 * m * l];
        Complex d0 = c0 + c4;
        Complex d1 = c0 - c4;
        Complex d2 = c2 + c6;
        Complex d3 = Complex(0.0, -1.0) * (c2 - c6);
        Complex d4 = c1 + c5;
        Complex d5 = c1 - c5;
        Complex d6 = c3 + c7;
        Complex d7 = c3 - c7;
        Complex e0 = d0 + d2;
        Complex e1 = d0 - d2;
        Complex e2 = d4 + d6;
        Complex e3 = Complex(0.0, -1.0) * (d4 - d6);
        Complex e4 = C81 * (d5 - d7);
        Complex e5 = Complex(0.0, -1.0) * C81 * (d5 + d7);
        Complex e6 = d1 + e4;
        Complex e7 = d1 - e4;
        Complex e8 = d3 + e5;
        Complex e9 = d3 - e5;
        b[i]             = e0 + e2;
        b[i + m]         = e6 + e8;
        b[i + 2 * m]     = e1 + e3;
        b[i + 3 * m]     = e7 - e9;
        b[i + 4 * m]     = e0 - e2;
        b[i + 5 * m]     = e7 + e9;
        b[i + 6 * m]     = e1 - e3;
        b[i + 7 * m]     = e6 - e8;
    }
    // Remaining groups
    for (int j = 1; j < l; ++j) {
        Complex w1 = w[j];
        Complex w2 = w[l + j];
        Complex w3 = w[2 * l + j];
        Complex w4 = w[3 * l + j];
        Complex w5 = w[4 * l + j];
        Complex w6 = w[5 * l + j];
        Complex w7 = w[6 * l + j];
        for (int i = 0; i < m; ++i) {
            Complex c0 = a[i + j * m];
            Complex c1 = a[i + j * m + m * l];
            Complex c2 = a[i + j * m + 2 * m * l];
            Complex c3 = a[i + j * m + 3 * m * l];
            Complex c4 = a[i + j * m + 4 * m * l];
            Complex c5 = a[i + j * m + 5 * m * l];
            Complex c6 = a[i + j * m + 6 * m * l];
            Complex c7 = a[i + j * m + 7 * m * l];
            Complex d0 = c0 + c4;
            Complex d1 = c0 - c4;
            Complex d2 = c2 + c6;
            Complex d3 = Complex(0.0, -1.0) * (c2 - c6);
            Complex d4 = c1 + c5;
            Complex d5 = c1 - c5;
            Complex d6 = c3 + c7;
            Complex d7 = c3 - c7;
            Complex e0 = d0 + d2;
            Complex e1 = d0 - d2;
            Complex e2 = d4 + d6;
            Complex e3 = Complex(0.0, -1.0) * (d4 - d6);
            Complex e4 = C81 * (d5 - d7);
            Complex e5 = Complex(0.0, -1.0) * C81 * (d5 + d7);
            Complex e6 = d1 + e4;
            Complex e7 = d1 - e4;
            Complex e8 = d3 + e5;
            Complex e9 = d3 - e5;
            b[i + 8 * j * m]         = e0 + e2;
            b[i + 8 * j * m + m]     = w1 * (e6 + e8);
            b[i + 8 * j * m + 2 * m] = w2 * (e1 + e3);
            b[i + 8 * j * m + 3 * m] = w3 * (e7 - e9);
            b[i + 8 * j * m + 4 * m] = w4 * (e0 - e2);
            b[i + 8 * j * m + 5 * m] = w5 * (e7 + e9);
            b[i + 8 * j * m + 6 * m] = w6 * (e1 - e3);
            b[i + 8 * j * m + 7 * m] = w7 * (e6 - e8);
        }
    }
}
