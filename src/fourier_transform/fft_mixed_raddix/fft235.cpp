// Radix-2, 3, 4, 5 and 8 FFT Routine
// Mixed-radix FFT driver and twiddle factor table setup

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

// Forward declarations for FFT kernel routines
void fft2(std::vector<Complex>& a, std::vector<Complex>& b, int m);
void fft3a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l);
void fft3b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l);
void fft4a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l);
void fft4b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l);
void fft5a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l);
void fft5b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l);
void fft8a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l);
void fft8b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l);
void factor(int n, int ip[3]);

// Dispatch to FFT3A or FFT3B depending on stride
static void fft3_dispatch(std::vector<Complex>& a, std::vector<Complex>& b,
                          const std::vector<Complex>& w, int m, int l) {
    if (m == 1) {
        fft3a(a, b, w, l);
    } else {
        fft3b(a, b, w, m, l);
    }
}

// Dispatch to FFT4A or FFT4B depending on stride
static void fft4_dispatch(std::vector<Complex>& a, std::vector<Complex>& b,
                          const std::vector<Complex>& w, int m, int l) {
    if (m == 1) {
        fft4a(a, b, w, l);
    } else {
        fft4b(a, b, w, m, l);
    }
}

// Dispatch to FFT5A or FFT5B depending on stride
static void fft5_dispatch(std::vector<Complex>& a, std::vector<Complex>& b,
                          const std::vector<Complex>& w, int m, int l) {
    if (m == 1) {
        fft5a(a, b, w, l);
    } else {
        fft5b(a, b, w, m, l);
    }
}

// Dispatch to FFT8A or FFT8B depending on stride
static void fft8_dispatch(std::vector<Complex>& a, std::vector<Complex>& b,
                          const std::vector<Complex>& w, int m, int l) {
    if (m == 1) {
        fft8a(a, b, w, l);
    } else {
        fft8b(a, b, w, m, l);
    }
}

// Mixed-radix FFT: n = 2^ip[0] * 3^ip[1] * 5^ip[2]
// a: input/output array, b: work array, w: twiddle factors
void fft235(std::vector<Complex>& a, std::vector<Complex>& b,
            const std::vector<Complex>& w, int n, const int ip[3]) {
    // Determine decomposition into radix-8, radix-4, and radix-2 stages
    int kp4, kp8;
    if (ip[0] != 1) {
        kp4 = 2 - (ip[0] + 2) % 3;
        kp8 = (ip[0] - kp4 * 2) / 3;
    } else {
        kp4 = 0;
        kp8 = 0;
    }

    int key = 1;
    int j = 0;  // Offset into twiddle factor array (0-indexed)
    int l = n;
    int m = 1;

    // Radix-8 stages
    for (int k = 0; k < kp8; ++k) {
        l /= 8;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 7);
        if (l >= 2) {
            if (key >= 0) {
                fft8_dispatch(a, b, wslice, m, l);
            } else {
                fft8_dispatch(b, a, wslice, m, l);
            }
            key = -key;
        } else {
            if (key >= 0) {
                fft8_dispatch(a, a, wslice, m, l);
            } else {
                fft8_dispatch(b, a, wslice, m, l);
            }
        }
        m *= 8;
        j += l * 7;
    }

    // Radix-5 stages
    for (int k = 0; k < ip[2]; ++k) {
        l /= 5;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 4);
        if (l >= 2) {
            if (key >= 0) {
                fft5_dispatch(a, b, wslice, m, l);
            } else {
                fft5_dispatch(b, a, wslice, m, l);
            }
            key = -key;
        } else {
            if (key >= 0) {
                fft5_dispatch(a, a, wslice, m, l);
            } else {
                fft5_dispatch(b, a, wslice, m, l);
            }
        }
        m *= 5;
        j += l * 4;
    }

    // Radix-4 stages
    for (int k = 0; k < kp4; ++k) {
        l /= 4;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 3);
        if (l >= 2) {
            if (key >= 0) {
                fft4_dispatch(a, b, wslice, m, l);
            } else {
                fft4_dispatch(b, a, wslice, m, l);
            }
            key = -key;
        } else {
            if (key >= 0) {
                fft4_dispatch(a, a, wslice, m, l);
            } else {
                fft4_dispatch(b, a, wslice, m, l);
            }
        }
        m *= 4;
        j += l * 3;
    }

    // Radix-3 stages
    for (int k = 0; k < ip[1]; ++k) {
        l /= 3;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 2);
        if (l >= 2) {
            if (key >= 0) {
                fft3_dispatch(a, b, wslice, m, l);
            } else {
                fft3_dispatch(b, a, wslice, m, l);
            }
            key = -key;
        } else {
            if (key >= 0) {
                fft3_dispatch(a, a, wslice, m, l);
            } else {
                fft3_dispatch(b, a, wslice, m, l);
            }
        }
        m *= 3;
        j += l * 2;
    }

    // Final radix-2 stage (if ip[0] == 1)
    if (ip[0] == 1) {
        if (key >= 0) {
            fft2(a, a, m);
        } else {
            fft2(b, a, m);
        }
    }
}

// Initialize twiddle factor table for a single radix stage
// w: output twiddle array of size (radix-1) x l
// radix: the radix (3, 4, 5, or 8)
// l: number of groups
static void settbl0(std::vector<Complex>& w, int offset, int radix, int l) {
    const double pi2 = 8.0 * std::atan(1.0);
    double px = -pi2 / (static_cast<double>(radix) * static_cast<double>(l));

    for (int j = 0; j < l; ++j) {
        for (int i = 0; i < radix - 1; ++i) {
            double temp = px * static_cast<double>(i + 1) * static_cast<double>(j);
            w[offset + i * l + j] = Complex(std::cos(temp), std::sin(temp));
        }
    }
}

// Initialize complete twiddle factor table for mixed-radix FFT
void settbl(std::vector<Complex>& w, int n) {
    int ip[3];
    factor(n, ip);

    int kp4, kp8;
    if (ip[0] != 1) {
        kp4 = 2 - (ip[0] + 2) % 3;
        kp8 = (ip[0] - kp4 * 2) / 3;
    } else {
        kp4 = 0;
        kp8 = 0;
    }

    // Calculate total twiddle table size
    int total = 0;
    int l = n;
    for (int k = 0; k < kp8; ++k) { l /= 8; total += l * 7; }
    for (int k = 0; k < ip[2]; ++k) { l /= 5; total += l * 4; }
    for (int k = 0; k < kp4; ++k) { l /= 4; total += l * 3; }
    for (int k = 0; k < ip[1]; ++k) { l /= 3; total += l * 2; }

    w.resize(total);

    int j = 0;
    l = n;
    for (int k = 0; k < kp8; ++k) {
        l /= 8;
        settbl0(w, j, 8, l);
        j += l * 7;
    }
    for (int k = 0; k < ip[2]; ++k) {
        l /= 5;
        settbl0(w, j, 5, l);
        j += l * 4;
    }
    for (int k = 0; k < kp4; ++k) {
        l /= 4;
        settbl0(w, j, 4, l);
        j += l * 3;
    }
    for (int k = 0; k < ip[1]; ++k) {
        l /= 3;
        settbl0(w, j, 3, l);
        j += l * 2;
    }
}

// Initialize 2-D twiddle factor table: w[i][j] = e^(-2*pi*i*(i)*(j) / (nx*ny))
void settbl2(std::vector<Complex>& w, int nx, int ny) {
    const double pi2 = 8.0 * std::atan(1.0);
    double px = -pi2 / (static_cast<double>(nx) * static_cast<double>(ny));

    w.resize(nx * ny);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double temp = px * static_cast<double>(i) * static_cast<double>(j);
            w[i * ny + j] = Complex(std::cos(temp), std::sin(temp));
        }
    }
}
