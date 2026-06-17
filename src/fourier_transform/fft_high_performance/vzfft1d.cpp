// 1-D Complex FFT Routine (for Vector Machines)
// Based on the six-step FFT approach with twiddle factor multiplication

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

// Forward declarations for external routines
void factor(int n, int ip[3]);
void settbl(std::vector<Complex>& w, int n);
void settbl2(std::vector<Complex>& w, int nx, int ny);
void getnxny(int n, int& nx, int& ny);
void mfft235a(std::vector<Complex>& a, std::vector<Complex>& b,
              const std::vector<Complex>& w, int ns, int n, const int ip[3]);
void mfft235b(std::vector<Complex>& a, std::vector<Complex>& b,
              const std::vector<Complex>& w, int ns, int n, const int ip[3]);

// Simple transpose with twiddle factor multiplication (nx x ny -> ny x nx)
void ztransmula(const std::vector<Complex>& a, std::vector<Complex>& b,
                const std::vector<Complex>& w, int nx, int ny) {
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            b[j * nx + i] = a[i * ny + j] * w[i * ny + j];
        }
    }
}

// Cache-friendly transpose with twiddle factor multiplication using diagonal traversal
void ztransmulb(const std::vector<Complex>& a, std::vector<Complex>& b,
                const std::vector<Complex>& w, int nx, int ny) {
    if (ny >= nx) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx - i; ++j) {
                b[j * nx + (i + j)] = a[(i + j) * ny + j] * w[(i + j) * ny + j];
            }
        }
        for (int i = 1; i <= ny - nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                b[(i + j) * nx + j] = a[j * ny + (i + j)] * w[j * ny + (i + j)];
            }
        }
        for (int i = ny - nx + 1; i < ny; ++i) {
            for (int j = 0; j < ny - i; ++j) {
                b[(i + j) * nx + j] = a[j * ny + (i + j)] * w[j * ny + (i + j)];
            }
        }
    } else {
        for (int i = 0; i < ny; ++i) {
            for (int j = 0; j < ny - i; ++j) {
                b[(i + j) * nx + j] = a[j * ny + (i + j)] * w[j * ny + (i + j)];
            }
        }
        for (int i = 1; i <= nx - ny; ++i) {
            for (int j = 0; j < ny; ++j) {
                b[j * nx + (i + j)] = a[(i + j) * ny + j] * w[(i + j) * ny + j];
            }
        }
        for (int i = nx - ny + 1; i < nx; ++i) {
            for (int j = 0; j < nx - i; ++j) {
                b[j * nx + (i + j)] = a[(i + j) * ny + j] * w[(i + j) * ny + j];
            }
        }
    }
}

// Transpose and multiply by twiddle factors
void ztransmul(const std::vector<Complex>& a, std::vector<Complex>& b,
               const std::vector<Complex>& w, int nx, int ny) {
    int lnx[3], lny[3];
    factor(nx, lnx);
    factor(ny, lny);

    if (nx == 1 || ny == 1) {
        for (int i = 0; i < nx * ny; ++i) {
            b[i] = a[i] * w[i];
        }
        return;
    }

    if (lnx[0] + lny[0] <= 1) {
        ztransmula(a, b, w, nx, ny);
    } else {
        ztransmulb(a, b, w, nx, ny);
    }
}

// 1-D complex FFT for vector machines
// a: input/output array of n complex values
// b: work/coefficient array of size 2*n
// n = (2^ip) * (3^iq) * (5^ir)
// iopt: 0 = initialize, -1 = forward, +1 = inverse
void vzfft1d(std::vector<Complex>& a, int n, int iopt,
             std::vector<Complex>& b,
             std::vector<Complex>& wx, std::vector<Complex>& wy) {
    // Conjugate for inverse transform
    if (iopt == 1) {
        for (int i = 0; i < n; ++i) {
            a[i] = std::conj(a[i]);
        }
    }

    int nx, ny;
    getnxny(n, nx, ny);

    // Initialize twiddle factor tables
    if (iopt == 0) {
        wx.resize(nx);
        wy.resize(ny);
        settbl(wx, nx);
        settbl(wy, ny);
        b.resize(2 * n);
        settbl2(b, nx, ny);   // Store twiddle factors in b[n..2n-1]
        return;
    }

    int lnx[3], lny[3];
    factor(nx, lnx);
    factor(ny, lny);

    // Six-step FFT:
    // Step 1-2: Multi-column NY-point FFTs along columns of (NX x NY) matrix
    mfft235a(a, b, wy, nx, ny, lny);

    // Step 3-4: Transpose with twiddle factor multiplication
    std::vector<Complex> tw(b.begin() + n, b.begin() + 2 * n);
    ztransmul(a, b, tw, nx, ny);

    // Step 5-6: Multi-column NX-point FFTs along columns of (NY x NX) matrix
    mfft235b(b, a, wx, ny, nx, lnx);

    // Scale for inverse transform
    if (iopt == 1) {
        double dn = 1.0 / static_cast<double>(n);
        for (int i = 0; i < n; ++i) {
            a[i] = std::conj(a[i]) * dn;
        }
    }
}
