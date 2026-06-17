// Radix-2, 3, 4, 5 and 8 Multiple FFT Routine
// Multi-column FFT and matrix transposition routines

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

// Forward declarations for FFT kernel routines
void fft2(std::vector<Complex>& a, std::vector<Complex>& b, int m);
void fft3b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l);
void fft4b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l);
void fft5b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l);
void fft8b(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int m, int l);
void fft3a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l);
void fft4a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l);
void fft5a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l);
void fft8a(std::vector<Complex>& a, std::vector<Complex>& b,
           const std::vector<Complex>& w, int l);
void factor(int n, int ip[3]);

// Helper: dispatch radix-3 (A or B variant based on stride)
static void fft3_dispatch(std::vector<Complex>& a, std::vector<Complex>& b,
                          const std::vector<Complex>& w, int m, int l) {
    if (m == 1) fft3a(a, b, w, l);
    else        fft3b(a, b, w, m, l);
}
static void fft4_dispatch(std::vector<Complex>& a, std::vector<Complex>& b,
                          const std::vector<Complex>& w, int m, int l) {
    if (m == 1) fft4a(a, b, w, l);
    else        fft4b(a, b, w, m, l);
}
static void fft5_dispatch(std::vector<Complex>& a, std::vector<Complex>& b,
                          const std::vector<Complex>& w, int m, int l) {
    if (m == 1) fft5a(a, b, w, l);
    else        fft5b(a, b, w, m, l);
}
static void fft8_dispatch(std::vector<Complex>& a, std::vector<Complex>& b,
                          const std::vector<Complex>& w, int m, int l) {
    if (m == 1) fft8a(a, b, w, l);
    else        fft8b(a, b, w, m, l);
}

// Multiple FFT variant A: ns simultaneous n-point FFTs
// Uses FFTrB kernels with stride ns*m
// Data layout: a[ns * n], each of ns columns is transformed
void mfft235a(std::vector<Complex>& a, std::vector<Complex>& b,
              const std::vector<Complex>& w, int ns, int n, const int ip[3]) {
    int kp4, kp8;
    if (ip[0] != 1) {
        kp4 = 2 - (ip[0] + 2) % 3;
        kp8 = (ip[0] - kp4 * 2) / 3;
    } else {
        kp4 = 0;
        kp8 = 0;
    }

    int key = 1;
    int j = 0;
    int l = n;
    int m = 1;

    for (int k = 0; k < kp8; ++k) {
        l /= 8;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 7);
        if (l >= 2) {
            if (key >= 0) fft8b(a, b, wslice, ns * m, l);
            else          fft8b(b, a, wslice, ns * m, l);
            key = -key;
        } else {
            if (key >= 0) fft8b(a, a, wslice, ns * m, l);
            else          fft8b(b, a, wslice, ns * m, l);
        }
        m *= 8;
        j += l * 7;
    }
    for (int k = 0; k < ip[2]; ++k) {
        l /= 5;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 4);
        if (l >= 2) {
            if (key >= 0) fft5b(a, b, wslice, ns * m, l);
            else          fft5b(b, a, wslice, ns * m, l);
            key = -key;
        } else {
            if (key >= 0) fft5b(a, a, wslice, ns * m, l);
            else          fft5b(b, a, wslice, ns * m, l);
        }
        m *= 5;
        j += l * 4;
    }
    for (int k = 0; k < kp4; ++k) {
        l /= 4;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 3);
        if (l >= 2) {
            if (key >= 0) fft4b(a, b, wslice, ns * m, l);
            else          fft4b(b, a, wslice, ns * m, l);
            key = -key;
        } else {
            if (key >= 0) fft4b(a, a, wslice, ns * m, l);
            else          fft4b(b, a, wslice, ns * m, l);
        }
        m *= 4;
        j += l * 3;
    }
    for (int k = 0; k < ip[1]; ++k) {
        l /= 3;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 2);
        if (l >= 2) {
            if (key >= 0) fft3b(a, b, wslice, ns * m, l);
            else          fft3b(b, a, wslice, ns * m, l);
            key = -key;
        } else {
            if (key >= 0) fft3b(a, a, wslice, ns * m, l);
            else          fft3b(b, a, wslice, ns * m, l);
        }
        m *= 3;
        j += l * 2;
    }
    if (ip[0] == 1) {
        if (key >= 0) fft2(a, a, ns * m);
        else          fft2(b, a, ns * m);
    }
}

// Multiple FFT variant B: ns simultaneous n-point FFTs
// Uses mixed A/B dispatch with stride ns*m
// Output is written to b (result ends in b, not a)
void mfft235b(std::vector<Complex>& a, std::vector<Complex>& b,
              const std::vector<Complex>& w, int ns, int n, const int ip[3]) {
    int kp4, kp8;
    if (ip[0] != 1) {
        kp4 = 2 - (ip[0] + 2) % 3;
        kp8 = (ip[0] - kp4) / 3;
    } else {
        kp4 = 0;
        kp8 = 0;
    }

    int key = 1;
    int j = 0;
    int l = n;
    int m = 1;

    for (int k = 0; k < kp8; ++k) {
        l /= 8;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 7);
        if (l >= 2) {
            if (key >= 0) fft8_dispatch(a, b, wslice, ns * m, l);
            else          fft8_dispatch(b, a, wslice, ns * m, l);
            key = -key;
        } else {
            if (key >= 0) fft8_dispatch(a, b, wslice, ns * m, l);
            else          fft8_dispatch(b, b, wslice, ns * m, l);
        }
        m *= 8;
        j += l * 7;
    }
    for (int k = 0; k < ip[2]; ++k) {
        l /= 5;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 4);
        if (l >= 2) {
            if (key >= 0) fft5_dispatch(a, b, wslice, ns * m, l);
            else          fft5_dispatch(b, a, wslice, ns * m, l);
            key = -key;
        } else {
            if (key >= 0) fft5_dispatch(a, b, wslice, ns * m, l);
            else          fft5_dispatch(b, b, wslice, ns * m, l);
        }
        m *= 5;
        j += l * 4;
    }
    for (int k = 0; k < kp4; ++k) {
        l /= 4;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 3);
        if (l >= 2) {
            if (key >= 0) fft4_dispatch(a, b, wslice, ns * m, l);
            else          fft4_dispatch(b, a, wslice, ns * m, l);
            key = -key;
        } else {
            if (key >= 0) fft4_dispatch(a, b, wslice, ns * m, l);
            else          fft4_dispatch(b, b, wslice, ns * m, l);
        }
        m *= 4;
        j += l * 3;
    }
    for (int k = 0; k < ip[1]; ++k) {
        l /= 3;
        std::vector<Complex> wslice(w.begin() + j, w.begin() + j + l * 2);
        if (l >= 2) {
            if (key >= 0) fft3_dispatch(a, b, wslice, ns * m, l);
            else          fft3_dispatch(b, a, wslice, ns * m, l);
            key = -key;
        } else {
            if (key >= 0) fft3_dispatch(a, b, wslice, ns * m, l);
            else          fft3_dispatch(b, b, wslice, ns * m, l);
        }
        m *= 3;
        j += l * 2;
    }
    if (ip[0] == 1) {
        if (key >= 0) fft2(a, b, ns * m);
        else          fft2(b, b, ns * m);
    }
}

// ============================================================
// Matrix transposition routines
// ============================================================

// Simple transpose (nx x ny -> ny x nx)
static void ztransa(const std::vector<Complex>& a, std::vector<Complex>& b,
                    int nx, int ny) {
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            b[j * nx + i] = a[i * ny + j];
        }
    }
}

// Cache-friendly transpose using diagonal traversal
static void ztransb(const std::vector<Complex>& a, std::vector<Complex>& b,
                    int nx, int ny) {
    if (ny >= nx) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx - i; ++j) {
                b[j * nx + (i + j)] = a[(i + j) * ny + j];
            }
        }
        for (int i = 1; i <= ny - nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                b[(i + j) * nx + j] = a[j * ny + (i + j)];
            }
        }
        for (int i = ny - nx + 1; i < ny; ++i) {
            for (int j = 0; j < ny - i; ++j) {
                b[(i + j) * nx + j] = a[j * ny + (i + j)];
            }
        }
    } else {
        for (int i = 0; i < ny; ++i) {
            for (int j = 0; j < ny - i; ++j) {
                b[(i + j) * nx + j] = a[j * ny + (i + j)];
            }
        }
        for (int i = 1; i <= nx - ny; ++i) {
            for (int j = 0; j < ny; ++j) {
                b[j * nx + (i + j)] = a[(i + j) * ny + j];
            }
        }
        for (int i = nx - ny + 1; i < nx; ++i) {
            for (int j = 0; j < nx - i; ++j) {
                b[j * nx + (i + j)] = a[(i + j) * ny + j];
            }
        }
    }
}

// Transpose: selects simple or diagonal variant based on factorization
void ztrans(const std::vector<Complex>& a, std::vector<Complex>& b,
            int nx, int ny) {
    int lnx[3], lny[3];
    factor(nx, lnx);
    factor(ny, lny);

    if (nx == 1 || ny == 1) {
        for (int i = 0; i < nx * ny; ++i) {
            b[i] = a[i];
        }
        return;
    }

    if (lnx[0] + lny[0] <= 1) {
        ztransa(a, b, nx, ny);
    } else {
        ztransb(a, b, nx, ny);
    }
}

// Batched 2-D transpose: transpose each z-slab independently
void ztrans2(const std::vector<Complex>& a, std::vector<Complex>& b,
             int nx, int ny, int nz) {
    for (int k = 0; k < nz; ++k) {
        std::vector<Complex> a_slab(a.begin() + k * nx * ny,
                                     a.begin() + (k + 1) * nx * ny);
        std::vector<Complex> b_slab(ny * nx);
        ztrans(a_slab, b_slab, nx, ny);
        std::copy(b_slab.begin(), b_slab.end(), b.begin() + k * ny * nx);
    }
}

// Multi-stride transpose: transpose with leading dimension ns
void mztrans(const std::vector<Complex>& a, std::vector<Complex>& b,
             int ns, int nx, int ny) {
    if (ns == 1) {
        ztrans(a, b, nx, ny);
    } else {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < ns; ++k) {
                    b[k + j * ns + i * ns * ny] = a[k + i * ns + j * ns * nx];
                }
            }
        }
    }
}
