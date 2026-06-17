// 2-D Complex-to-Real FFT Routine (with OpenMP and Cache Blocking)
// Inverse of DZFFT2D: converts half-complex input to real output

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

// Forward declarations for external routines
void factor(int n, int ip[3]);
void settbl(std::vector<Complex>& w, int n);
void fft235(std::vector<Complex>& a, std::vector<Complex>& b,
            const std::vector<Complex>& w, int n, const int ip[3]);

static const int NBLK = 16;
static const int NP = 8;

// Inner blocked 2-D complex-to-real FFT computation
void zdfft2d_inner(std::vector<Complex>& a, std::vector<double>& da,
                   std::vector<Complex>& b, int nx, int ny,
                   const std::vector<Complex>& wx, const std::vector<Complex>& wy,
                   const int lnx[3], const int lny[3]) {
    int nxh = nx / 2 + 1;
    double dn = 1.0 / (static_cast<double>(nx) * static_cast<double>(ny));
    std::vector<Complex> cx(nx);
    std::vector<Complex> cy(ny + NP);
    std::vector<Complex> d(std::max(nx, ny));

    // Phase 1: Blocked ny-point inverse FFTs on each frequency bin
    for (int ii = 0; ii < nxh; ii += NBLK) {
        int iend = std::min(ii + NBLK, nxh);

        for (int i = ii; i < iend; ++i) {
            for (int j = 0; j < ny; ++j) {
                cy[j] = std::conj(a[i * ny + j]);
            }
            fft235(cy, d, wy, ny, lny);
            for (int j = 0; j < ny; ++j) {
                b[i * ny + j] = cy[j];
            }
        }
    }

    // Phase 2: Reconstruct full complex spectrum and inverse x-direction FFTs
    if (ny % 2 == 0) {
        for (int j = 0; j < ny; j += 2) {
            // Reconstruct full-length complex vector from half-complex
            cx[0] = Complex(b[0 * ny + j].real(), b[0 * ny + j + 1].real());
            for (int i = 1; i < nxh; ++i) {
                Complex temp = Complex(0.0, 1.0) * b[i * ny + j + 1];
                cx[i] = b[i * ny + j] + temp;
                cx[nx - i] = std::conj(b[i * ny + j] - temp);
            }
            fft235(cx, d, wx, nx, lnx);
            for (int i = 0; i < nx; ++i) {
                da[i * ny + j]     = cx[i].real() * dn;
                da[i * ny + j + 1] = cx[i].imag() * dn;
            }
        }
    } else {
        for (int j = 0; j < ny - 1; j += 2) {
            cx[0] = Complex(b[0 * ny + j].real(), b[0 * ny + j + 1].real());
            for (int i = 1; i < nxh; ++i) {
                Complex temp = Complex(0.0, 1.0) * b[i * ny + j + 1];
                cx[i] = b[i * ny + j] + temp;
                cx[nx - i] = std::conj(b[i * ny + j] - temp);
            }
            fft235(cx, d, wx, nx, lnx);
            for (int i = 0; i < nx; ++i) {
                da[i * ny + j]     = cx[i].real() * dn;
                da[i * ny + j + 1] = cx[i].imag() * dn;
            }
        }
        // Handle last odd row
        cx[0] = Complex(b[0 * ny + ny - 1].real(), 0.0);
        for (int i = 1; i < nxh; ++i) {
            cx[i] = b[i * ny + ny - 1];
            cx[nx - i] = std::conj(b[i * ny + ny - 1]);
        }
        fft235(cx, d, wx, nx, lnx);
        for (int i = 0; i < nx; ++i) {
            da[i * ny + ny - 1] = cx[i].real() * dn;
        }
    }
}

// 2-D complex-to-real FFT
// a: complex input array of (nx/2+1)*ny values (half-complex)
// da: real output array of nx*ny values
// nx, ny: transform lengths
// iopt: 0 = initialize, +1 = inverse transform
void zdfft2d(std::vector<Complex>& a, std::vector<double>& da,
             int nx, int ny, int iopt,
             std::vector<Complex>& wx, std::vector<Complex>& wy) {
    if (iopt == 0) {
        wx.resize(nx);
        wy.resize(ny);
        settbl(wx, nx);
        settbl(wy, ny);
        return;
    }

    int lnx[3], lny[3];
    factor(nx, lnx);
    factor(ny, lny);

    int nxh = nx / 2 + 1;
    std::vector<Complex> b(nxh * ny);

    zdfft2d_inner(a, da, b, nx, ny, wx, wy, lnx, lny);
}
