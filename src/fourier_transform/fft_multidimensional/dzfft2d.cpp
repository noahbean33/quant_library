// 2-D Real-to-Complex FFT Routine (with OpenMP and Cache Blocking)
// Converts real input to half-complex output using conjugate symmetry

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

// Inner blocked 2-D real-to-complex FFT computation
void dzfft2d_inner(const std::vector<double>& da, std::vector<Complex>& a,
                   std::vector<Complex>& b, int nx, int ny,
                   const std::vector<Complex>& wx, const std::vector<Complex>& wy,
                   const int lnx[3], const int lny[3]) {
    int nxh = nx / 2 + 1;
    std::vector<Complex> cx(nx);
    std::vector<Complex> cy(ny + NP);
    std::vector<Complex> d(std::max(nx, ny));

    // Phase 1: Pack pairs of real rows into complex, perform nx-point FFTs,
    // then extract half-complex spectra
    if (ny % 2 == 0) {
        for (int j = 0; j < ny; j += 2) {
            // Pack two real rows into one complex vector
            for (int i = 0; i < nx; ++i) {
                cx[i] = Complex(da[i * ny + j], da[i * ny + j + 1]);
            }
            fft235(cx, d, wx, nx, lnx);

            // Extract half-complex output using conjugate symmetry
            b[0 * ny + j]     = Complex(cx[0].real(), 0.0);
            b[0 * ny + j + 1] = Complex(cx[0].imag(), 0.0);
            for (int i = 1; i < nxh; ++i) {
                b[i * ny + j]     = 0.5 * (cx[i] + std::conj(cx[nx - i]));
                b[i * ny + j + 1] = Complex(0.0, -0.5) * (cx[i] - std::conj(cx[nx - i]));
            }
        }
    } else {
        for (int j = 0; j < ny - 1; j += 2) {
            for (int i = 0; i < nx; ++i) {
                cx[i] = Complex(da[i * ny + j], da[i * ny + j + 1]);
            }
            fft235(cx, d, wx, nx, lnx);

            b[0 * ny + j]     = Complex(cx[0].real(), 0.0);
            b[0 * ny + j + 1] = Complex(cx[0].imag(), 0.0);
            for (int i = 1; i < nxh; ++i) {
                b[i * ny + j]     = 0.5 * (cx[i] + std::conj(cx[nx - i]));
                b[i * ny + j + 1] = Complex(0.0, -0.5) * (cx[i] - std::conj(cx[nx - i]));
            }
        }
        // Handle last odd row
        for (int i = 0; i < nx; ++i) {
            cx[i] = Complex(da[i * ny + ny - 1], 0.0);
        }
        fft235(cx, d, wx, nx, lnx);
        for (int i = 0; i < nxh; ++i) {
            b[i * ny + ny - 1] = cx[i];
        }
    }

    // Phase 2: Blocked ny-point FFTs on each frequency bin
    for (int ii = 0; ii < nxh; ii += NBLK) {
        int iend = std::min(ii + NBLK, nxh);

        for (int i = ii; i < iend; ++i) {
            for (int j = 0; j < ny; ++j) {
                cy[j] = b[i * ny + j];
            }
            fft235(cy, d, wy, ny, lny);
            for (int j = 0; j < ny; ++j) {
                a[i * ny + j] = cy[j];
            }
        }
    }
}

// 2-D real-to-complex FFT
// da: real input array of nx*ny values
// a: complex output array of (nx/2+1)*ny values
// nx, ny: transform lengths
// iopt: 0 = initialize, -1 = forward transform
void dzfft2d(const std::vector<double>& da, std::vector<Complex>& a,
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

    dzfft2d_inner(da, a, b, nx, ny, wx, wy, lnx, lny);
}
