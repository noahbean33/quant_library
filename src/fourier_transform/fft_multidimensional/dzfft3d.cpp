// 3-D Real-to-Complex FFT Routine (with OpenMP and Cache Blocking)
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

// Inner blocked 3-D real-to-complex FFT computation
void dzfft3d_inner(const std::vector<double>& da, std::vector<Complex>& a,
                   std::vector<Complex>& b, int nx, int ny, int nz,
                   const std::vector<Complex>& wx, const std::vector<Complex>& wy,
                   const std::vector<Complex>& wz,
                   const int lnx[3], const int lny[3], const int lnz[3]) {
    int nxh = nx / 2 + 1;
    std::vector<Complex> cx(nx);
    std::vector<Complex> cy(ny + NP);
    std::vector<Complex> cz(nz + NP);
    std::vector<Complex> d(std::max({nx, ny, nz}));

    // Phase 1: Packed real-to-complex x-direction FFTs for each z-slab
    for (int k = 0; k < nz; ++k) {
        if (ny % 2 == 0) {
            for (int j = 0; j < ny; j += 2) {
                for (int i = 0; i < nx; ++i) {
                    cx[i] = Complex(da[i * ny * nz + j * nz + k],
                                    da[i * ny * nz + (j + 1) * nz + k]);
                }
                fft235(cx, d, wx, nx, lnx);

                b[0 * ny * nz + j * nz + k]       = Complex(cx[0].real(), 0.0);
                b[0 * ny * nz + (j + 1) * nz + k] = Complex(cx[0].imag(), 0.0);
                for (int i = 1; i < nxh; ++i) {
                    b[i * ny * nz + j * nz + k] =
                        0.5 * (cx[i] + std::conj(cx[nx - i]));
                    b[i * ny * nz + (j + 1) * nz + k] =
                        Complex(0.0, -0.5) * (cx[i] - std::conj(cx[nx - i]));
                }
            }
        } else {
            for (int j = 0; j < ny - 1; j += 2) {
                for (int i = 0; i < nx; ++i) {
                    cx[i] = Complex(da[i * ny * nz + j * nz + k],
                                    da[i * ny * nz + (j + 1) * nz + k]);
                }
                fft235(cx, d, wx, nx, lnx);

                b[0 * ny * nz + j * nz + k]       = Complex(cx[0].real(), 0.0);
                b[0 * ny * nz + (j + 1) * nz + k] = Complex(cx[0].imag(), 0.0);
                for (int i = 1; i < nxh; ++i) {
                    b[i * ny * nz + j * nz + k] =
                        0.5 * (cx[i] + std::conj(cx[nx - i]));
                    b[i * ny * nz + (j + 1) * nz + k] =
                        Complex(0.0, -0.5) * (cx[i] - std::conj(cx[nx - i]));
                }
            }
            // Handle last odd row
            for (int i = 0; i < nx; ++i) {
                cx[i] = Complex(da[i * ny * nz + (ny - 1) * nz + k], 0.0);
            }
            fft235(cx, d, wx, nx, lnx);
            for (int i = 0; i < nxh; ++i) {
                b[i * ny * nz + (ny - 1) * nz + k] = cx[i];
            }
        }

        // Phase 2: Blocked ny-point FFTs on each frequency bin for this z-slab
        for (int ii = 0; ii < nxh; ii += NBLK) {
            int iend = std::min(ii + NBLK, nxh);
            for (int i = ii; i < iend; ++i) {
                for (int j = 0; j < ny; ++j) {
                    cy[j] = b[i * ny * nz + j * nz + k];
                }
                fft235(cy, d, wy, ny, lny);
                for (int j = 0; j < ny; ++j) {
                    b[i * ny * nz + j * nz + k] = cy[j];
                }
            }
        }
    }

    // Phase 3: Blocked nz-point FFTs along z-dimension
    for (int j = 0; j < ny; ++j) {
        for (int ii = 0; ii < nxh; ii += NBLK) {
            int iend = std::min(ii + NBLK, nxh);
            for (int i = ii; i < iend; ++i) {
                for (int k = 0; k < nz; ++k) {
                    cz[k] = b[i * ny * nz + j * nz + k];
                }
                fft235(cz, d, wz, nz, lnz);
                for (int k = 0; k < nz; ++k) {
                    a[i * ny * nz + j * nz + k] = cz[k];
                }
            }
        }
    }
}

// 3-D real-to-complex FFT
// da: real input array of nx*ny*nz values
// a: complex output array of (nx/2+1)*ny*nz values
// nx, ny, nz: transform lengths
// iopt: 0 = initialize, -1 = forward transform
void dzfft3d(const std::vector<double>& da, std::vector<Complex>& a,
             int nx, int ny, int nz, int iopt,
             std::vector<Complex>& wx, std::vector<Complex>& wy,
             std::vector<Complex>& wz) {
    if (iopt == 0) {
        wx.resize(nx);
        wy.resize(ny);
        wz.resize(nz);
        settbl(wx, nx);
        settbl(wy, ny);
        settbl(wz, nz);
        return;
    }

    int lnx[3], lny[3], lnz[3];
    factor(nx, lnx);
    factor(ny, lny);
    factor(nz, lnz);

    int nxh = nx / 2 + 1;
    std::vector<Complex> b(nxh * ny * nz);

    dzfft3d_inner(da, a, b, nx, ny, nz, wx, wy, wz, lnx, lny, lnz);
}
