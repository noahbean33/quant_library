// 3-D Complex-to-Real FFT Routine (with OpenMP and Cache Blocking)
// Inverse of DZFFT3D: converts half-complex input to real output

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

// Inner blocked 3-D complex-to-real FFT computation
void zdfft3d_inner(std::vector<Complex>& a, std::vector<double>& da,
                   std::vector<Complex>& b, int nx, int ny, int nz,
                   const std::vector<Complex>& wx, const std::vector<Complex>& wy,
                   const std::vector<Complex>& wz,
                   const int lnx[3], const int lny[3], const int lnz[3]) {
    int nxh = nx / 2 + 1;
    double dn = 1.0 / (static_cast<double>(nx) * static_cast<double>(ny)
                        * static_cast<double>(nz));
    std::vector<Complex> cx(nx);
    std::vector<Complex> cy(ny + NP);
    std::vector<Complex> cz(nz + NP);
    std::vector<Complex> d(std::max({nx, ny, nz}));

    // Phase 1: Blocked nz-point inverse FFTs along z-dimension
    for (int j = 0; j < ny; ++j) {
        for (int ii = 0; ii < nxh; ii += NBLK) {
            int iend = std::min(ii + NBLK, nxh);
            for (int i = ii; i < iend; ++i) {
                for (int k = 0; k < nz; ++k) {
                    cz[k] = std::conj(a[i * ny * nz + j * nz + k]);
                }
                fft235(cz, d, wz, nz, lnz);
                for (int k = 0; k < nz; ++k) {
                    b[i * ny * nz + j * nz + k] = cz[k];
                }
            }
        }
    }

    // Phase 2-3: ny-point inverse FFTs + reconstruct + x-direction inverse FFTs
    if (ny % 2 == 0) {
        for (int k = 0; k < nz; ++k) {
            // Phase 2: Blocked ny-point FFTs on each frequency bin
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

            // Phase 3: Reconstruct full spectrum + inverse x-direction FFTs
            for (int j = 0; j < ny; j += 2) {
                cx[0] = Complex(b[0 * ny * nz + j * nz + k].real(),
                                b[0 * ny * nz + (j + 1) * nz + k].real());
                for (int i = 1; i < nxh; ++i) {
                    Complex temp = Complex(0.0, 1.0) * b[i * ny * nz + (j + 1) * nz + k];
                    cx[i] = b[i * ny * nz + j * nz + k] + temp;
                    cx[nx - i] = std::conj(b[i * ny * nz + j * nz + k] - temp);
                }
                fft235(cx, d, wx, nx, lnx);
                for (int i = 0; i < nx; ++i) {
                    da[i * ny * nz + j * nz + k]       = cx[i].real() * dn;
                    da[i * ny * nz + (j + 1) * nz + k] = cx[i].imag() * dn;
                }
            }
        }
    } else {
        for (int k = 0; k < nz; ++k) {
            // Phase 2: Blocked ny-point FFTs
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

            // Phase 3: Reconstruct + inverse x-direction FFTs (paired rows)
            for (int j = 0; j < ny - 1; j += 2) {
                cx[0] = Complex(b[0 * ny * nz + j * nz + k].real(),
                                b[0 * ny * nz + (j + 1) * nz + k].real());
                for (int i = 1; i < nxh; ++i) {
                    Complex temp = Complex(0.0, 1.0) * b[i * ny * nz + (j + 1) * nz + k];
                    cx[i] = b[i * ny * nz + j * nz + k] + temp;
                    cx[nx - i] = std::conj(b[i * ny * nz + j * nz + k] - temp);
                }
                fft235(cx, d, wx, nx, lnx);
                for (int i = 0; i < nx; ++i) {
                    da[i * ny * nz + j * nz + k]       = cx[i].real() * dn;
                    da[i * ny * nz + (j + 1) * nz + k] = cx[i].imag() * dn;
                }
            }
            // Handle last odd row
            cx[0] = Complex(b[0 * ny * nz + (ny - 1) * nz + k].real(), 0.0);
            for (int i = 1; i < nxh; ++i) {
                cx[i] = b[i * ny * nz + (ny - 1) * nz + k];
                cx[nx - i] = std::conj(b[i * ny * nz + (ny - 1) * nz + k]);
            }
            fft235(cx, d, wx, nx, lnx);
            for (int i = 0; i < nx; ++i) {
                da[i * ny * nz + (ny - 1) * nz + k] = cx[i].real() * dn;
            }
        }
    }
}

// 3-D complex-to-real FFT
// a: complex input array of (nx/2+1)*ny*nz values (half-complex)
// da: real output array of nx*ny*nz values
// nx, ny, nz: transform lengths
// iopt: 0 = initialize, +1 = inverse transform
void zdfft3d(std::vector<Complex>& a, std::vector<double>& da,
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

    zdfft3d_inner(a, da, b, nx, ny, nz, wx, wy, wz, lnx, lny, lnz);
}
