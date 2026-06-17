// 3-D Complex FFT Routine (for Vector Machines)
// Row-column approach with transpositions for cache efficiency

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

// Forward declarations for external routines
void factor(int n, int ip[3]);
void settbl(std::vector<Complex>& w, int n);
void ztrans(const std::vector<Complex>& a, std::vector<Complex>& b, int nx, int ny);
void mfft235a(std::vector<Complex>& a, std::vector<Complex>& b,
              const std::vector<Complex>& w, int ns, int n, const int ip[3]);
void mfft235b(std::vector<Complex>& a, std::vector<Complex>& b,
              const std::vector<Complex>& w, int ns, int n, const int ip[3]);

// 3-D complex FFT for vector machines
// a: input/output array of nx*ny*nz complex values
// nx, ny, nz: transform lengths in x, y, z directions
// iopt: 0 = initialize, -1 = forward, +1 = inverse
void vzfft3d(std::vector<Complex>& a, int nx, int ny, int nz, int iopt,
             std::vector<Complex>& wx, std::vector<Complex>& wy,
             std::vector<Complex>& wz) {
    // Initialize twiddle factor tables
    if (iopt == 0) {
        wx.resize(nx);
        wy.resize(ny);
        wz.resize(nz);
        settbl(wx, nx);
        settbl(wy, ny);
        settbl(wz, nz);
        return;
    }

    int ntot = nx * ny * nz;

    // Conjugate for inverse transform
    if (iopt == 1) {
        for (int i = 0; i < ntot; ++i) {
            a[i] = std::conj(a[i]);
        }
    }

    int lnx[3], lny[3], lnz[3];
    factor(nx, lnx);
    factor(ny, lny);
    factor(nz, lnz);

    std::vector<Complex> b(ntot);

    // Step 1: (nx*ny) multi-column nz-point FFTs (along z-dimension)
    mfft235a(a, b, wz, nx * ny, nz, lnz);

    // Step 2: Transpose (nx*ny x nz -> nz x nx*ny)
    ztrans(a, b, nx * ny, nz);

    // Step 3: (nz*nx) multi-column ny-point FFTs (along y-dimension)
    mfft235a(b, a, wy, nz * nx, ny, lny);

    // Step 4: Transpose (nz*nx x ny -> ny x nz*nx)
    ztrans(b, a, nz * nx, ny);

    // Step 5: (ny*nz) multi-column nx-point FFTs (along x-dimension)
    mfft235b(a, b, wx, ny * nz, nx, lnx);

    // Step 6: Transpose (ny*nz x nx -> nx x ny*nz)
    ztrans(b, a, ny * nz, nx);

    // Scale for inverse transform
    if (iopt == 1) {
        double dn = 1.0 / (static_cast<double>(nx) * static_cast<double>(ny)
                            * static_cast<double>(nz));
        for (int i = 0; i < ntot; ++i) {
            a[i] = std::conj(a[i]) * dn;
        }
    }
}
