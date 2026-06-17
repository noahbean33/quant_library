// 2-D Complex FFT Routine (for Vector Machines)
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

// 2-D complex FFT for vector machines
// a: input/output array of nx*ny complex values (row-major: a[i*ny + j])
// nx, ny: transform lengths in x and y directions
// iopt: 0 = initialize, -1 = forward, +1 = inverse
void vzfft2d(std::vector<Complex>& a, int nx, int ny, int iopt,
             std::vector<Complex>& wx, std::vector<Complex>& wy) {
    // Initialize twiddle factor tables
    if (iopt == 0) {
        wx.resize(nx);
        wy.resize(ny);
        settbl(wx, nx);
        settbl(wy, ny);
        return;
    }

    // Conjugate for inverse transform
    if (iopt == 1) {
        for (int i = 0; i < nx * ny; ++i) {
            a[i] = std::conj(a[i]);
        }
    }

    int lnx[3], lny[3];
    factor(nx, lnx);
    factor(ny, lny);

    std::vector<Complex> b(nx * ny);

    // Step 1: nx multi-column ny-point FFTs
    mfft235a(a, b, wy, nx, ny, lny);

    // Step 2: Transpose (nx x ny -> ny x nx)
    ztrans(a, b, nx, ny);

    // Step 3: ny multi-column nx-point FFTs
    mfft235a(b, a, wx, ny, nx, lnx);

    // Step 4: Transpose (ny x nx -> nx x ny)
    ztrans(b, a, ny, nx);

    // Scale for inverse transform
    if (iopt == 1) {
        double dn = 1.0 / (static_cast<double>(nx) * static_cast<double>(ny));
        for (int i = 0; i < nx * ny; ++i) {
            a[i] = std::conj(a[i]) * dn;
        }
    }
}
