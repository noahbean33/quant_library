// Extended Split-Radix Decimation-in-Frequency FFT
//
// Reference:
//   D. Takahashi, "An extended split-radix FFT algorithm,"
//   IEEE Signal Processing Lett., vol. 8, pp. 145-147, May 2001.

#include <cmath>
#include <vector>
#include <algorithm>

static constexpr double C21   = 0.70710678118654752;
static constexpr double TWOPI = 6.28318530717958647;

// iv =  1 : forward FFT
// iv = -1 : inverse FFT
// n = 2^m
void esrfft(std::vector<double>& x, std::vector<double>& y,
            std::vector<int>& ibeta, int n, int m, int iv) {

    // --- Build beta vector table ---
    ibeta[0] = 1;
    ibeta[1] = 0;
    ibeta[2] = 0;
    ibeta[3] = 0;
    int id = 1;
    for (int k = 3; k <= m - 1; ++k) {
        for (int j = 1; j <= 4; ++j) {
            int is = (j + 3) * id;
            for (int i = is; i < is + id; ++i) {
                ibeta[i] = ibeta[i - is];
            }
        }
        id *= 2;
    }

    // --- Conjugate imaginary part for inverse FFT ---
    if (iv == -1) {
        for (int i = 0; i < n; ++i) {
            y[i] = -y[i];
        }
    }

    // --- L-shaped butterflies (radix-8 stages) ---
    int L  = 1;
    int n2 = n;
    for (int k = 1; k <= m - 2; ++k) {
        int n8 = n2 / 8;
        double e = -TWOPI / static_cast<double>(n2);
        double a = 0.0;

        for (int j = 0; j < n8; ++j) {
            double a3 = 3.0 * a;
            double a5 = 5.0 * a;
            double a7 = 7.0 * a;
            double cc1 = std::cos(a),  ss1 = std::sin(a);
            double cc3 = std::cos(a3), ss3 = std::sin(a3);
            double cc5 = std::cos(a5), ss5 = std::sin(a5);
            double cc7 = std::cos(a7), ss7 = std::sin(a7);
            a = static_cast<double>(j + 1) * e;

            for (int i = 0; i < L; ++i) {
                if (ibeta[i] == 1) {
                    int i0 = i * n2 + j;
                    int i1 = i0 + n8;
                    int i2 = i1 + n8;
                    int i3 = i2 + n8;
                    int i4 = i3 + n8;
                    int i5 = i4 + n8;
                    int i6 = i5 + n8;
                    int i7 = i6 + n8;

                    double x0 = x[i0] - x[i4];
                    double y0 = y[i0] - y[i4];
                    double x1 = x[i1] - x[i5];
                    double y1 = y[i1] - y[i5];
                    double x2 = y[i2] - y[i6];
                    double y2 = x[i6] - x[i2];
                    double x3 = x[i3] - x[i7];
                    double y3 = y[i3] - y[i7];

                    x[i0] += x[i4]; y[i0] += y[i4];
                    x[i1] += x[i5]; y[i1] += y[i5];
                    x[i2] += x[i6]; y[i2] += y[i6];
                    x[i3] += x[i7]; y[i3] += y[i7];

                    double u0 = x0 + C21 * (x1 - x3);
                    double v0 = y0 + C21 * (y1 - y3);
                    double u1 = x0 - C21 * (x1 - x3);
                    double v1 = y0 - C21 * (y1 - y3);
                    double u2 = x2 + C21 * (y1 + y3);
                    double v2 = y2 - C21 * (x1 + x3);
                    double u3 = x2 - C21 * (y1 + y3);
                    double v3 = y2 + C21 * (x1 + x3);

                    x[i4] = cc1 * (u0 + u2) - ss1 * (v0 + v2);
                    y[i4] = cc1 * (v0 + v2) + ss1 * (u0 + u2);
                    x[i5] = cc5 * (u1 + u3) - ss5 * (v1 + v3);
                    y[i5] = cc5 * (v1 + v3) + ss5 * (u1 + u3);
                    x[i6] = cc3 * (u1 - u3) - ss3 * (v1 - v3);
                    y[i6] = cc3 * (v1 - v3) + ss3 * (u1 - u3);
                    x[i7] = cc7 * (u0 - u2) - ss7 * (v0 - v2);
                    y[i7] = cc7 * (v0 - v2) + ss7 * (u0 - u2);
                }
            }
        }
        L  *= 2;
        n2 /= 2;
    }

    // --- Length-4 butterflies ---
    for (int i = 0; i < n / 4; ++i) {
        if (ibeta[i] == 1) {
            int i0 = 4 * i;
            int i1 = i0 + 1;
            int i2 = i1 + 1;
            int i3 = i2 + 1;

            double x0 = x[i0] - x[i2];
            double y0 = y[i0] - y[i2];
            double x1 = y[i1] - y[i3];
            double y1 = x[i3] - x[i1];

            x[i0] += x[i2]; y[i0] += y[i2];
            x[i1] += x[i3]; y[i1] += y[i3];
            x[i2] = x0 + x1; y[i2] = y0 + y1;
            x[i3] = x0 - x1; y[i3] = y0 - y1;
        }
    }

    // --- Length-2 butterflies ---
    for (int i = 0; i < n / 2; ++i) {
        if (ibeta[i] == 1) {
            int i0 = 2 * i;
            int i1 = i0 + 1;

            double x0 = x[i0] - x[i1];
            double y0 = y[i0] - y[i1];
            x[i0] += x[i1]; y[i0] += y[i1];
            x[i1] = x0;     y[i1] = y0;
        }
    }

    // --- Bit-reversal permutation ---
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(x[i], x[j]);
            std::swap(y[i], y[j]);
        }
        int half = n / 2;
        while (half <= j) {
            j -= half;
            half /= 2;
        }
        j += half;
    }

    // --- Normalize for inverse FFT ---
    if (iv == -1) {
        double inv_n = 1.0 / static_cast<double>(n);
        for (int i = 0; i < n; ++i) {
            x[i] *=  inv_n;
            y[i] *= -inv_n;
        }
    }
}
