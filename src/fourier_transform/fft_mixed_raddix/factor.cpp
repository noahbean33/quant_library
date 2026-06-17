// Factorization Routine
// Factors n into 2^ip[0] * 3^ip[1] * 5^ip[2] and finds balanced NX*NY split

#include <cmath>
#include <algorithm>

// Factor n into powers of 2, 3, and 5
// ip[0] = power of 2, ip[1] = power of 3, ip[2] = power of 5
void factor(int n, int ip[3]) {
    ip[0] = 0;
    ip[1] = 0;
    ip[2] = 0;

    int n2 = n;
    if (n % 2 != 0 && n % 3 != 0 && n % 5 != 0) return;

    while (n2 > 1) {
        if (n2 % 2 == 0) {
            ip[0]++;
            n2 /= 2;
        } else if (n2 % 3 == 0) {
            ip[1]++;
            n2 /= 3;
        } else if (n2 % 5 == 0) {
            ip[2]++;
            n2 /= 5;
        } else {
            break;
        }
    }
}

// Find NX and NY such that N = NX * NY, with NX closest to sqrt(N)
// Both NX and NY are products of powers of 2, 3, and 5
void getnxny(int n, int& nx, int& ny) {
    int ip[3], lnx[3], lny[3];

    int isqrtn = static_cast<int>(std::sqrt(static_cast<double>(n)));
    factor(n, ip);

    for (int i = 0; i < 3; ++i) {
        lnx[i] = 0;
    }

    int ires = isqrtn;

    for (int k = 0; k <= (ip[2] + 1) / 2; ++k) {
        for (int j = 0; j <= (ip[1] + 1) / 2; ++j) {
            for (int i = 0; i <= (ip[0] + 1) / 2; ++i) {
                int candidate = 1;
                for (int p = 0; p < i; ++p) candidate *= 2;
                for (int p = 0; p < j; ++p) candidate *= 3;
                for (int p = 0; p < k; ++p) candidate *= 5;

                if (candidate <= isqrtn) {
                    int ires2 = isqrtn - candidate;
                    if (ires2 < ires) {
                        lnx[0] = i;
                        lnx[1] = j;
                        lnx[2] = k;
                        ires = ires2;
                    }
                }
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        lny[i] = ip[i] - lnx[i];
    }

    nx = 1;
    for (int p = 0; p < lnx[0]; ++p) nx *= 2;
    for (int p = 0; p < lnx[1]; ++p) nx *= 3;
    for (int p = 0; p < lnx[2]; ++p) nx *= 5;

    ny = 1;
    for (int p = 0; p < lny[0]; ++p) ny *= 2;
    for (int p = 0; p < lny[1]; ++p) ny *= 3;
    for (int p = 0; p < lny[2]; ++p) ny *= 5;
}
