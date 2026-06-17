// Parallel factorization routines (C++ equivalent of pfactor.f)
//
// Original Fortran: FFTE package by Daisuke Takahashi, University of Tsukuba.

#include <cmath>
#include <array>
#include <algorithm>

// Factor N into 2^ip[0] * 3^ip[1] * 5^ip[2]
void factor8(long long N, std::array<int,3>& ip) {
    ip = {0, 0, 0};
    long long n2 = N;
    if (n2 % 2 != 0 && n2 % 3 != 0 && n2 % 5 != 0) return;
    while (n2 > 1) {
        if (n2 % 2 == 0)      { ip[0]++; n2 /= 2; }
        else if (n2 % 3 == 0) { ip[1]++; n2 /= 3; }
        else if (n2 % 5 == 0) { ip[2]++; n2 /= 5; }
        else break;
    }
}

// Factor a regular int
void factor(int N, std::array<int,3>& ip) {
    factor8(static_cast<long long>(N), ip);
}

// Compute integer power
static long long ipow(int base, int exp) {
    long long result = 1;
    for (int i = 0; i < exp; ++i) result *= base;
    return result;
}

// Choose NX, NY such that N = NX * NY, NX ≈ √N, both divisible by NPU
void pgetnxny(long long N, int& NX, int& NY, int NPU) {
    int isqrtn = static_cast<int>(std::sqrt(static_cast<double>(N)));

    std::array<int,3> lnpu, ip, lnx;
    factor(NPU, lnpu);
    factor8(N, ip);
    lnx = {0, 0, 0};

    int ires = isqrtn;
    for (int k = lnpu[2]; k <= (ip[2] + 1) / 2; ++k) {
        for (int j = lnpu[1]; j <= (ip[1] + 1) / 2; ++j) {
            for (int i = lnpu[0]; i <= (ip[0] + 1) / 2; ++i) {
                int nx_cand = static_cast<int>(ipow(2, i) * ipow(3, j) * ipow(5, k));
                if (nx_cand <= isqrtn) {
                    int ires2 = isqrtn - nx_cand;
                    if (ires2 < ires) {
                        lnx = {i, j, k};
                        ires = ires2;
                    }
                }
            }
        }
    }

    std::array<int,3> lny;
    for (int i = 0; i < 3; ++i) lny[i] = ip[i] - lnx[i];

    NX = static_cast<int>(ipow(2, lnx[0]) * ipow(3, lnx[1]) * ipow(5, lnx[2]));
    NY = static_cast<int>(ipow(2, lny[0]) * ipow(3, lny[1]) * ipow(5, lny[2]));
}
