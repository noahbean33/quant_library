// Vectorized radix-2 FFT kernel using SSE3 intrinsics

#include <pmmintrin.h>  // SSE3

// SSE3 complex multiplication (defined in sse3_complex_multiply.cpp)
static __inline __m128d ZMUL(__m128d a, __m128d b)
{
    __m128d ar, ai;
    ar = _mm_movedup_pd(a);
    ar = _mm_mul_pd(ar, b);
    ai = _mm_unpackhi_pd(a, a);
    b  = _mm_shuffle_pd(b, b, 1);
    ai = _mm_mul_pd(ai, b);
    return _mm_addsub_pd(ar, ai);
}

// Radix-2 butterfly kernel using SSE3.
// a: input (interleaved complex doubles), b: output, w: twiddle factors
// m: multirow width (butterflies per group), l: number of groups
void fft_vec(double *a, double *b, double *w, int m, int l)
{
    int i, i0, i1, i2, i3, j;
    __m128d t0, t1, w0;

    for (j = 0; j < l; j++) {
        w0 = _mm_load_pd(&w[j << 1]);          // load twiddle for group j
        for (i = 0; i < m; i++) {
            i0 = (i << 1) + (j * m << 1);      // input: first half
            i1 = i0 + (m * l << 1);            // input: second half
            i2 = (i << 1) + (j * m << 2);      // output: even
            i3 = i2 + (m << 1);                // output: odd

            t0 = _mm_load_pd(&a[i0]);
            t1 = _mm_load_pd(&a[i1]);

            _mm_store_pd(&b[i2], _mm_add_pd(t0, t1));
            _mm_store_pd(&b[i3], ZMUL(w0, _mm_sub_pd(t0, t1)));
        }
    }
}
