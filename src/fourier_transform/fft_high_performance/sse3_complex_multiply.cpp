// Complex multiplication using SSE3 intrinsics

#include <pmmintrin.h>  // SSE3

// Multiplies two complex numbers packed in __m128d registers.
// Each register holds [real, imag].
// Returns [a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r]
static __inline __m128d ZMUL(__m128d a, __m128d b)
{
    __m128d ar, ai;
    ar = _mm_movedup_pd(a);         // ar = [a.r, a.r]
    ar = _mm_mul_pd(ar, b);         // ar = [a.r*b.r, a.r*b.i]
    ai = _mm_unpackhi_pd(a, a);     // ai = [a.i, a.i]
    b  = _mm_shuffle_pd(b, b, 1);   // b  = [b.i, b.r]
    ai = _mm_mul_pd(ai, b);         // ai = [a.i*b.i, a.i*b.r]
    return _mm_addsub_pd(ar, ai);   // [a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r]
}
