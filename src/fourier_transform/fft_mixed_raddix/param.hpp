// FFTE Parameter Header
// Defines compile-time constants for the FFT library

#ifndef FFTE_PARAM_HPP
#define FFTE_PARAM_HPP

// The maximum supported number of processors
static const int MAXNPU = 65536;

// The maximum supported 2-D transform length
static const int NDA2 = 65536;

// The maximum supported 3-D transform length
static const int NDA3 = 4096;

// Blocking parameter for cache optimization
static const int NBLK = 16;

// Blocking parameter for NVIDIA GPUs
static const int NB = 128;

// Padding parameter to avoid cache conflicts in FFT routines
static const int NP = 8;

// Size of L2 cache (in bytes)
static const int L2SIZE = 8388608;

#endif // FFTE_PARAM_HPP
