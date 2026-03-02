#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <cstdint>
#include <complex>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>

// Note we are going with double-precision complex so limited (please VERY limit) intrinsics operations :)
using cudaComplex_t = cuda::std::complex<double>;
using complex_t = std::complex<double>;

std::vector<complex_t> fft_pow_of_2_cuda(const std::vector<complex_t>& X);
std::vector<complex_t> naive_dft_cuda(const std::vector<complex_t>& X);
std::vector<complex_t> dft_cuda(const std::vector<complex_t>& X);