#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "../src/fourier-cuda.hpp"    // Hard-coding paths cuz LOL

namespace py = pybind11;

PYBIND11_MODULE(acmFourierCUDA, m) {
    m.def("fft_pow_of_2_cuda", &fft_pow_of_2_cuda);
    m.def("inverse_fft_pow_of_2_cuda", &inverse_fft_pow_of_2_cuda);
    m.def("dft_cuda", &dft_cuda);
    m.def("inverse_dft_cuda", &inverse_dft_cuda);
}