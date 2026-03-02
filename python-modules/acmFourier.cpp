#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "../src/fourier.hpp" // Hard-coding paths cuz LOL

namespace py = pybind11;

PYBIND11_MODULE(acmFourier, m) {
    m.def("dft", &dft);
    m.def("fft_recurse", &fft_recurse);
    m.def("fft_iterative_pow_of_2", &fft_iterative_pow_of_2);
    m.def("inverse_dft", &inverse_dft);
    m.def("inverse_fft_recurse", &inverse_fft_recurse);
    m.def("inverse_fft_iterative_pow_of_2", &inverse_fft_iterative_pow_of_2);
}