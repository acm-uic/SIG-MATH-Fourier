import numpy as np
import acmFourier

# Very basic test case for start
def test_dft_basic():
    x = np.random.rand(32) + 1j*np.random.rand(32)
    np_fft = np.fft.fft(x)
    cpp_fft = np.array(acmFourier.dft(x))
    assert np.allclose(np_fft, cpp_fft, atol=1e-10)


def test_recursive_fft_basic():
    x = np.random.rand(32) + 1j*np.random.rand(32)
    np_fft = np.fft.fft(x)
    cpp_fft = np.array(acmFourier.fft_recurse(x))
    assert np.allclose(np_fft, cpp_fft, atol=1e-10)


def test_bit_reversed_fft_basic():
    x = np.random.rand(32) + 1j*np.random.rand(32)
    np_fft = np.fft.fft(x)
    cpp_fft = np.array(acmFourier.fft_iterative_pow_of_2(x))
    assert np.allclose(np_fft, cpp_fft, atol=1e-10)