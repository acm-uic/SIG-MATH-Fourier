import numpy as np
import acmFourier # Note that this the Python binding of our C++ code

# Very basic test case for start
def test_dft_basic():
    x = np.random.rand(128) + 1j*np.random.rand(128)
    np_fft = np.fft.fft(x)
    cpp_fft = np.array(acmFourier.dft(x))
    assert np.allclose(np_fft, cpp_fft, atol=1e-10)

def test_idft_basic():
    x = np.random.rand(128) + 1j*np.random.rand(128)
    np_ifft = np.fft.ifft(x)
    cpp_ifft = np.array(acmFourier.inverse_dft(x))
    assert np.allclose(np_ifft, cpp_ifft, atol=1e-10)


def test_recursive_fft_basic():
    x = np.random.rand(128) + 1j*np.random.rand(128)
    np_fft = np.fft.fft(x)
    cpp_fft = np.array(acmFourier.fft_recurse(x))
    assert np.allclose(np_fft, cpp_fft, atol=1e-10)

def test_recursive_ifft_basic():
    x = np.random.rand(128) + 1j*np.random.rand(128)
    np_ifft = np.fft.ifft(x)
    cpp_ifft = np.array(acmFourier.inverse_fft_recurse(x))
    assert np.allclose(np_ifft, cpp_ifft, atol=1e-10)


def test_bit_reversed_fft_basic():
    x = np.random.rand(128) + 1j*np.random.rand(128)
    np_fft = np.fft.fft(x)
    cpp_fft = np.array(acmFourier.fft_iterative_pow_of_2(x))
    assert np.allclose(np_fft, cpp_fft, atol=1e-10)

def test_bit_reversed_ifft_basic():
    x = np.random.rand(128) + 1j*np.random.rand(128)
    np_ifft = np.fft.ifft(x)
    cpp_ifft = np.array(acmFourier.inverse_fft_iterative_pow_of_2(x))
    assert np.allclose(np_ifft, cpp_ifft, atol=1e-10)

def test_fft_cuda():
    x = np.random.rand(128) + 1j*np.random.rand(128)
    np_ifft = np.fft.fft(x)
    cpp_ifft = np.array(acmFourier.fft_pow_of_2_cuda(x))
    assert np.allclose(np_ifft, cpp_ifft, atol=1e-10)