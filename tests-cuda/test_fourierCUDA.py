import numpy as np
import acmFourierCUDA # Note that this the Python binding of our CUDA code

# Very basic test case for start
def test_fft_cuda():
    x = np.random.rand(128) + 1j*np.random.rand(128)
    np_fft = np.fft.fft(x)
    cuda_fft = np.array(acmFourierCUDA.fft_pow_of_2_cuda(x))
    assert np.allclose(np_fft, cuda_fft, atol=1e-10)

def test_ifft_cuda():
    x = np.random.rand(128) + 1j*np.random.rand(128)
    np_ifft = np.fft.ifft(x)
    cuda_ifft = np.array(acmFourierCUDA.inverse_fft_pow_of_2_cuda(x))
    assert np.allclose(np_ifft, cuda_ifft, atol=1e-10)

def test_dft_cuda():
    x = np.random.rand(128) + 1j*np.random.rand(128)
    np_dft = np.fft.fft(x)
    cuda_dft = np.array(acmFourierCUDA.dft_cuda(x))
    assert np.allclose(np_dft, cuda_dft, atol=1e-10)

def test_idft_cuda():
    x = np.random.rand(128) + 1j*np.random.rand(128)
    np_idft = np.fft.ifft(x)
    cuda_idft = np.array(acmFourierCUDA.inverse_dft_cuda(x))
    assert np.allclose(np_idft, cuda_idft, atol=1e-10)