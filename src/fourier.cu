#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <cstdint>
#include <complex>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>

using cudaComplex_t = cuda::std::complex<double>;
using complex_t = std::complex<double>;

#define THREADS_PER_BLOCK 256

/*
* Discrete Fourier Transform
*/
__global__ void dft_kernel(cudaComplex_t* input, cudaComplex_t* output, uint32_t N)
{
    // Point index
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the Fourier transform of this point
    if (k < N) {
        cudaComplex_t sum(0.0, 0.0);

        for (uint32_t m = 0; m < N; m++) {
            double theta = -2.0*M_PI*k*m / N;
            cudaComplex_t root_of_unity(cos(theta), sin(theta));
            sum += input[m] * root_of_unity;
        }

        // Storing the sum output
        output[k] = sum;
    }
}

std::vector<complex_t> dft_cuda(const std::vector<complex_t>& X)
{
    const uint32_t N = X.size();
    if ((N & (N-1)) != 0) {
        throw std::invalid_argument("Input size is not a power of 2");
    }

    // Initialize device data
    cudaComplex_t* d_input;
    cudaComplex_t* d_output;
    cudaMalloc(&d_input, N*sizeof(cudaComplex_t));
    cudaMalloc(&d_output, N*sizeof(cudaComplex_t));
    cudaMemcpy(d_input, X.data(), N*sizeof(cudaComplex_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, X.data(), N*sizeof(cudaComplex_t), cudaMemcpyHostToDevice);

    // Kernel launch and compute on device 
    const uint32_t n_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dft_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);


    // Copy data back to host
    std::vector<complex_t> output(N);
    cudaMemcpy(output.data(), d_output, N*sizeof(complex_t), cudaMemcpyDeviceToHost);

    // Free device data
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}


/*
* Bit-reversal kernel 
*/
__global__ void bit_reversal_kernel()
{
    // TODO
}


/*
* Butterfly operation kernel 
*/
__global__ void butterfly_kernel()
{
    // TODO
}