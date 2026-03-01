#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <cstdint>
#include <complex>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>

// Note we are going with double-precision complex so limited (please VERY limit) intrinsics acceleration :)
using cudaComplex_t = cuda::std::complex<double>;
using complex_t = std::complex<double>;

#define THREADS_PER_BLOCK 256


/***
*   The MVP for improving performance of Fourier Transform kernels: Pre-computing twiddle factors
***/

// Forward Fourier Transform twiddles
__global__ void precompute_twiddles(cudaComplex_t* twiddles, uint32_t N)
{
    // Index
    uint32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= N)
        return;

    // Computing twiddle factors
    double theta = 2.0*M_PI*k / N;
    twiddles[k] = cudaComplex_t(cos(theta), sin(theta));
}

// Inverse Fourier Transform Twiddles
__global__ void precompute_inverse_twiddles(cudaComplex_t* twiddles, uint32_t N)
{
    // Index
    uint32_t k = blockIdx.x*blockDim.x + threadIdx.x;

    // Computing twiddle factors
    if (k < N) {
        double theta = 2.0*M_PI*k / N;
        twiddles[k] = cudaComplex_t(cos(theta), sin(theta));
    }
}

/***
*   Discrete Fourier Transform
***/
__global__ void naive_dft_kernel(cudaComplex_t* input, cudaComplex_t* output, cudaComplex_t* twiddles, uint32_t N)
{
    // Point index
    uint32_t k = blockIdx.x*blockDim.x + threadIdx.x;

    // Compute the Fourier transform of this point
    if (k < N) {
        cudaComplex_t sum(0.0, 0.0);

        for (uint32_t m = 0; m < N; m++) {

            // Computing twiddle index (type conversion mess but this is just to be safe from overflowing)
            uint32_t twiddle_index = static_cast<uint32_t>(static_cast<uint64_t>(k)*m % N);
            sum += input[m] * twiddles[twiddle_index];
        }

        // Storing the sum output
        output[k] = sum;
    }
}

std::vector<complex_t> naive_dft_cuda(const std::vector<complex_t>& X)
{
    const uint32_t N = X.size();
    if ((N & (N-1)) != 0)
        throw std::invalid_argument("Input size is not a power of 2");
    if (N <= 0)
        return {};

    // Initialize device data
    cudaComplex_t* d_input;
    cudaComplex_t* d_output;
    cudaMalloc(&d_input, N*sizeof(cudaComplex_t));
    cudaMalloc(&d_output, N*sizeof(cudaComplex_t));
    cudaMemcpy(d_input, X.data(), N*sizeof(cudaComplex_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, X.data(), N*sizeof(cudaComplex_t), cudaMemcpyHostToDevice);

    // Kernel launch and compute on device 
    const uint32_t n_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaComplex_t* d_twiddles;
    cudaMalloc(&d_twiddles, N*sizeof(cudaComplex_t));
    precompute_twiddles<<<n_blocks, THREADS_PER_BLOCK>>>(d_twiddles, N);
    cudaDeviceSynchronize();

    naive_dft_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, d_twiddles, N);
    cudaDeviceSynchronize();

    // Copy data back to host
    std::vector<complex_t> output(N);
    cudaMemcpy(output.data(), d_output, N*sizeof(complex_t), cudaMemcpyDeviceToHost);

    // Free device data
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_twiddles);

    return output;
}

/***
*   Optimize way to do DFT with CUDA
***/

// Optimizing the DFT summing kernel via tiling the sum in blocks and pre-load inputs into shared memory region
// Essentially optimize the memory access pattern of cuda threads
__global__ void dft_kernel(cudaComplex_t* input, cudaComplex_t* output, cudaComplex_t* twiddles, uint32_t N)
{
    // Shared memory
    extern __shared__ cudaComplex_t shared_data[];

    // Point index
    uint32_t tid = threadIdx.x;
    uint32_t k = blockIdx.x*blockDim.x + tid;

    // Compute the Fourier transform of this point
    cudaComplex_t sum = cudaComplex_t(0.0, 0.0);
    for (uint32_t tile = 0; tile < N; tile += blockDim.x) {
        
        // Index within this block tile
        uint32_t m = tile + tid;

        // Populate the shared memory block with input data
        shared_data[tid] = (m < N) ? input[m] : 0.0;
        __syncthreads(); // Barrier: Make sure all shared data are populated before moving on

        // Computing the sum with respect to this tile
        if (k < N) {
            for (uint32_t j = 0; ((j < blockDim.x) && (tile+j < N)); j++) {

                // Computing twiddle index (type conversion mess but this is just to be safe)
                uint32_t twiddle_index = static_cast<uint32_t>((uint64_t)(k * (tile + j)) % N);

                // Accumulate the sum using shared data and pre-computed twiddles
                sum += shared_data[j] * twiddles[twiddle_index];
            }
        }

        // Again barrier for making sure operations are completed before moving onto next tile
        __syncthreads();
    }


    if (k < N) {
        output[k] = sum;
    }
}

// Host-side function to launch DFT kernels
std::vector<complex_t> dft_cuda(const std::vector<complex_t>& X)
{
    const uint32_t N = X.size();
    if ((N & (N-1)) != 0)
        throw std::invalid_argument("Input size is not a power of 2");
    if (N <= 0)
        return {};

    // Initialize device data
    cudaComplex_t* d_input;
    cudaComplex_t* d_output;
    cudaMalloc(&d_input, N*sizeof(cudaComplex_t));
    cudaMalloc(&d_output, N*sizeof(cudaComplex_t));
    cudaMemcpy(d_input, X.data(), N*sizeof(cudaComplex_t), cudaMemcpyHostToDevice);

    // Kernel launch data
    const uint32_t n_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const uint32_t shared_size = THREADS_PER_BLOCK * sizeof(cudaComplex_t);

    // Initialize and pre-compute twiddles on device
    cudaComplex_t* d_twiddles;
    cudaMalloc(&d_twiddles, N*sizeof(cudaComplex_t));
    precompute_twiddles<<<n_blocks, THREADS_PER_BLOCK>>>(d_twiddles, N);
    cudaDeviceSynchronize();

    // DFT kernel launch and compute on device 
    dft_kernel<<<n_blocks, THREADS_PER_BLOCK, shared_size>>>(d_input, d_output, d_twiddles, N);
    cudaDeviceSynchronize();

    // Copy data back to host
    std::vector<complex_t> output(N);
    cudaMemcpy(output.data(), d_output, N*sizeof(complex_t), cudaMemcpyDeviceToHost);

    // Free device data
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_twiddles);

    return output;
}


/***
* Fast Fourier Transform with CUDA
***/

// Bit-reversal kernel
__global__ void bit_reversal_permutation_kernel(cudaComplex_t* X, uint32_t N, uint32_t log2N)
{
    // Point index
    uint32_t k = blockDim.x*blockIdx.x + threadIdx.x;
    if (k >= N)
        return;

    // Bit-reversing the index
    uint32_t bit_rev_k = __brev(k);

    // Swap and making sure we don't double swap
    if (k < bit_rev_k) {
        cudaComplex_t tmp = X[k];
        X[k] = X[bit_rev_k];
        X[bit_rev_k] = tmp;
    }
}


// Butterfly operation kernel 
__global__ void butterfly_kernel()
{
    // TODO
}

// Host-side function to launch FFT workloads
std::vector<complex_t> fft_cuda(const std::vector<complex_t>& X)
{
    // Input size handling
    const uint32_t N = X.size();
    if ((N & (N-1)) != 0)
        throw std::invalid_argument("Input size is not a power of 2");
    if (N <= 0)
        return {};

    // Calculate log_2 of the size
    uint32_t log2N = 0;
    uint32_t temp = N;
    while (temp > 1) {
        temp /= 2;
        log2N++;
    }

    // Kernel launch data
    const uint32_t n_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Pre-compute twiddles factors
    cudaComplex_t* d_twiddles;
    cudaMalloc(&d_twiddles, N*sizeof(cudaComplex_t));
    precompute_twiddles<<<n_blocks, THREADS_PER_BLOCK>>>(d_twiddles, N);

    // Initialize device data
    cudaComplex_t* d_X;
    cudaMalloc(&d_X, N*sizeof(cudaComplex_t));
    cudaMemcpy(d_X, X.data(), N*sizeof(cudaComplex_t), cudaMemcpyHostToDevice);

    // Bit-reversal
    bit_reversal_permutation_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(d_X, N, log2N);
    cudaDeviceSynchronize();

    // TODO: Finish the butterfly kernel and compute

    // Freeing device data
    cudaFree(d_X);
    cudaFree(d_twiddles);

    return {};
}
