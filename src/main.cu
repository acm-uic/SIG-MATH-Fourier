#include "fourier.cu"

/***
*   main() code for timing and print debugging lol
***/
int main(int argc, char** argv)
{
    // Size of 1d signal
    uint32_t N = (1 << 16);

    // Generate basic array for testing
    std::vector<complex_t> X; 
    for (uint32_t k = 0; k < N; k++) {
        X.push_back(k);
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<complex_t> X_dft =  fft_pow_of_2_cuda(X);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration for CUDA FFT operation (including copy back output to host): " << duration.count() << " microseconds"  << std::endl;

/*
    for (uint32_t k = 0; k < N; k++) {
        std::cout << X_dft[k] << std::endl;
    }
*/

    return 0;
}