#include "fourier.cu"

/***
*   main() code for timing and print debugging lol
***/
int main(int argc, char** argv)
{
    // Size of 1d signal
    uint32_t N = (1 << 16);

    // Calculate log_2 of the size
    uint32_t log2N = 0;
    uint32_t temp = N;
    while (temp > 1) {
        temp /= 2;
        log2N++;
    }

    // Generate basic array for testing
    std::vector<complex_t> X; 
    for (uint32_t k = 0; k < N; k++) {
        X.push_back(k);
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<complex_t> X_dft =  dft_cuda(X);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration for CUDA DFT operation (including copy back output to host): " << duration.count() << " microseconds"  << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<complex_t> X_naive_dft =  naive_dft_cuda(X);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration for naive CUDA DFT operation (including copy back output to host): " << duration.count() << " microseconds"  << std::endl;

    //for (uint32_t k = 0; k < N; k++) {
    //    std::cout << X_dft[k] << std::endl;
    //}

    return 0;
}