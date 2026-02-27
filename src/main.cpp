#include "fourier.hpp"

#include <chrono>

int main(int argc, char* argv[])
{
    std::vector<Complex> X;
    unsigned int N = 8192;
    for (unsigned int k = 0; k < N; k++) {
        X.push_back(k);   
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Complex> result = dft(X);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration for DFT: " << duration.count() << " microseconds"  << std::endl;

    start = std::chrono::high_resolution_clock::now();
    result = fft_recurse(X);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration for recursive FFT: " << duration.count() << " microseconds"  << std::endl;

    start = std::chrono::high_resolution_clock::now();
    result = fft_iterative_pow_of_2(X);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration for bit-reversed FFT: " << duration.count() << " microseconds"  << std::endl;

/*
    for (unsigned int k = 0; k < N; k++) {
        std::cout << result.at(k) << std::endl; 
    }
*/

    return 0;
}