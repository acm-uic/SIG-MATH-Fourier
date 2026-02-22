#include <iostream>
#include <complex>
#include <vector>
#include<chrono>

using  Complex =  std::complex<double>;

std::vector<Complex> dft(std::vector<Complex>& X) 
{
    unsigned int N = X.size();
    std::vector<Complex> result(N, 0.0);
    
    for (unsigned int k = 0; k < N; k++) {
        Complex sum = Complex(0.0,0.0);
        for (unsigned int n = 0; n < N; n++) {
            Complex root_of_unity = static_cast<Complex>(std::polar(1.0, -2*M_PI*k*n / N));
            sum += X.at(n)*root_of_unity;
        }

        result.at(k) = sum;
    }

    return result;
}

std::vector<Complex> fft_recurse(std::vector<Complex> X)
{
    // Length and helper vectors
    const unsigned int N = X.size();
    std::vector<Complex> evens(N/2);
    std::vector<Complex> odds(N/2);

    if (N == 1 )
        return X;     // Recursion base case

    // Copy the evens and odds
    for (unsigned int k = 0; k < N/2; k++) {
        evens.at(k) = X.at(2*k);
        odds.at(k) = X.at(2*k + 1);
    }

    // Recurse the DFT
    evens = fft_recurse(evens);
    odds = fft_recurse(odds);

    // Computation result. 
    // Note that we only need to pass through half the list since the second half's root of unity uses the opposite sign of the first half.
    for (unsigned int k = 0; k < N/2; k++) {
        Complex root_of_unity = std::polar(1.0, -2*M_PI*k / N);
        X.at(k) = evens.at(k) + root_of_unity * odds.at(k);
        X.at(k + N/2) = evens.at(k) - root_of_unity * odds.at(k);
    }
    return X;
}

int main(int argc, char* argv[])
{
    std::vector<Complex> X;
    unsigned int N = 32;
    for (unsigned int k = 0; k < N; k++) {
        X.push_back(k);   
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Complex> result = dft(X);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Duration for DFT: " << duration.count() << " microseconds"  << std::endl;

    for (unsigned int k = 0; k < N; k++) {
        std::cout << result.at(k) << std::endl; 
    }


    return 0;
}