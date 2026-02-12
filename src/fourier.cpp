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

int main(int argc, char* argv[])
{
    std::vector<Complex> X;
    unsigned int N = 8;
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