#include <iostream>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

/***
* Forward Discrete Fourier Transform
***/
std::vector<Complex> dft(std::vector<Complex>& X) 
{
    unsigned int N = X.size();
    if ((N & (N-1)) != 0)
        throw std::invalid_argument("Input size is not a power of 2");
    if (N <= 0)
        return {};

    std::vector<Complex> result(N, 0.0);
    
    for (unsigned int k = 0; k < N; k++) {
        Complex sum = Complex(0.0, 0.0); 
        for (unsigned int n = 0; n < N; n++) {
            Complex root_of_unity = static_cast<Complex>(std::polar(1.0, -2*M_PI*k*n / N));
            sum += X.at(n)*root_of_unity;
        }

        result.at(k) = sum;
    }

    return result;
}

/***
* Inverse Discrete Fourier Transform
***/
std::vector<Complex> inverse_dft(std::vector<Complex>& X) 
{
    unsigned int N = X.size();
    if ((N & (N-1)) != 0)
        throw std::invalid_argument("Input size is not a power of 2");
    if (N <= 0)
        return {};

    std::vector<Complex> result(N, 0.0);
    
    for (unsigned int k = 0; k < N; k++) {
        Complex sum = Complex(0.0, 0.0); 
        for (unsigned int n = 0; n < N; n++) {
            Complex root_of_unity = static_cast<Complex>(std::polar(1.0, 2*M_PI*k*n / N));
            sum += X.at(n)*root_of_unity;
        }

        // Note that we have to renormalize the sum for the inverse transform
        result.at(k) = sum / static_cast<double>(N);
    }

    return result;
}

/***
* Recursive implementation of FFT
***/
std::vector<Complex> fft_recurse(std::vector<Complex> X)
{
    // Length and helper vectors
    const unsigned int N = X.size();
    if ((N & (N-1)) != 0)
        throw std::invalid_argument("Input size is not a power of 2");
    if (N <= 0)
        return {};

    std::vector<Complex> evens(N/2);
    std::vector<Complex> odds(N/2);

    if (N == 1)
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

/***
* Recursive implementation of iFFT
***/
std::vector<Complex> inverse_fft_recurse(std::vector<Complex> X)
{
    // Length and helper vectors
    const unsigned int N = X.size();
    if ((N & (N-1)) != 0)
        throw std::invalid_argument("Input size is not a power of 2");
    if (N <= 0)
        return {};

    std::vector<Complex> evens(N/2);
    std::vector<Complex> odds(N/2);

    if (N == 1)
        return X;     // Recursion base case

    // Copy the evens and odds
    for (unsigned int k = 0; k < N/2; k++) {
        evens.at(k) = X.at(2*k);
        odds.at(k) = X.at(2*k + 1);
    }

    // Recurse the iDFT
    evens = inverse_fft_recurse(evens);
    odds = inverse_fft_recurse(odds);

    // Computation result. 
    // Note that we only need to pass through half the list since the second half's root of unity uses the opposite sign of the first half.
    for (unsigned int k = 0; k < N/2; k++) {
        Complex root_of_unity = std::polar(1.0, 2*M_PI*k / N);
        X.at(k) = evens.at(k) + root_of_unity * odds.at(k);
        X.at(k + N/2) = evens.at(k) - root_of_unity * odds.at(k);
    }

    // Normalization at each recursion level
    if (N > 1) {
        //#pragma omp parallel for
        for (unsigned int i = 0; i < N; i++) {
            X.at(i) /= 2.0;
        }
    }

    return X;
}

/***
* Bit-reversal power of 2 implementation of FFT
***/
std::vector<Complex> fft_iterative_pow_of_2(std::vector<Complex> X)
{
    // Length
    const unsigned int N = X.size();
    if ((N & (N-1)) != 0)
        throw std::invalid_argument("Input size is not a power of 2");
    if (N <= 0)
        return {};


    // Bit-reversal permutation
    unsigned int index_bit_reversed = 0;
    for (unsigned int index = 0; index < N; index++) {
        // Swapping and make sure we don't double swap
        if (index_bit_reversed > index)
            std::swap(X.at(index), X.at(index_bit_reversed));

        // Calculate the bit reversal of next index
        unsigned int right_shift = N >> 1;
        while ((right_shift >= 1) && (index_bit_reversed >= right_shift)) {
            index_bit_reversed -= right_shift;
            right_shift >>= 1;
        }
        index_bit_reversed += right_shift;
    }

    // FFT
    for (unsigned int stage = 2; stage <= N; stage *= 2) {

        // The stage's root of unity
        Complex root_of_unity = static_cast<Complex>(std::polar(1.0, -2*M_PI / stage));

        for (unsigned int group = 0; group < N; group += stage) {
            Complex twiddle = Complex(1.0, 0.0);
            for (unsigned int k = 0; k < stage/2 ; k++) {

                // Calculate the butterfly parts
                Complex p1 = X.at(group + k);
                Complex p2 = twiddle * X.at(group + k + stage/2);

                X.at(group + k) =  p1 + p2;            // Lower half
                X.at(group + k + stage/2) = p1 - p2;   // Higher half

                // Update twiddle 
                twiddle *= root_of_unity;
            }
        }
    }

    return X;
}

/***
* Bit-reversal power of 2 implementation of iFFT
***/
std::vector<Complex> inverse_fft_iterative_pow_of_2(std::vector<Complex> X)
{
    // Length
    const unsigned int N = X.size();
    if ((N & (N-1)) != 0)
        throw std::invalid_argument("Input size is not a power of 2");
    if (N <= 0)
        return {};


    // Bit-reversal permutation
    unsigned int index_bit_reversed = 0;
    for (unsigned int index = 0; index < N; index++) {
        // Swapping and make sure we don't double swap
        if (index_bit_reversed > index)
            std::swap(X.at(index), X.at(index_bit_reversed));

        // Calculate the bit reversal of next index
        unsigned int right_shift = N >> 1;
        while ((right_shift >= 1) && (index_bit_reversed >= right_shift)) {
            index_bit_reversed -= right_shift;
            right_shift >>= 1;
        }
        index_bit_reversed += right_shift;
    }

    // Inverse FFT
    for (unsigned int stage = 2; stage <= N; stage *= 2) {

        // The stage's root of unity
        Complex root_of_unity = static_cast<Complex>(std::polar(1.0, 2*M_PI / stage));

        for (unsigned int group = 0; group < N; group += stage) {
            Complex twiddle = Complex(1.0, 0.0);
            for (unsigned int k = 0; k < stage/2 ; k++) {

                // Calculate the butterfly parts
                Complex p1 = X.at(group + k);
                Complex p2 = twiddle * X.at(group + k + stage/2);

                X.at(group + k) =  p1 + p2;            // Lower half
                X.at(group + k + stage/2) = p1 - p2;   // Higher half

                // Update twiddle 
                twiddle *= root_of_unity;
            }
        }
    }

    // Renormalization
    //#pragma omp parallel for 
    for (unsigned int k = 0; k < N; k++) { X.at(k) /= static_cast<double>(N);}

    return X;
}