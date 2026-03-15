#include "fourier.hpp"
#include <functional>
#include <cmath>

#define square(x) ((x)*(x))


struct HeatParameters
{
    std::function<double(double)> initial_condition;
    double x_0;     // Lower physical domain bound
    double x_final; // Upper physical domain bound
    double t_0;     // Initial time
    double T;       // Final time
    double dt;      // Time stepsize
    double D;       // Diffusivity
    unsigned int N; // Number of spatial points

    // Default config
    HeatParameters()
        : initial_condition([] (double x) {return -std::sin(M_PI*x);} )
        , x_0(-1.0), x_final(1.0)
        , t_0(0.0), T(1.0), dt(1e-5)
        , D(0.2)
        , N(512)
    {}
};

class HeatPDESolver1D
{
    public:
        HeatPDESolver1D()
            : config(HeatParameters())
        {}

        void solve();   // TODO
        
    private:
        
        /* Data fields */
        HeatParameters config;
        std::vector<double> x;
        std::vector<double> k_squared;

        /* Private methods */
        void setup();
        std::vector<Complex> fourier_time_derivative(const std::vector<Complex>& u_hat);
        std::vector<Complex> rk4_explicit_step(const std::vector<Complex>& u_hat);
        inline std::vector<Complex> _rk_4_daxpy(const std::vector<Complex>& X, const std::vector<Complex>& Y, double a);
};


// Setting up the domains (both physical and frequency)
void HeatPDESolver1D::setup()
{
    // Compute neccessary information from the config
    const double L = config.x_final - config.x_0;
    const unsigned int N = config.N;

    for (unsigned int n = 0; n < N; n++) {

        // Computing the physical domain based on config
        x[n] = (L * n) / N;

        // Compute the (squared) frequency domain
        k_squared[n] = (n <= N/2) ? square(2*M_PI*n / L) : square((2*M_PI*(n - N)) / L);
    }
}

// Computing the Heat's equation's spectral time derivative
std::vector<Complex> HeatPDESolver1D::fourier_time_derivative(const std::vector<Complex>& u_hat)
{
    std::vector<Complex> d_uhat_dt(config.N);

    for (unsigned int n = 0; n < config. N; n++) {
        d_uhat_dt[n] = -config.D * u_hat[n] * k_squared[n];
    }

    return d_uhat_dt;
}

// Explicit 4-th order Runge-Kutta step
// DAXPY operations inlining helper
inline std::vector<Complex> HeatPDESolver1D::_rk_4_daxpy(const std::vector<Complex>& X, const std::vector<Complex>& Y, double a)
{
    std::vector<Complex> result(config.N);
    for (unsigned int n = 0; n < config.N; n++) {
        result[n] = X[n] + a*Y[n]; 
    }
    return result;
}

// Actual time-stepping 
std::vector<Complex> HeatPDESolver1D::rk4_explicit_step(const std::vector<Complex>& u_hat)
{
    std::vector<Complex> k1 = fourier_time_derivative(u_hat);
    std::vector<Complex> k2 = fourier_time_derivative(_rk_4_daxpy(u_hat, k1, 0.5*config.dt));
    std::vector<Complex> k3 = fourier_time_derivative(_rk_4_daxpy(u_hat, k2, 0.5*config.dt));
    std::vector<Complex> k4 = fourier_time_derivative(_rk_4_daxpy(u_hat, k3, config.dt));

    std::vector<Complex> result(config.N);
    for (unsigned int n = 0; n < config.N; n++) {
        result[n] = u_hat[n] + (k1[n] + 2.0*k2[n] + 2.0*k3[n] + k4[n]) * (config.dt / 6.0);
    }

    return result;
}



/*
*   main()
*/
int main(int argc, char** argv)
{
    return 0;
}