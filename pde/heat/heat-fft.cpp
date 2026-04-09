#include "fourier.hpp"
#include <functional>
#include <cmath>
#include <memory>
#include <numeric>

#define square(x) ((x)*(x))


// Basically here to specify the config of the problem
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
        , t_0(0.0), T(1.0), dt(1e-3)
        , D(0.2)
        , N(512)
    {}
};

// Have to make a Solution wrapper because we are not requiring C++23,... yet
// Psst, g++ 15 support for std::mdspan are still iffy for most machines at the time of writing
class Solution 
{
    public:
        /* Constructors */
        Solution(unsigned int time_steps, unsigned int spatial_points)
            : data(std::make_unique<double[]>((time_steps + 1) * spatial_points))
            , num_time_steps(time_steps) 
            , num_spatial_points(spatial_points)
        {}

        /* Reading solution data */
        double operator() (unsigned int row, unsigned int col) {
            // Bounds check
            if (row > num_time_steps || col >= num_spatial_points)
                throw std::out_of_range("Solution indexing out of range");

            // Return (note read-only)
            return data[row*num_spatial_points + col];
        }
    
    private:
        /* Data fields */
        std::unique_ptr<double[]> data;
        unsigned int num_time_steps;
        unsigned int num_spatial_points;
        
        // Only can the solving engine write to the solution
        friend class HeatPDESolver1D;

        /* Solution indexing for write */
        double& at(unsigned int row, unsigned int col) {
            // Bounds check
            if (row > num_time_steps || col >= num_spatial_points)
                throw std::out_of_range("Solution indexing out of range");

            // Return data reference for writing
            return data[row*num_spatial_points + col];
        }
};


// Actual Solving class
class HeatPDESolver1D
{
    public:
        HeatPDESolver1D()
            : config(HeatParameters())
        {}

        /* Public methods */
        Solution solve();
        const HeatParameters get_config() {return this->config;}
        const std::vector<double> get_x() {return this->x;}
        
    private:
        
        /* Data fields */
        HeatParameters config;
        std::vector<double> x;
        std::vector<double> k_squared;

        /* Private methods */
        inline void setup();
        std::vector<Complex> fourier_time_derivative(const std::vector<Complex>& u_hat);
        std::vector<Complex> rk4_explicit_step(const std::vector<Complex>& u_hat);
        inline std::vector<Complex> _rk_4_daxpy(const std::vector<Complex>& X, const std::vector<Complex>& Y, double a);
};


// Setting up the domains (both physical and frequency)
inline void HeatPDESolver1D::setup()
{
    // Compute neccessary information from the config
    const double L = config.x_final - config.x_0;
    const unsigned int N = config.N;

    // Initalize domains
    x.resize(N);
    k_squared.resize(N);
    for (unsigned int n = 0; n < N; n++) {

        // Computing the physical domain based on config
        x[n] = config.x_0 + (L * n) / N;

        // Compute the (squared) frequency domain
        k_squared[n] = (n <= N/2) ? square(2*M_PI*n / L) : square((2*M_PI*(static_cast<int>(n) - static_cast<int>(N))) / L);
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

Solution HeatPDESolver1D::solve()
{
    // Setup
    setup();
    unsigned int time_steps = static_cast<unsigned int>(std::ceil((config.T - config.t_0) / config.dt));
    unsigned int N = config.N;

    // Setup solution buffer
    Solution solution(time_steps, N);

    // Initial condition applying
    std::vector<Complex> u_hat(N);
    for (unsigned int n = 0; n < N; n++) {
        double init_val = config.initial_condition(x[n]);

        solution.at(0, n) = init_val;
        u_hat[n] = Complex(init_val, 0.0);
    }

    // Fourier Transform the initial condition
    u_hat = fft_iterative_pow_of_2(u_hat);

    // Time-stepping
    std::vector<Complex> u_phys(N);
    for (unsigned int step = 0; step < time_steps; step++) {
        
        // Computing physical solution
        u_hat = rk4_explicit_step(u_hat);
        u_phys = inverse_fft_iterative_pow_of_2(u_hat);
        
        // Storing the solution
        for (unsigned int col = 0; col < N; col++) {
            solution.at(step+1, col) = u_phys[col].real();
        }
    }


    return solution;
}

/*
*   main()
*/
int main(int argc, char** argv)
{
    // Obtaining the computed solution
    HeatPDESolver1D solver;
    Solution heat_sol = solver.solve();

    // Config
    const HeatParameters cfg = solver.get_config();

    // Spatial doamin info
    const std::vector<double> x_vals = solver.get_x();
    const unsigned int Nx = x_vals.size();

    // Time domain info
    unsigned int Nt = static_cast<unsigned int>(std::ceil((cfg.T - cfg.t_0) / cfg.dt) + 1);
    std::vector<double> t_vals(Nt);
    for (unsigned int k = 0; k < Nt; k++) {t_vals[k] = cfg.t_0 + k*cfg.dt;}

    // Setting up meshgrid for plotting
    std::vector<std::vector<double>> u_matrix(Nt, std::vector<double>(Nx));
    for (unsigned int i = 0; i < Nt; ++i) {
        for (unsigned int j = 0; j < Nx; ++j) {
            u_matrix[i][j] = heat_sol(i, j);
        }
    }

    return 0;
}