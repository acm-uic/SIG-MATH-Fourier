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
            : params(HeatParameters())
        {}
        
    private:
        HeatParameters params;
};

int main(int argc, char** argv)
{
    return 0;
}