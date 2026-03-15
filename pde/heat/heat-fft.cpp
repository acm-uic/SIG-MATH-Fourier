#include "fourier.hpp"

struct HeatParameters
{
    double x_0;     // Lower physical domain bound
    double x_final; // Upper physical domain bound
    double t_0;     // Initial time
    double T;       // Final time
    double D;       // Diffusivity
    unsigned int N; // Number of spatial points

    // Default config
    HeatParameters()
        : x_0(-1.0), x_final(1.0)
        , t_0(0.0), T(1.0)
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