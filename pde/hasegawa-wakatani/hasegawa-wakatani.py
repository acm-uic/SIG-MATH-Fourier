#%%
import numpy as np
import matplotlib.pyplot as plt

# Importing (Num|Cu)Py aliased as xp based on systems GPU availability
try:
    import cupy as xp
    CUDA_AVAILABLE=True
    _dummy = xp.zeros(1)    # Warm-up the gpu
    xp.cuda.Stream.null.synchronize()
    print("CUDA available with CuPy")
except:
    import numpy as xp
    CUDA_AVAILABLE=False
    print("CPU compute. No CUDA availability")

#%%
"""
Parameters for the problem
"""
# Spatial and temporal information of the problem
Nx, Ny = 512, 512                       # Resolution of the grid: (Nx x Ny)
x_low, x_high = [-20*np.pi, 20*np.pi]   # x-dimension rectangular bounds
y_low, y_high = [-20*np.pi, 20*np.pi]   # y-dimension rectangular bounds
Lx, Ly = x_high-x_low, y_high-y_low     # Length of the rectangle

dt = 0.005                          # Time-step size
plot_interval = 20                  # Simulation interval before rendering

# Actual coupled equations parameters
alpha = 0.1     # Adiabicity (note that this is constant i.e. this is a 2D problem)
mu = 1.0        # Hyper-diffusion coefficient
kappa = 1.0     # Background gradient

#%%
"""
Grid building (build on CPU first then copy over to GPU if available)
"""
# Physical space grid
x_np = np.linspace(x_low, x_high, Nx, endpoint=False)
y_np = np.linspace(y_low, y_high, Ny, endpoint=False)
X_np, Y_np = np.meshgrid(x_np, y_np, indexing="ij")

# Frequency space grid
kx_np , ky_np = np.fft.fftfreq(Nx, d=Lx/Nx) , np.fft.fftfreq(Ny, d=Ly/Ny) 
KX_np , KY_np = np.meshgrid(kx_np, ky_np, indexing="ij")
K2_np = KX_np**2 + KY_np**2
K4_np = (K2_np)**2
inv_K2_np = np.where(K2_np == 0, 0.0, 1.0 / np.where(K2_np == 0, 1.0, K2_np))

# 2/3 Dealiasing information
kx_max = np.max(abs(kx_np))
ky_max = np.max(abs(ky_np))
dealias_np = (
                (np.abs(KX_np) <= (2/3)*kx_max) & 
                (np.abs(KY_np) <= (2/3)*ky_max)
).astype(np.float64)

# Upload to grid to GPU once (if available)
X = xp.asarray(X_np)
Y = xp.asarray(Y_np)
KX = xp.asarray(KX_np)
KY = xp.asarray(KY_np)
K2 = xp.asarray(K2_np)
K4 = xp.asarray(K4_np)
inv_K2 = xp.asarray(inv_K2_np)
DEALIAS = xp.asarray(dealias_np)

#%%
"""
Initial profile (condition)
"""
s = 2.0
def initial_density(X,Y,s):
    """
    We are considering the Gaussian initial density profile
    n_0(x) = e^{-(x^2+y^2)/s^2}
    """
    return xp.exp(-(X**2 + Y**2) / s**2)


#%%
"""
Poisson bracket and Nonlinear Term
"""

# Helpers for computing spectral gradients
def dx(f_hat): return 1j * KX * f_hat
def dy(f_hat): return 1j * KY * f_hat

# Poisson bracket calculation
def poisson_bracket(f_hat, g_hat):
    """
    Computing the psuedo-spectral Poisson-bracket of f and g (with dealiasing)
        {f,g} = FFT[ (iFFT(f_x) * iFFT(g_y)) - (iFFT(f_y) * iFFT(g_x)) ]
    """
    return DEALIAS * xp.fft.fft(
        xp.fft.ifft2(dx(f_hat)) * xp.fft.ifft2(dy(g_hat)) - xp.fft.ifft2(dy(f_hat)) * xp.fft.ifft2(dx(g_hat))
    )

#%%

"""
Time derivative and stepping
"""
def spectral_time_derivative(vorticity_hat, density_hat):
    """
        Computing the resulting spectral time-gradients of the system:
            dt_vort = alpha(phi_hat - density_hat) - mu*K_4*vortcity_hat - {phi_hat, vorticity_hat}
            dt_dens = alpha(phi_hat - density_hat) - kappa*i*k_y*phi_hat - {phi_hat, density_hat}
    """
    phi_hat = -vorticity_hat * inv_K2   # Spectral stream function
    coupling_term = alpha*(phi_hat - density_hat)

    dt_vort = coupling_term - mu*K4*vorticity_hat - poisson_bracket(phi_hat, vorticity_hat)
    dt_dense = coupling_term - poisson_bracket(phi_hat, density_hat) - kappa*dy(phi_hat)
    return dt_vort, dt_dense
    

def explicit_rk4_step(vorticity_hat, density_hat):

   # Runge-Kutta components
   k1_vort, k1_dens = spectral_time_derivative(vorticity_hat,                   density_hat)
   k2_vort, k2_dens = spectral_time_derivative(vorticity_hat + 0.5*k1_vort*dt,  density_hat + 0.5*k1_dens*dt)
   k3_vort, k3_dens = spectral_time_derivative(vorticity_hat + 0.5*k2_vort*dt,  density_hat + 0.5*k2_dens*dt)
   k4_vort, k4_dens = spectral_time_derivative(vorticity_hat + k3_vort*dt,      density_hat + k3_dens*dt)

   # RK4-step (with dealiasing)
   vort_update = DEALIAS * (k1_vort + 2.0*k2_vort + 2.0*k3_vort + k4_vort)*(dt/6.0)
   dens_update = DEALIAS * (k1_dens + 2.0*k2_dens + 2.0*k3_dens + k4_dens)*(dt/6.0)
   return vort_update, dens_update

#%%
"""
Main solver
"""

if __name__ == "__main__":

    # Initializing states
    density_hat = xp.fft.fft2(initial_density(X,Y,s))
    vorticity_hat  = xp.zeros((Nx, Ny), dtype=complex)

    # Initializing temporal information
    t = 0.0
    step = 0

    # Infinite compute
    try:
        while True:
            vorticity_hat, density_hat = explicit_rk4_step(vorticity_hat, density_hat)
            t += dt
            step += 1
    except:
        print("\nStopped.")