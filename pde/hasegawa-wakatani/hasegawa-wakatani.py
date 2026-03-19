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

#%%
"""
Grid building (build on CPU first then copy over to GPU if available)
"""
# Physical space grid
x_np = np.linspace(x_low, x_high, Nx, endpoint=False)
y_np = np.linspace(y_low, y_high, endpoint=False)
X_np, Y_np = np.meshgrid(x_np, y_np, indexing="ij")

# Frequency space grid
kx_np , ky_np = np.fft.fftfreq(Nx, d=Lx/Nx) , np.fft.fftfreq(Ny, d=Ly/Ny) 
KX_np , KY_np = np.meshgrid(kx_np, ky_np, indexing="ij")
K_squared_np = KX_np**2 + KY_np**2

# Upload to grid to GPU once (if available)
KX = xp.asarray(KX_np)
KY = xp.asarray(KY_np)
K_SQUARED = xp.asarray(K_squared_np)

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

def kappa(X,s):
    """
    kappa = -d/dx(ln n_0)
    """
    return 2.0*X**2 / s**2

# TODO

#%%
"""
Dealiasing for stability
"""


#%%
"""
Poisson bracket and Nonlinear Term
"""
def poisson_bracket():
    pass

#%%

"""
Time derivative and stepping
"""
def spectral_time_derivative():
    pass

def explicit_rk4_step():
    pass

# TODO

#%%
"""
Main solver
"""