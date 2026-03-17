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
Nx, Ny = 512, 512                   # Sizes along dimension of domain
x_low, x_high = [-2*np.pi, 2*np.pi] # x-dimension rectangular bounds
y_low, y_high = [-2*np.pi, 2*np.pi] # y-dimension rectangular bounds
dt = 0.005                          # Time-step size
plot_interval = 20                  # Simulation interval before rendering

# Actual coupled equations parameters
alpha = 0.1                         # Adiabicity

# TODO

#%%
"""
Grid building (build on CPU first then copy over to GPU if available)
"""
x_np = np.linspace(x_low, x_high, Nx, endpoint=False)
y_np = np.linspace(y_low, y_high, endpoint=False)
X_np, Y_np = np.meshgrid(x_np, y_np, indexing='ij')

#%%
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

#%%
"""
Initial profile (condition)
"""

# TODO