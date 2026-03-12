#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
# Problem parameters for the Heat equation problem
x_0, x_end = [-1,1] # Domain
N = 512             # Spatial points
D = 0.2             # Thermal diffusivity
T = 1.0             # Final time

#%%
# Initial condition
def initial_condition(x):
    return -np.sin(np.pi * x)

# Boundary condition (assuming periodic)
def boundary_condition(t):
    return 0*t

#%%
# Actual solving (should have encapsulate this into an object but oh well)
# https://scicomp.stackexchange.com/questions/43962/what-is-the-advantage-of-using-a-particular-rk-scheme 

# Temporal information
dt = 1e-05                      # Time-step size
n_t = int(np.ceil(T/dt))        # Numer of time-steps

# Physical domain
L = x_end - x_0
x  = x_0 + np.linspace(0, L, N, endpoint=False)

# Frequency domain
k = np.zeros(N)
for n in range(N):
    if n <= N//2:
        k[n] = (2*np.pi*n) / L
    else:
        k[n] = (2*np.pi*(n - N)) / L

k_squared = k**2

#%%
# Time integration on frequency domain
def fourier_time_derivative(u_hat):
    return -D * k_squared * u_hat

def rk4_explicit_step(u_hat, dt):
    k1 = fourier_time_derivative(u_hat)
    k2 = fourier_time_derivative(u_hat + 0.5*k1*dt)
    k3 = fourier_time_derivative(u_hat + 0.5*k2*dt)
    k4 = fourier_time_derivative(u_hat + k3*dt)
    return u_hat + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)

solutions = np.zeros((n_t + 1, N))
solutions[0] = initial_condition(x)
u_hat = np.fft.fft(initial_condition(x))

for t_index in range(n_t):
    u_hat = rk4_explicit_step(u_hat, dt)
    solutions[t_index + 1] = np.fft.ifft(u_hat).real

#%%

# Plot solutions
