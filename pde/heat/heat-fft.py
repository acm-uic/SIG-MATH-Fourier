#%%
import numpy as np
import matplotlib.pyplot as plt
import sigMathFourier

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
# Temporal information
dt = 1e-05                  # Time-step size
num_t = int(np.ceil(T/dt))  # Numer of time-steps

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

solutions = np.zeros((num_t + 1, N))
solutions[0] = initial_condition(x)
u_hat = sigMathFourier.fft_iterative_pow_of_2(initial_condition(x))

for t_index in range(num_t):
    u_hat = rk4_explicit_step(u_hat, dt)
    solutions[t_index + 1] = np.array(sigMathFourier.inverse_fft_iterative_pow_of_2(u_hat)).real

#%%

# Plot solutions
from mpl_toolkits.mplot3d import Axes3D

# Grid setup
t = np.linspace(0, T, num_t + 1)
X, TT = np.meshgrid(x, t)

# Analytical solution and pointwise error
analytic = -np.sin(np.pi * X) * np.exp(-D * (np.pi**2) * TT) # Analytical solution
pt_err = np.abs(analytic - solutions)

# Figure
fig = plt.figure(figsize=(15,6))

# Numerical solution plot
ax1 = fig.add_subplot(131, projection="3d")
ax1.plot_surface(X, TT, solutions,cmap="viridis")
ax1.set_xlabel("x")
ax1.set_ylabel("t")
ax1.set_zlabel("u(x,t)")
ax1.set_title("Numerical Solution with Fourier Transforms")

# Analytical solution plot
ax2 = fig.add_subplot(132, projection="3d")
ax2.plot_surface(X, TT, analytic, cmap="viridis")
ax2.set_xlabel("x")
ax2.set_ylabel("t")
ax2.set_zlabel("u(x,t)")
ax2.set_title("Analytical Solution")

# Absolute error
ax3 = fig.add_subplot(133)
pcm = ax3.pcolormesh(X, TT, pt_err, cmap="magma")
ax3.set_xlabel("x")
ax3.set_ylabel("t")
ax3.set_title("Absolute Error")
fig.colorbar(pcm, ax=ax3, label="Pointwise error")

plt.tight_layout()
plt.savefig("heat-fft.png", dpi=400)
plt.show()

#%%
print(f"Maximum pointwise error: {np.max(pt_err)}")
print(f"Mean pointwise error: {np.mean(pt_err)}")