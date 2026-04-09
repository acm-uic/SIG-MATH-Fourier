#%%
import numpy as np
import matplotlib
from pathlib import Path

import cupy as xp
from cuda.bindings.driver import (
    cuMemcpy2D,
    CUDA_MEMCPY2D,
    CUmemorytype,
    CUresult,
    CUgraphicsRegisterFlags,
    cuGraphicsGLRegisterImage,
    cuGraphicsMapResources,
    cuGraphicsUnmapResources,
    cuGraphicsSubResourceGetMappedArray,
    cuGraphicsUnregisterResource,
)

import moderngl
import moderngl_window as mglw
from OpenGL.GL import GL_TEXTURE_2D

# Windows configuration
import os
import sys
if (os.name == "nt"):
    # CUDA path configuration
    CUDA_VERSION = 13.2   # Change this to ther Windows verion
    cuda_path = fr"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{CUDA_VERSION}\bin\x64"

    # Validate and add the valid CUDA path
    if (os.path.exists(cuda_path)):
        os.add_dll_directory(cuda_path)
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
    else:
        print("Invalid Windows CUDA path! Terminating!")
        sys.exit(1)

#%%
"""
Parameters for the problem
"""
# Spatial and temporal information of the problem
Nx, Ny = 512, 512                       # Resolution of the grid: (Nx x Ny)
x_low, x_high = [-10*np.pi, 10*np.pi]   # x-dimension rectangular bounds
y_low, y_high = [-10*np.pi, 10*np.pi]   # y-dimension rectangular bounds
Lx, Ly = x_high-x_low, y_high-y_low     # Length of the rectangle

dt = 0.01                          # Time-step size
plot_interval = 1                  # Simulation interval before rendering

# Actual coupled equations parameters
alpha = 0.1     # Adiabicity (note that this is constant i.e. this is a 2D problem)
mu = 1e-4       # Hyper-diffusion coefficient
kappa = 1.0     # Background gradient

# Extraneous params processing
if (Nx != Ny):
    min_N = max(min(Nx, Ny), 128)
    print(f"Currently having problems with non-squared resolutions! Default to {min_N} x {min_N} grid discretization\n")
    Nx, Ny = min_N, min_N

#%%
"""
Grid building (build on CPU first then copy over to GPU if available)
"""
# Physical space grid
x = xp.linspace(x_low, x_high, Nx, endpoint=False)
y = xp.linspace(y_low, y_high, Ny, endpoint=False)
X, Y = xp.meshgrid(x, y, indexing="ij")
 
# Frequency space grid
kx, ky = 2.0*xp.pi*xp.fft.fftfreq(Nx, d=Lx/Nx), 2.0*xp.pi*xp.fft.fftfreq(Ny, d=Ly/Ny)
KX, KY = xp.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2
K4 = K2**2
inv_K2 = xp.where((K2 == 0), 0.0, 1.0/xp.where((K2 == 0), 1.0, K2))
 
# 2/3 Dealiasing mask
kx_max = xp.max(xp.abs(kx))
ky_max = xp.max(xp.abs(ky))
DEALIAS = (
    (xp.abs(KX) <= (2/3)*kx_max) &
    (xp.abs(KY) <= (2/3)*ky_max)
).astype(xp.float64)

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

def initial_vorticity(X,Y,s):
    """
    Initial vorticity profile derived from the Laplacian: vorticity = nabla^{2}(phi)
    """
    return (4.0*(X**2 + Y**2)/s**4 - 4.0/s**2) * xp.exp(-(X**2 + Y**2) / s**2)


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
    return DEALIAS * xp.fft.fft2(
        xp.fft.ifft2(dx(f_hat)).real * xp.fft.ifft2(dy(g_hat)).real - 
        xp.fft.ifft2(dy(f_hat)).real * xp.fft.ifft2(dx(g_hat)).real
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
   return (vorticity_hat + vort_update), (density_hat + dens_update)

#%%
"""
Color map LUTs for rendering
"""
def make_colormap_lut(colormap, n=256) -> xp.ndarray:
    """
    Return RGB colormap Look-Up-Table (LUTs) based Matplotlib's colormaps
    """
    # Accept either a string name or a matplotlib colormap object
    if isinstance(colormap, str):
        cmap = matplotlib.colormaps[colormap]
    else:
        cmap = colormap

    # Normalized RGB scale of the colormap and store on GPU
    RGBA_scale = cmap(np.linspace(0, 1, n))
    return xp.asarray(RGBA_scale.astype(xp.float32))

#%%
"""
GPU Compute-Render Interop config (CUDA for now)
"""
# General metadata for the interop
BYTES_PER_PIXEL = 4*4   # Note float32 is 4 bytes and we are dealing with RGBA

# CUDA error checking in Python
def CUDA_CHECK(result: CUresult, message: str) -> None:
    # Make sure to unwrap the case of tuple returns
    if (isinstance(result, tuple)):
        result = result[0]

    # Actual error checks
    if (result != CUresult.CUDA_SUCCESS):
        raise RuntimeError(f"{message}, ERROR_CODE: {result}")

#%%
"""
Simulation Texture wrapper around context for the 2D grids rendering
"""
class SimulationTexture:
    def __init__(self, ctx: moderngl.Context, Nx: int, Ny:int, cmap_lut: xp.ndarray):
        # Context
        self.ctx = ctx
        self.Nx = Nx
        self.Ny = Ny
        self.cmap_lut = cmap_lut

        # Texture (32-bit floats)
        self.texture = ctx.texture((Nx, Ny), components=4, dtype="f4")
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Just a precaution
        self.texture.repeat_x = False
        self.texture.repeat_y = False

        # Texture buffer
        self.rgba_buffer = xp.empty((Nx * Ny, 4), dtype=xp.float32)

        # Register OpenGL texture with CUDA
        err, self.gl_resource = cuGraphicsGLRegisterImage(
            self.texture.glo,   # OpenGL Texture ID
            GL_TEXTURE_2D,      # 2D target texture
            CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD # CUDA writes only
        )
        CUDA_CHECK(err, "cuGraphicsGLRegisterImage failed")


    def update(self, field: xp.ndarray):
        # Flatten the field's data without making new copy
        f = field.ravel().astype(xp.float32)
        f_max, f_min = f.max(), f.min()

        # Computing corresponding indices in the Look-up-Table
        scale = (self.cmap_lut.shape[0]) - 1
        indices = xp.clip(((f - f_min) / max(f_max - f_min, 1e-16) * scale), 0, scale).astype(xp.uint32)

        # Gathering the LUT values into the buffer
        self.rgba_buffer[:] = self.cmap_lut[indices]

        # Make sure the buffer is flat to hand it off to CUDA
        data = xp.ascontiguousarray(self.rgba_buffer)

        # Hand texture ownership from OpenGL to CUDA
        CUDA_CHECK(cuGraphicsMapResources(1, self.gl_resource, None), "cuGraphicsMapResources failed")

        # Get a writable pointer to the texture's GPU memory
        err, cu_array = cuGraphicsSubResourceGetMappedArray(self.gl_resource, 0, 0)
        CUDA_CHECK(err, "cuGraphicsSubResourceGetMappedArray failed")

        # Setting up memory copy for flat CUDA device memory to tiled GL texture memory
        p = CUDA_MEMCPY2D()
        p.Height = self.Nx
        p.WidthInBytes = self.Ny * BYTES_PER_PIXEL
        p.srcPitch = self.Ny * BYTES_PER_PIXEL
        p.dstArray = cu_array
        p.srcDevice = data.data.ptr
        p.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE
        p.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY
        # Unused fields
        p.dstPitch = 0; p.dstXInBytes = 0; p.dstY = 0
        p.srcXInBytes = 0; p.srcY = 0
        p.srcArray = None; p.srcHost = None; p.dstDevice = None; p.dstHost = None

        # Perform the copy
        CUDA_CHECK(cuMemcpy2D(p), "cuMemcpy2D failed")

        # Hand texture back to OpenGL for sampling render
        CUDA_CHECK(cuGraphicsUnmapResources(1, self.gl_resource, None), "cuGraphicsUnmapResources failed")

        # Return min-max value pairs for analytics
        return f_min, f_max

    def release(self) -> None:
        """
        Texture release wrapper that also unregister the CUDA buffer
        """
        CUDA_CHECK(cuGraphicsUnregisterResource(self.gl_resource), "cuGraphicsUnregisterResource failed")
        self.texture.release()

#%%
"""
ModernGL rendering Window
"""
class SimulationWindow(mglw.WindowConfig):
    # Window data
    resizable = True
    vsync = False # For uncapped FPS
    aspect_ratio = None
    window_size = (1920, 1920/2)
    gl_version = (4, 5)
    resource_dir = (Path(__file__).parent).resolve()

    # Construct the window
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Quad vertex buffer
        quad_vertices = xp.array([
            -1.0, -1.0, 0.0, 0.0,   # Bottom-left
             1.0, -1.0, 1.0, 0.0,   # Bottom-right
            -1.0,  1.0, 0.0, 1.0,   # Top-left
             1.0,  1.0, 1.0, 1.0    # Top-right
        ]).astype(xp.float32)
        
        self.vbo = self.ctx.buffer(quad_vertices.tobytes())

        # Shaders
        self.prog = self.load_program(
            vertex_shader="shaders/turbulence.vert",
            fragment_shader="shaders/turbulence.frag"
        )

        self.quad = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "position_in", "uv_in")]
        )

        # Textures
        self.texture_vort = SimulationTexture(self.ctx, Nx, Ny, make_colormap_lut("jet", n=8192*2))
        self.texture_dens = SimulationTexture(self.ctx, Nx, Ny, make_colormap_lut("viridis", n=8192*2))

        # Initalize simulation states
        self.density_hat = xp.fft.fft2(initial_density(X,Y,s))
        self.vorticity_hat  = xp.fft.fft2(initial_vorticity(X,Y,s))

        # Initialize temporal and analytics data
        self.t = 0.0
        self.step = 0

    def draw(self, texture: SimulationTexture, screen_offset: tuple) -> None:
        texture.texture.use(location=0)
        self.prog["field_texture"] = 0
        self.prog["offset"] = screen_offset
        self.quad.render(moderngl.TRIANGLE_STRIP)

    def close(self) -> None:
        """
        Release texture results on exiting
        """
        self.texture_dens.release()
        self.texture_vort.release()

    def on_render(self, time: float, frametime: float) -> None:
        # Wiping previous screen
        self.ctx.clear(0.0, 0.0, 0.0)

        # Time-step computing until plotting
        for _ in range(plot_interval):
            self.vorticity_hat, self.density_hat = explicit_rk4_step(self.vorticity_hat, self.density_hat)
            self.t += dt
            self.step += 1
        
        # Update the texture based on the physical space values of the fields
        vort_min, vort_max = self.texture_vort.update(xp.fft.ifft2(self.vorticity_hat).real)
        dens_min, dens_max = self.texture_dens.update(xp.fft.ifft2(self.density_hat).real)

        # Draw updated field
        self.draw(self.texture_vort, (0,0)) # VORTICITY on the LEFT
        self.draw(self.texture_dens, (1,0)) # DENSITY on the RIGHT

        # Crude analytics on the title bar
        self.wnd.title = (
            f"Hasegawa-Wakatani Turbulence | Simulation time: {self.t:.3f} | "
            f"Vorticity range: [{vort_min:.3f}, {vort_max:.3f}] | Density range: [{dens_min:.3f}, {dens_max:.3f}] | "
            f"FPS: {self.timer.fps:.1f} | Total runtime: {time:.2f}"
        )

#%%
"""
Main simulation rendering
"""
if __name__ == "__main__":
    mglw.run_window_config(SimulationWindow)