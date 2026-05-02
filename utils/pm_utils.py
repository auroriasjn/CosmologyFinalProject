import jax
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
from .cosmo_utils import CosmologicalParameters, D_plus, H

from functools import partial
from jax import jit, grad
import jax.numpy as jnp


@partial(jit, static_argnums=(1,))
def cic_deposit(positions, Ng: int, L: float, values=None):
    """
    Deposit particle quantities onto a grid using CIC.

    Parameters
    ----------
    positions : array, shape (Np, 3)
        Particle positions in [0, L).
    Ng : int
        Grid size per dimension.
    L : float
        Box size.
    values : None or array, shape (Np,)
        If None, deposit unit particle weights.
        If array, deposit values[p] carried by each particle.

    Returns
    -------
    grid : array, shape (Ng, Ng, Ng)
        CIC-deposited field.
    """
    h = L / Ng
    positions = jnp.mod(positions, L)

    cell = positions / h - 0.5
    cell_index = jnp.floor(cell).astype(jnp.int32)
    delta = cell - cell_index

    weights = jnp.stack([1.0 - delta, delta], axis=-1)

    if values is None:
        values = jnp.ones(positions.shape[0], dtype=positions.dtype)

    grid = jnp.zeros((Ng, Ng, Ng), dtype=values.dtype)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                weight = (
                    weights[:, 0, i]
                    * weights[:, 1, j]
                    * weights[:, 2, k]
                )

                ix = (cell_index[:, 0] + i) % Ng
                iy = (cell_index[:, 1] + j) % Ng
                iz = (cell_index[:, 2] + k) % Ng

                grid = grid.at[ix, iy, iz].add(values * weight)

    return grid


def cic_interpolate(grid, positions, L: float):
    """Interpolate grid values at given positions using CIC."""
    Ng = grid.shape[0]
    h = L / Ng
    
    # Wrap positions into [0, L)
    positions = jnp.mod(positions, L)

    # Grid index of lower-left corner and fractional offset
    cell = (positions / h - 0.5)
    cell_index = jnp.floor(cell).astype(int)
    delta = cell - cell_index # fractional distance

    # CIC weights: 1 - dx for the lower grid point, dx for the upper
    weights = jnp.stack([1 - delta, delta], axis=-1)

    # Build the 2^3 = 8 contributions
    interpolated_values = jnp.zeros(positions.shape[0])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                weight = weights[:, 0, i] * weights[:, 1, j] * weights[:, 2, k]
                ix = (cell_index[:, 0] + i) % Ng
                iy = (cell_index[:, 1] + j) % Ng
                iz = (cell_index[:, 2] + k) % Ng
                interpolated_values += weight * grid[ix, iy, iz]

    return interpolated_values


@partial(jit, static_argnums=(3,))
def compute_velocity_divergence(pos, mom, a_snap, Ng: int, L: float, eps=1e-10):
    """
    Compute the real-space velocity divergence using exact spectral derivatives.
    Includes CIC window deconvolution to recover high-k modes.
    """
    # 1. CIC Deposit (Real Space)
    rho = cic_deposit(pos, Ng, L)
    mom_x = cic_deposit(pos, Ng, L, values=mom[:, 0])
    mom_y = cic_deposit(pos, Ng, L, values=mom[:, 1])
    mom_z = cic_deposit(pos, Ng, L, values=mom[:, 2])

    # Calculate mass-weighted peculiar velocity
    vel_x = (mom_x / (rho + eps)) / a_snap
    vel_y = (mom_y / (rho + eps)) / a_snap
    vel_z = (mom_z / (rho + eps)) / a_snap

    # 2. Forward FFT to Fourier Space
    vx_hat = jnp.fft.fftn(vel_x)
    vy_hat = jnp.fft.fftn(vel_y)
    vz_hat = jnp.fft.fftn(vel_z)

    # 3. Setup k-grid for spectral derivative
    k_1d = jnp.fft.fftfreq(Ng, d=L/Ng) * 2.0 * jnp.pi
    kx, ky, kz = jnp.meshgrid(k_1d, k_1d, k_1d, indexing='ij')

    # 4. Exact Spectral Divergence
    # theta(k) = i * (kx*vx + ky*vy + kz*vz)
    theta_hat = 1j * (kx * vx_hat + ky * vy_hat + kz * vz_hat)

    # 5. Deconvolve CIC Window Function
    # jnp.sinc calculates sin(pi*x)/(pi*x), so we divide by pi
    dx = L / Ng
    sinc_x = jnp.sinc(kx * dx / (2.0 * jnp.pi)) 
    sinc_y = jnp.sinc(ky * dx / (2.0 * jnp.pi))
    sinc_z = jnp.sinc(kz * dx / (2.0 * jnp.pi))
    
    # The CIC window in power is sinc^4, but for the field amplitude it's sinc^2
    window_function = (sinc_x * sinc_y * sinc_z)**2
    
    # Correct the field (adding eps to avoid zero division at Nyquist)
    theta_hat_corrected = theta_hat / (window_function + 1e-10)

    # 6. Inverse FFT back to Real Space
    # Returns a real-space array so measure_pk can ingest it naturally!
    theta_real = jnp.fft.ifftn(theta_hat_corrected).real

    return theta_real

    
def make_uniform_grid(Np: int, L: float):
    """Create a uniform grid of Np^3 particles in a box of side L."""
    x = jnp.linspace(0, L, Np, endpoint=False) + L / (2 * Np)
    grid = jnp.meshgrid(x, x, x, indexing='ij')
    pos = jnp.stack([g.ravel() for g in grid], axis=-1)
    return pos


def make_green_function(Ng: int, L: float):
    """
    Precompute the Green's function for the PM force calculation.

    Returns the gradient of the inverse Laplacian in Fourier space:
    G_i(k) = D_i(k) / L(k), with discrete operators.

    Returns
    -------
    green_x, green_y, green_z : arrays, shape (Ng, Ng, Ng)
        Fourier-space Green's function for each force component.
    """
    h = L / Ng

    # Wavenumber indices
    k = jnp.fft.fftfreq(Ng, d=1.0/Ng)  # integer wavenumber indices 0..Ng/2..-1
    kx, ky, kz = jnp.meshgrid(k, k, k, indexing='ij')

    # Discrete Laplacian: -sum_i (4/h^2) sin^2(pi k_i / Ng)
    laplacian = -(4.0 / h**2) * (
        jnp.sin(jnp.pi * kx / Ng)**2 +
        jnp.sin(jnp.pi * ky / Ng)**2 +
        jnp.sin(jnp.pi * kz / Ng)**2
    )

    # Avoid division by zero at k=0
    laplacian = laplacian.at[0, 0, 0].set(1.0)

    # Discrete gradient: (i/h) sin(2 pi k_i / Ng)
    grad_x = 1j / h * jnp.sin(2.0 * jnp.pi * kx / Ng)
    grad_y = 1j / h * jnp.sin(2.0 * jnp.pi * ky / Ng)
    grad_z = 1j / h * jnp.sin(2.0 * jnp.pi * kz / Ng)

    # Green's function: G_i = D_i / L
    green_x = grad_x / laplacian
    green_y = grad_y / laplacian
    green_z = grad_z / laplacian

    # Zero the k=0 mode (no mean force)
    green_x = green_x.at[0, 0, 0].set(0.0)
    green_y = green_y.at[0, 0, 0].set(0.0)
    green_z = green_z.at[0, 0, 0].set(0.0)

    return green_x, green_y, green_z

@partial(jit, static_argnums=(5, 7))
def pm_force(pos, a, green_x, green_y, green_z, Ng: int, L: float, params: CosmologicalParameters):
    """
    Compute -grad(Phi) at each particle position via the PM method.

    The kick step is: dp = -grad(Phi) * K(a1, a2).
    This function returns -grad(Phi), including the Poisson prefactor
    (3/2) Omega_m / a.  The caller multiplies by the kick factor.

    Parameters
    ----------
    pos : array, shape (N, 3)
        Particle positions in [0, L).
    a : float
        Scale factor.
    green_x, green_y, green_z : arrays
        Precomputed Green's functions.
    Ng : int
        Grid size per dimension.
    L : float
        Box size.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    force : array, shape (N, 3)
        -grad(Phi) at each particle position.
    """
    # 1. CIC deposit
    rho = cic_deposit(pos, Ng, L)

    # 2. Overdensity: delta = rho / rho_bar - 1, where rho_bar = N / Ng^3
    rho_bar = pos.shape[0] / Ng**3
    delta = rho / rho_bar - 1.0

    # 3. Forward FFT
    delta_hat = jnp.fft.fftn(delta)

    # 4. Poisson prefactor: (3/2) Omega_m / a  (in H_0=1 units)
    prefactor = 1.5 * (params.H0 ** 2 * params.Omega_m) / a

    # 5. Force in Fourier space: -prefactor * G_i * delta_hat
    fx_hat = -prefactor * green_x * delta_hat
    fy_hat = -prefactor * green_y * delta_hat
    fz_hat = -prefactor * green_z * delta_hat

    # 6. Inverse FFT (force fields are real)
    fx = jnp.fft.ifftn(fx_hat).real
    fy = jnp.fft.ifftn(fy_hat).real
    fz = jnp.fft.ifftn(fz_hat).real

    # 7. CIC interpolate to particle positions
    ax = cic_interpolate(fx, pos, L)
    ay = cic_interpolate(fy, pos, L)
    az = cic_interpolate(fz, pos, L)

    return jnp.stack([ax, ay, az], axis=-1)


def make_cosmo_ic(Np, Ng, L, pk_func, a_init, params: CosmologicalParameters,seed=42):
    """
    Generate cosmological Zel'dovich ICs from a power spectrum.

    Parameters
    ----------
    Np : int
        Particles per dimension.
    Ng : int
        Grid cells per dimension.
    L : float
        Box size (same units as P(k), e.g. Mpc/h).
    pk_func : callable
        P(k) function, k in h/Mpc, returns (Mpc/h)^3.
    a_init : float
        Initial scale factor.
    params : CosmologicalParameters
        Cosmological parameters.
    seed : int
        Random seed.

    Returns
    -------
    pos, mom, q : arrays
        Positions, momenta, and Lagrangian grid.
    """
    h = L / Ng
    V = L**3

    # Wavenumber grid (angular wavenumber k in h/Mpc)
    # fftfreq returns frequency nu = n/(N*h); multiply by 2pi for k
    kfreq = 2 * np.pi * np.fft.fftfreq(Ng, d=h)   # k in h/Mpc
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    kmag[0, 0, 0] = 1.0  # avoid division by zero

    # Power spectrum on the grid
    pk_grid = pk_func(kmag)
    pk_grid[0, 0, 0] = 0.0  # no mean overdensity

    # Generate Gaussian random field by "coloring" white noise.
    # Start with real-space white noise, FFT it (reality condition is
    # automatic since the input is real), then multiply by sqrt(P(k))
    # to imprint the power spectrum.
    #
    # Normalization: E[|noise_hat_k|^2] = Ng^3 for white noise.
    # We want E[|delta_hat_k|^2] = Ng^6 P(k) / V, so multiply
    # noise_hat by Ng^{3/2} sqrt(P(k)/V).
    rng = np.random.default_rng(seed)
    white_noise = rng.standard_normal((Ng, Ng, Ng))
    noise_hat = np.fft.fftn(white_noise)

    amplitude = Ng**1.5 * np.sqrt(pk_grid / V)
    delta_hat = amplitude * noise_hat
    delta_hat[0, 0, 0] = 0.0  # no mean overdensity

    # Displacement field: Psi_i(k) = -i k_i / k^2 * delta_hat(k)
    inv_k2 = 1.0 / (kmag**2)
    inv_k2[0, 0, 0] = 0.0

    psi_hat_x = -1j * kx * inv_k2 * delta_hat
    psi_hat_y = -1j * ky * inv_k2 * delta_hat
    psi_hat_z = -1j * kz * inv_k2 * delta_hat

    # Transform to real space
    psi_x = np.fft.ifftn(psi_hat_x).real
    psi_y = np.fft.ifftn(psi_hat_y).real
    psi_z = np.fft.ifftn(psi_hat_z).real

    # Lagrangian grid
    q = make_uniform_grid(Np, L)
    q_np = np.array(q)

    # Interpolate displacement to particle positions
    # For Np = Ng, particles sit at grid centers — direct indexing
    idx = np.round(q_np / h - 0.5).astype(int) % Ng

    psi_at_q = np.stack([
        psi_x[idx[:, 0], idx[:, 1], idx[:, 2]],
        psi_y[idx[:, 0], idx[:, 1], idx[:, 2]],
        psi_z[idx[:, 0], idx[:, 1], idx[:, 2]],
    ], axis=-1)

    # Zel'dovich: x = q + D_+(a) * Psi, p = a^{3/2} * Psi  (EdS)
    D_a = D_plus(a_init, params)
    H_a = H(a_init, params)

    # 2. Get the gradient of H(a) - this is perfectly safe!
    dH_da = grad(H, argnums=0)(a_init, params)

    # 3. Calculate dD/da analytically
    dD_da = (dH_da / H_a) * D_a + (2.5 * params.Omega_m) / (a_init**3 * H_a**2)

    # 4. Apply general Zel'dovich equations
    pos = jnp.array((q_np + D_a * psi_at_q) % L)
    mom = jnp.array((a_init**3 * H_a * dD_da) * psi_at_q)

    # Also return the z=0 linear density field (delta_hat is at z=0)
    delta0 = np.fft.ifftn(delta_hat).real

    return pos, mom, q, delta0


def make_zeldovich_ic(Np, L, n_mode, A, a_init, params: CosmologicalParameters):
    """
    Create Zel'dovich initial conditions for a single sine-wave mode.
    Now generalized for any cosmology.
    """
    # Lagrangian grid
    q = make_uniform_grid(Np, L)
    k = 2 * jnp.pi * n_mode / L

    # Displacement field: Psi_x = -(A/k) sin(k q_x)
    psi_x = -(A / k) * jnp.sin(k * q[:, 0])

    # Cosmology parameters at a_init
    D_a = D_plus(a_init, params)
    H_a = H(a_init, params)
    
    dH_da = grad(H, argnums=0)(a_init, params)
    dD_da = (dH_da / H_a) * D_a + (2.5 * params.Omega_m) / (a_init**3 * H_a**2)

    # Positions: x = q + D_+(a) * Psi
    pos = q.at[:, 0].add(D_a * psi_x)
    pos = pos % L  # periodic wrapping

    # Momenta: p = a^3 * H(a) * dD_+/da * Psi
    mom = jnp.zeros_like(q)
    mom = mom.at[:, 0].set((a_init**3 * H_a * dD_da) * psi_x)

    return pos, mom, q


def zeldovich_prediction(q, n_mode, A, a, L, params: CosmologicalParameters):
    """
    Zel'dovich position and momentum at scale factor a.
    Now generalized for any cosmology.
    """
    k = 2 * jnp.pi * n_mode / L
    psi_x = -(A / k) * jnp.sin(k * q[:, 0])

    # Cosmology parameters at current scale factor 'a'
    D_a = D_plus(a, params)
    H_a = H(a, params)
    
    dH_da = grad(H, argnums=0)(a, params)
    dD_da = (dH_da / H_a) * D_a + (2.5 * params.Omega_m) / (a**3 * H_a**2)

    # Exact linear positions: x = q + D_+(a) * Psi
    pos_exact = q.at[:, 0].add(D_a * psi_x) % L
    
    # Exact linear momenta: p = a^3 * H(a) * dD_+/da * Psi
    mom_exact = jnp.zeros_like(q)
    mom_exact = mom_exact.at[:, 0].set((a**3 * H_a * dD_da) * psi_x)

    return pos_exact, mom_exact