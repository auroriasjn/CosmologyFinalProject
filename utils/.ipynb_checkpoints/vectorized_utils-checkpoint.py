import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial, cache
from scipy.interpolate import interp1d
from utils.pm_utils import cic_deposit

cic_deposit_jit = jit(cic_deposit, static_argnums=(1,))
def deposit_to_delta(pos, Ng, L):
    """Deposit particles and return the overdensity field delta as a numpy array."""
    rho = cic_deposit_jit(pos, Ng, L)
    rho_bar = pos.shape[0] / Ng**3
    return np.array(rho / rho_bar - 1.0)

@cache
def make_k_grid(Ng, L, n_mu_bins=10):
    """
    Precompute wavenumber grid, bin assignments, and mu for P(k, mu) measurement.

    Returns a dict with:
        kmag       : (Ng, Ng, Ng) array of |k|
        mu         : (Ng, Ng, Ng) array of k_z / |k| (LOS along z)
        ik         : (Ng^3,) int array of radial k-bin index (flattened)
        imu        : (Ng^3,) int array of |mu| bin index (flattened)
        k_centers  : (n_k_bins,) radial bin centers
        mu_centers : (n_mu_bins,) |mu| bin centers
        n_k_bins   : number of radial bins
        n_mu_bins  : number of |mu| bins
        norm       : h^6 / V prefactor for P(k) normalization
    """
    h = L / Ng
    V = L**3
    kfreq = 2 * jnp.pi * jnp.fft.fftfreq(Ng, d=h)
    kx, ky, kz = jnp.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    kmag = jnp.sqrt(kx**2 + ky**2 + kz**2)

    # mu = k_z / |k| (line of sight along z-axis)
    kmag_safe = jnp.where(kmag == 0, 1.0, kmag)
    mu = kz / kmag_safe
    mu = jnp.where(kmag == 0, 0.0, mu)

    # Radial k bins: k_fund/2 to k_Nyquist, spaced by k_fund
    k_fund = 2 * jnp.pi / L
    k_nyq = jnp.pi * Ng / L
    k_edges = jnp.arange(k_fund / 2, k_nyq + k_fund, k_fund)
    n_k_bins = len(k_edges) - 1
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    # |mu| bins: 0 to 1 (symmetric in mu, so bin |mu|)
    mu_edges = jnp.linspace(0, 1, n_mu_bins + 1)
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])

    # Bin assignments (flattened)
    ik = jnp.digitize(kmag.ravel(), k_edges) - 1
    imu = jnp.digitize(jnp.abs(mu).ravel(), mu_edges) - 1
    imu = jnp.clip(imu, 0, n_mu_bins - 1)  # |mu|=1 edge case

    # Effective k per bin: mean |k| of modes in each bin
    kmag_flat = kmag.ravel()
    k_sum = jnp.zeros(n_k_bins).at[ik].add(kmag_flat, mode='drop')
    k_count = jnp.zeros(n_k_bins).at[ik].add(jnp.ones_like(kmag_flat), mode='drop')
    k_eff = jnp.where(k_count > 0, k_sum / k_count, k_centers)

    return {
        'ik': ik, 'imu': imu,
        'k_eff': k_eff, 'k_centers': k_centers, 'mu_centers': mu_centers,
        'k_edges': k_edges, 'mu_edges': mu_edges,
        'n_k_bins': n_k_bins, 'n_mu_bins': n_mu_bins,
        'norm': h**6 / V,
    }

@partial(jax.jit, static_argnums=(2,))
def _bin_power(power_flat, ik, n_k):
    """Scatter-add power into radial k-bins."""
    # mode='drop' silently ignores out-of-range indices (k=0, k>k_Ny)
    pk_sum = jnp.zeros(n_k).at[ik].add(power_flat, mode='drop')
    counts = jnp.zeros(n_k).at[ik].add(jnp.ones_like(power_flat), mode='drop')
    pk = jnp.where(counts > 0, pk_sum / counts, 0.0)
    return pk, counts

def measure_pk(delta, kgrid):
    """
    Measure the isotropic power spectrum P(k) from a 3D density field.

    Uses the DFT convention from Lecture 20:
        delta_hat_continuous(k) = h^3 * delta_hat_DFT(k)
        P(k) = (1/V) * |delta_hat_continuous(k)|^2

    Parameters
    ----------
    delta : array, shape (Ng, Ng, Ng)
        Overdensity field.
    kgrid : dict
        Output of make_k_grid.

    Returns
    -------
    k_centers : array — bin centers in h/Mpc.
    pk : array — power spectrum in (Mpc/h)^3.
    counts : array — number of modes per bin.
    """
    delta_hat = jnp.fft.fftn(jnp.asarray(delta))
    power_flat = jnp.abs(delta_hat).ravel()**2 * kgrid['norm']
    pk, counts = _bin_power(power_flat, kgrid['ik'], kgrid['n_k_bins'])
    return kgrid['k_eff'], pk, counts
