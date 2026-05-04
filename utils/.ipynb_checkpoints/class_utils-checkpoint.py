import jax
import numpy as np
import jax.numpy as jnp
from functools import cache
from scipy.interpolate import interp1d
from classy import Class

def create_cosmo_instance(
        h: float = 0.7,
        omega_b: float = 0.049,
        omega_cdm: float = 0.268,
        A_s: float = 2.1e-9,
        n_s: float = 0.9649,
        max_k: float = 50.0,
        max_z: float = 0.0
) -> Class:
    """Create a CLASS instance with specified cosmological parameters."""
    cosmo = Class()

    cosmo.set({
        'output': 'mPk',
        'h': h,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm,
        'A_s': A_s,
        'n_s': n_s,
        'P_k_max_h/Mpc': max_k,
        'z_max_pk': max_z
    })

    cosmo.compute()
    return cosmo

@cache
def _build_pk_interpolator(cosmo, z: float):
    """Hidden helper that caches the interpolator based on cosmo and redshift."""
    k_arr, pk_arr = create_power_spectrum(cosmo, z=z) 
    
    return interp1d(
        np.log(k_arr), 
        np.log(pk_arr), 
        kind='cubic', 
        fill_value=-100.0, 
        bounds_error=False
    )

def pk_of_k(k, cosmo: Class, z: float = 0.0):
    """Helper function to compute P(k) for a single k at a specific redshift."""
    # The cache is now keyed by both the cosmo object AND the redshift
    log_pk_interp = _build_pk_interpolator(cosmo, z)
    return np.exp(log_pk_interp(np.log(k)))

def create_power_spectrum(cosmo: Class, z: float = 0.0, k_min: float = 1e-4, k_max: float = 10.0, n_k: int = 1000):
    """Create the linear matter power spectrum P(k) from the CLASS instance."""

    k_values = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    P_k = np.array([cosmo.pk(k, z) for k in k_values])
    
    return k_values, P_k

def pk_of_k(k, cosmo: Class, z: float = 0.0):
    """Helper function to compute P(k) for a single k."""
    log_pk_interp = _build_pk_interpolator(cosmo=cosmo, z=z)
    return np.exp(log_pk_interp(np.log(k)))