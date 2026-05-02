from dataclasses import dataclass, field
from .jax_utils import _jax_integral
import jax.numpy as jnp

"""
This module defines cosmological parameters and functions for computing the Hubble parameter and growth factor in a general cosmology. 
It also includes utilities for computing symplectic integration factors (drift and kick) that depend on the cosmology. 
The CosmologicalParameters class is designed to be immutable and hashable, making it suitable for caching results of expensive computations.
"""

@dataclass(frozen=True)
class GeneralCosmologicalParameters:
    H0: float = 1.0           # Hubble constant at present time
    Omega_m: float = 0.3      # Matter density parameter
    Omega_Lambda: float = 0.7 # Dark energy density parameter
    Omega_k: float = 0.0      # Curvature density parameter
    Omega_r: float = 0.0      # Radiation density parameter - usually negligible at late times

@dataclass(frozen=True)
class CosmologicalParameters(GeneralCosmologicalParameters):
    """
    Cosmological parameters where Omega_k is derived from
    Omega_m and Omega_Lambda.
    """

    Omega_k: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "Omega_k",
            1.0 - self.Omega_m - self.Omega_Lambda,
        )

def H(a, params: CosmologicalParameters):
    """Calculate the Hubble parameter at scale factor a."""
    return params.H0 * jnp.sqrt((params.Omega_m / a**3) + (params.Omega_r / a**4) + params.Omega_Lambda + (params.Omega_k / a**2))

def D_plus(a, params: CosmologicalParameters, a_min: float = 1e-8):
    """Calculate the linear growth factor D+(a) for the given cosmology."""
    # Integral for D+(a) in a general cosmology
    def integrand(a_prime):
        return 1.0 / (a_prime**3 * H(a_prime, params)**3)

    integral = _jax_integral(integrand, a_min, a, 1001)
    return (5.0 * params.Omega_m / 2.0) * H(a, params) * integral


