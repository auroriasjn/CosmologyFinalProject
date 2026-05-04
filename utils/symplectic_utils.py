import jax
import jax.numpy as jnp

from .cosmo_utils import H, CosmologicalParameters
from .pm_utils import pm_force
from .jax_utils import _jax_integral
from functools import lru_cache, partial
from jax import jit

@partial(jit, static_argnames=["params", "n_quad"])
def drift_factor(a_1, a_2, params, n_quad=201):
    f_log = lambda log_a: 1.0 / (jnp.exp(log_a)**2 * H(jnp.exp(log_a), params=params))
    return _jax_integral(f_log, jnp.log(a_1), jnp.log(a_2), n_quad)

@partial(jit, static_argnames=["params", "n_quad"])
def kick_factor(a_1, a_2, params, n_quad=201):
    f_log = lambda log_a: 1.0 / H(jnp.exp(log_a), params=params)
    return _jax_integral(f_log, jnp.log(a_1), jnp.log(a_2), n_quad)

@lru_cache(maxsize=None)
def create_symplectic_grid(a_1, a_2, n_steps, params: CosmologicalParameters):
    """Create a symplectic grid for integration."""
    a_values = jnp.linspace(a_1, a_2, n_steps + 1)
    
    drift_factors = [drift_factor(a_values[i], a_values[i + 1], params) for i in range(n_steps)]
    kick_factors = [kick_factor(a_values[i], a_values[i + 1], params) for i in range(n_steps)]
    
    return a_values, drift_factors, kick_factors

@partial(jit, static_argnums=(6,), static_argnames=["params"])
def kdk_step(pos, mom, a_start, a_end, green_x, green_y, green_z,
             Ng, L, params: CosmologicalParameters):
    """
    One KDK leapfrog step from a_start to a_end.

    Parameters
    ----------
    pos : array, shape (N, 3)
        Comoving positions.
    mom : array, shape (N, 3)
        Conjugate momenta (p = m a^2 dx/dt; m=1 here).
    a_start, a_end : float
        Scale factor at start and end of step.
    green_x, green_y, green_z : arrays
        Precomputed Green's functions.

    Returns
    -------
    pos_new, mom_new : arrays
        Updated positions and momenta.
    """
    a_mid = 0.5 * (a_start + a_end)

    # Half kick
    K1 = kick_factor(a_start, a_mid, params=params)
    force = pm_force(pos, a_start, green_x, green_y, green_z, Ng, L, params)
    mom = mom + force * K1

    # Full drift
    D = drift_factor(a_start, a_end, params=params)
    pos = pos + mom * D
    pos = pos % L   # periodic wrapping

    # Half kick
    K2 = kick_factor(a_mid, a_end, params=params)
    force = pm_force(pos, a_end, green_x, green_y, green_z, Ng, L, params)
    mom = mom + force * K2

    return pos, mom

@partial(jit, static_argnums=(4, 8), static_argnames=['params'])
def run_simulation(pos_init, mom_init, a_start, a_end, n_steps,
                   green_x, green_y, green_z, Ng, L, params: CosmologicalParameters):
    """
    Run PM N-body simulation from a_start to a_end.

    Uses equally-spaced steps in ln(a), and avoids redundant force
    evaluations by merging half-kicks. The entire simulation is
    JIT-compiled using jax.lax.fori_loop.

    Parameters
    ----------
    n_steps : int (static)
        Number of timesteps.
    Ng : int (static)
        Grid size per dimension.

    Returns
    -------
    pos, mom : arrays
        Final positions and momenta.
    """
    # Due to new params we have to make this different
    drift_f = partial(drift_factor, params=params)
    kick_f = partial(kick_factor, params=params)
    
    # Steps equally spaced in ln(a)
    a_values = jnp.exp(jnp.linspace(jnp.log(a_start), jnp.log(a_end), n_steps + 1))
    a_mid = 0.5 * (a_values[:-1] + a_values[1:])  # midpoints, length n_steps

    # Precompute all drift factors: D(a_n, a_{n+1})
    D_arr = jax.vmap(drift_f)(a_values[:-1], a_values[1:])

    # Precompute all kick factors for the merged scheme.
    # Step i kicks from a_mid[i] to a_mid[i+1],
    # except the last step kicks from a_mid[-1] to a_values[-1].
    kick_ends = jnp.concatenate([a_mid[1:], a_values[-1:]])
    K_arr = jax.vmap(kick_f)(a_mid, kick_ends)

    # Initial half-kick
    K_init = kick_f(a_values[0], a_mid[0])
    force = pm_force(pos_init, a_values[0], green_x, green_y, green_z, Ng, L, params)
    mom_init = mom_init + force * K_init

    def body(i, carry):
        pos, mom = carry
        # Full drift
        pos = pos + mom * D_arr[i]
        pos = pos % L
        # Force at a_{n+1}, then merged kick
        force = pm_force(pos, a_values[i + 1], green_x, green_y, green_z, Ng, L, params)
        mom = mom + force * K_arr[i]
        return pos, mom

    pos, mom = jax.lax.fori_loop(0, n_steps, body, (pos_init, mom_init))
    return pos, mom
