import jax.numpy as jnp

def _jax_integral(f, a_1, a_2, n_quad):
    # Simpson's rule needs an odd number of points
    if n_quad % 2 == 0:
        raise ValueError("n_quad must be odd for Simpson's rule.")

    x = jnp.linspace(a_1, a_2, n_quad)
    y = f(x)
    h = (a_2 - a_1) / (n_quad - 1)

    return h / 3.0 * (
        y[0]
        + y[-1]
        + 4.0 * jnp.sum(y[1:-1:2], axis=0)
        + 2.0 * jnp.sum(y[2:-2:2], axis=0)
    )