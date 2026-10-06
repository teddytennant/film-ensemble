"""3D dynamical system and RK4 rollout (Appendix B.1).

State z = (z1, z2, z3):

    dz1/dt = a1 z1 + b z2 z3
    dz2/dt = a2 z2 - b z1 z3
    dz3/dt = -gamma z3 + b z1 z2 + epsilon z1

Reference parameters use gamma = 1. The coarse model is the same vector
field with gamma = 15 (extra damping on z3); epsilon stays 0.05.
"""

from __future__ import annotations

import jax.numpy as jnp


def reference_params():
    """Reference parameters: a1=2, a2=2, b=20, gamma=1, epsilon=0.05."""
    return {
        "a1": jnp.asarray(2.0, dtype=jnp.float32),
        "a2": jnp.asarray(2.0, dtype=jnp.float32),
        "b": jnp.asarray(20.0, dtype=jnp.float32),
        "gamma": jnp.asarray(1.0, dtype=jnp.float32),
        "epsilon": jnp.asarray(0.05, dtype=jnp.float32),
    }


def coarse_params():
    """Coarse model: reference parameters with gamma=15."""
    params = reference_params()
    params["gamma"] = jnp.asarray(15.0, dtype=jnp.float32)
    return params


def _read_params(params):
    """Accept a dict, a sequence (a1, a2, b, gamma, epsilon), or attributes."""
    if isinstance(params, dict):
        return params
    if hasattr(params, "a1"):
        return {
            "a1": params.a1,
            "a2": params.a2,
            "b": params.b,
            "gamma": params.gamma,
            "epsilon": params.epsilon,
        }
    a1, a2, b, gamma, epsilon = params
    return {"a1": a1, "a2": a2, "b": b, "gamma": gamma, "epsilon": epsilon}


def dzdt(z, params):
    """Right-hand side of the 3D system."""
    p = _read_params(params)
    z = jnp.asarray(z, dtype=jnp.float32)
    z1, z2, z3 = z[0], z[1], z[2]
    a1 = jnp.asarray(p["a1"], dtype=jnp.float32)
    a2 = jnp.asarray(p["a2"], dtype=jnp.float32)
    b = jnp.asarray(p["b"], dtype=jnp.float32)
    gamma = jnp.asarray(p["gamma"], dtype=jnp.float32)
    epsilon = jnp.asarray(p["epsilon"], dtype=jnp.float32)
    dz1 = a1 * z1 + b * z2 * z3
    dz2 = a2 * z2 - b * z1 * z3
    dz3 = -gamma * z3 + b * z1 * z2 + epsilon * z1
    return jnp.stack([dz1, dz2, dz3])


def rk4_step(z, params, dt):
    """One classical RK4 step. Returns the updated state, shape (3,)."""
    z = jnp.asarray(z, dtype=jnp.float32)
    dt = jnp.asarray(dt, dtype=jnp.float32)
    k1 = dzdt(z, params)
    k2 = dzdt(z + 0.5 * dt * k1, params)
    k3 = dzdt(z + 0.5 * dt * k2, params)
    k4 = dzdt(z + dt * k3, params)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(z0, params, steps, dt):
    """Integrate `steps` RK4 steps.

    Returns the state after each step, shape (steps, 3). The initial state
    is not included; ``rollout(...)[-1]`` is the state at t = steps * dt.
    """
    import jax

    z0 = jnp.asarray(z0, dtype=jnp.float32)
    dt = jnp.asarray(dt, dtype=jnp.float32)
    steps = int(steps)

    def body(z, _):
        z_next = rk4_step(z, params, dt)
        return z_next, z_next

    _, traj = jax.lax.scan(body, z0, xs=None, length=steps)
    return traj
