"""Short CPU demo: 200-step rollouts, ensemble eigenvalue, 10 SGD steps."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from film_ensemble.ensemble import context_mean_std_eigs
from film_ensemble.model import init_params
from film_ensemble.system import coarse_params, reference_params, rollout
from film_ensemble.train import train_step


def main():
    z0 = jnp.array([0.1, 0.1, 0.1], dtype=jnp.float32)
    dt = 0.01
    steps = 200
    ref_params = reference_params()
    coarse = coarse_params()

    ref = rollout(z0, ref_params, steps, dt)
    coarse_traj = rollout(z0, coarse, steps, dt)

    eps = 1e-3
    directions = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, -1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True) * eps
    member_z0 = z0[None, :] + directions

    def one_member(z):
        return rollout(z, ref_params, steps, dt)

    members = jax.vmap(one_member)(member_z0)  # (M, T, 3)
    ensemble = jnp.swapaxes(members, 0, 1)  # (T, M, 3)
    context_last = context_mean_std_eigs(ensemble[-1])
    leading = context_last[6]
    print(f"leading_eigenvalue {float(leading):.8e}")

    window = 8
    x = coarse_traj[:window]
    z_ref = ref[:window]
    context = context_mean_std_eigs(ensemble[window - 1])

    params = init_params(jax.random.PRNGKey(0))
    losses = []
    for _ in range(10):
        params, loss, _grads, _parts = train_step(params, x, z_ref, context, lr=1e-2)
        losses.append(float(loss))
    print(f"loss_first {losses[0]:.8e}")
    print(f"loss_last {losses[-1]:.8e}")


if __name__ == "__main__":
    main()
