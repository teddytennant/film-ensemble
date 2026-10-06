"""Window losses (section 3.2) and one SGD step.

L = L_path + 0.01 L_delta + 0.1 L_mom + 0.1 L_tail.

These lambdas are not copied from the paper. They are not stated as a
single number in the appendix slice used here, and they are small so the
path term dominates.

L_tail is a stand-in for the paper's exceedance-weighted tail term:
w = 1 + relu(|z_ref| - tau), with a fixed threshold tau (default 1), and
L_tail = mean(w * (z_hat - z_ref)^2). Errors on large |z_ref| are
upweighted. This is not a copied hyperparameter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from film_ensemble.model import apply


LAM_DELTA = 0.01
LAM_MOM = 0.1
LAM_TAIL = 0.1
DEFAULT_TAU = 1.0
DEFAULT_LR = 1e-2


def path_loss(z_hat, z_ref):
    """L_path = mean((z_hat - z_ref)^2)."""
    return jnp.mean((z_hat - z_ref) ** 2)


def delta_loss(delta):
    """L_delta = mean(delta^2). Depends only on the pre-gate correction."""
    return jnp.mean(delta ** 2)


def momentum_loss(z_hat, z_ref):
    """Squared difference of window means, summed over the 3 coordinates."""
    diff = jnp.mean(z_hat, axis=0) - jnp.mean(z_ref, axis=0)
    return jnp.sum(diff ** 2)


def tail_loss(z_hat, z_ref, quantile=DEFAULT_TAU):
    """Exceedance-weighted path error. ``quantile`` is the fixed threshold tau."""
    weight = 1.0 + jax.nn.relu(jnp.abs(z_ref) - quantile)
    return jnp.mean(weight * (z_hat - z_ref) ** 2)


def total_loss(z_hat, z_ref, delta, quantile=DEFAULT_TAU):
    """Return (L, parts) with parts path, delta, mom, tail."""
    parts = {
        "path": path_loss(z_hat, z_ref),
        "delta": delta_loss(delta),
        "mom": momentum_loss(z_hat, z_ref),
        "tail": tail_loss(z_hat, z_ref, quantile),
    }
    total = (
        parts["path"]
        + LAM_DELTA * parts["delta"]
        + LAM_MOM * parts["mom"]
        + LAM_TAIL * parts["tail"]
    )
    return total, parts


def loss_and_parts(params, x, z_ref, context, quantile=DEFAULT_TAU):
    out = apply(params, x, context)
    total, parts = total_loss(out.z_hat, z_ref, out.delta, quantile)
    return total, parts


def sgd_step(params, grads, lr):
    """Vanilla SGD: p <- p - lr * g."""
    lr = jnp.asarray(lr, dtype=jnp.float32)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def train_step(params, x, z_ref, context, lr=DEFAULT_LR, quantile=DEFAULT_TAU):
    """One SGD step. Returns new params, loss, grads, and loss parts.

    ``context`` must be a length-8 vector. FiLM parameters are used, so
    they receive gradient when that context is nonzero and the path error
    depends on the modulation.
    """

    def loss_fn(p):
        return loss_and_parts(p, x, z_ref, context, quantile)

    (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    new_params = sgd_step(params, grads, lr)
    return new_params, loss, grads, parts


def train_steps(params, x, z_ref, context, n_steps, lr=DEFAULT_LR, quantile=DEFAULT_TAU):
    """Run ``n_steps`` SGD updates. Returns params, losses (n_steps,), last grads."""

    def body(carry, _):
        p = carry
        p, loss, grads, _parts = train_step(p, x, z_ref, context, lr, quantile)
        return p, (loss, grads)

    params, (losses, grads) = jax.lax.scan(body, params, xs=None, length=int(n_steps))
    return params, losses, grads
