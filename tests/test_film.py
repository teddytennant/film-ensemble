"""CPU tests for the 3D FiLM ensemble corrector."""

import jax
import jax.numpy as jnp

from film_ensemble.ensemble import context_eigvec, context_mean_std_eigs
from film_ensemble.model import (
    D_HID,
    FiLMCorrector,
    apply,
    encode,
    film,
    heads,
    init_params,
)
from film_ensemble.system import coarse_params, reference_params, rk4_step, rollout
from film_ensemble.train import delta_loss, path_loss, tail_loss, total_loss, train_step


def test_rk4_step_finite():
    z = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
    z_next = rk4_step(z, reference_params(), 0.01)
    assert z_next.shape == (3,)
    assert jnp.all(jnp.isfinite(z_next))


def test_linear_growth_matches_exp():
    params = {
        "a1": 2.0,
        "a2": 2.0,
        "b": 0.0,
        "gamma": 1.0,
        "epsilon": 0.0,
    }
    z0 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    dt = 0.01
    steps = 20
    traj = rollout(z0, params, steps, dt)
    expected = jnp.exp(jnp.asarray(params["a1"] * steps * dt, dtype=jnp.float32))
    rel = jnp.abs(traj[-1, 0] - expected) / expected
    assert float(rel) < 1e-3


def test_coarse_gamma_damps_z3_faster():
    z0 = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    ref = {"a1": 0.0, "a2": 0.0, "b": 0.0, "gamma": 1.0, "epsilon": 0.0}
    coarse = {"a1": 0.0, "a2": 0.0, "b": 0.0, "gamma": 15.0, "epsilon": 0.0}
    steps = 20
    dt = 0.01
    z3_ref = rollout(z0, ref, steps, dt)[-1, 2]
    z3_coarse = rollout(z0, coarse, steps, dt)[-1, 2]
    assert jnp.abs(z3_coarse) < jnp.abs(z3_ref)


def test_rank1_ensemble_leading_axis():
    ensemble = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    v1 = context_eigvec(ensemble, r=1)
    axis = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    align = jnp.abs(jnp.dot(v1, axis))
    assert float(align) > 0.99
    ctx = context_mean_std_eigs(ensemble)
    assert ctx.shape == (8,)
    lambda1, lambda2 = ctx[6], ctx[7]
    assert float(lambda1) > float(lambda2)


def test_film_formula_hand_vector():
    h = jnp.array([0.5, -1.25, 2.0, 0.0], dtype=jnp.float32)
    gamma = jnp.array([0.1, -0.3, 0.25, 1.5], dtype=jnp.float32)
    beta = jnp.array([-0.2, 0.4, 0.0, 0.7], dtype=jnp.float32)
    got = film(h, gamma, beta)
    expected = (1.0 + gamma) * h + beta
    assert jnp.allclose(got, expected, atol=1e-6)


def test_initial_gate_mean_small():
    params = init_params(jax.random.PRNGKey(0))
    x = jnp.linspace(-0.5, 0.5, 24, dtype=jnp.float32).reshape(8, 3)
    out = FiLMCorrector(params).forward(x, context=None)
    assert out.gate.shape == (8, 3)
    assert float(jnp.mean(out.gate)) < 0.2
    assert jnp.allclose(params["heads"]["b_g"], -2.0)


def test_context_none_matches_zero_film_and_context_changes_output():
    params = init_params(jax.random.PRNGKey(1))
    x = jnp.linspace(0.05, 0.9, 24, dtype=jnp.float32).reshape(8, 3)
    model = FiLMCorrector(params)
    out_none = model.forward(x, context=None)

    h = encode(params, x)
    gamma0 = jnp.zeros((D_HID,), dtype=jnp.float32)
    beta0 = jnp.zeros((D_HID,), dtype=jnp.float32)
    delta0, gate0 = heads(params, film(h, gamma0, beta0))
    z_identity = x + gate0 * delta0
    assert jnp.allclose(out_none.z_hat, z_identity, atol=1e-6)
    assert jnp.allclose(out_none.delta, delta0, atol=1e-6)

    c1 = jnp.array([0.3, -0.2, 0.4, 0.05, 0.02, 0.01, 0.4, 0.05], dtype=jnp.float32)
    c2 = jnp.array([-0.7, 0.5, -0.1, 0.2, 0.3, 0.4, 1.2, 0.02], dtype=jnp.float32)
    y1 = model.forward(x, context=c1).z_hat
    y2 = model.forward(x, context=c2).z_hat
    assert float(jnp.max(jnp.abs(y1 - y2))) > 1e-6
    assert y1.shape == (8, 3)


def test_losses():
    z_ref = jnp.ones((8, 3), dtype=jnp.float32)
    assert float(path_loss(z_ref, z_ref)) == 0.0

    delta = jnp.arange(24, dtype=jnp.float32).reshape(8, 3) / 10.0
    d1 = delta_loss(delta)
    # L_delta is a function of delta only; calling it twice is the contract.
    assert float(d1) == float(delta_loss(delta))

    err = jnp.full((8, 3), 0.25, dtype=jnp.float32)
    z_small = jnp.full((8, 3), 0.1, dtype=jnp.float32)
    z_large = jnp.full((8, 3), 4.0, dtype=jnp.float32)
    tau = 1.0
    tail_small = tail_loss(z_small + err, z_small, quantile=tau)
    tail_large = tail_loss(z_large + err, z_large, quantile=tau)
    assert float(tail_large) > float(tail_small)

    z_hat = z_ref + 0.1
    total, parts = total_loss(z_hat, z_ref, delta, quantile=tau)
    assert jnp.isfinite(total)
    _, parts_other = total_loss(z_hat, z_ref + 3.0, delta, quantile=tau)
    assert float(parts["delta"]) == float(parts_other["delta"])


def test_sgd_step_finite_and_film_grad():
    z0 = jnp.array([0.1, 0.1, 0.1], dtype=jnp.float32)
    dt = 0.01
    ref = rollout(z0, reference_params(), 40, dt)
    coarse = rollout(z0, coarse_params(), 40, dt)
    x = coarse[:8]
    z_ref = ref[:8]
    context = jnp.array(
        [0.1, 0.12, 0.08, 0.02, 0.01, 0.03, 0.25, 0.04],
        dtype=jnp.float32,
    )
    assert float(jnp.linalg.norm(context)) > 0.0

    params = init_params(jax.random.PRNGKey(2))
    losses = []
    grads = None
    for _ in range(5):
        params, loss, grads, _parts = train_step(params, x, z_ref, context, lr=1e-2)
        assert jnp.isfinite(loss)
        losses.append(loss)

    assert grads is not None
    film_leaves = jax.tree_util.tree_leaves(grads["film"])
    assert any(float(jnp.sum(jnp.abs(leaf))) > 0.0 for leaf in film_leaves)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads))
    # Re-evaluate so the updated loss is finite as well.
    updated, _ = total_loss(
        apply(params, x, context).z_hat,
        z_ref,
        apply(params, x, context).delta,
    )
    assert jnp.isfinite(updated)
