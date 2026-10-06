"""Tiny FiLM residual-attention corrector (Appendix B.3, reduced widths).

Widths are smaller than Table 5: d_hid 16 not 32, one pre-norm block not
two, context hidden 8 not 48 (the backbone is tiny). FFN width is 32.
Two attention heads. Gate bias is initialized to -2.

    h_film = (1 + gamma) * h + beta
    delta = W_delta h_film + b_delta
    g = sigmoid(W_g h_film + b_g)
    z_hat = z_coarse + g * delta

``forward(x, context=None)`` skips FiLM (gamma = 0, beta = 0, identity).
``forward(x, context)`` runs the context net and applies FiLM.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


D_IN = 3
D_HID = 16
D_CTX = 8
D_CTX_HID = 8
N_HEADS = 2
FFN_WIDTH = 32
H_DEFAULT = 8


class ForwardOut(NamedTuple):
    z_hat: jnp.ndarray
    delta: jnp.ndarray
    gate: jnp.ndarray


def film(h, gamma, beta):
    """FiLM modulation: (1 + gamma) * h + beta. Broadcasts over a leading axis."""
    return (1.0 + gamma) * h + beta


def sinusoidal_position_encoding(length, d_hid):
    """Standard sinusoidal position encoding, shape (length, d_hid)."""
    position = jnp.arange(length, dtype=jnp.float32)[:, None]
    index = jnp.arange(d_hid, dtype=jnp.float32)[None, :]
    exponent = (2.0 * jnp.floor(index / 2.0)) / jnp.asarray(d_hid, dtype=jnp.float32)
    angles = position / jnp.power(10000.0, exponent)
    return jnp.where(index % 2 == 0, jnp.sin(angles), jnp.cos(angles))


def layer_norm(h, scale, bias, eps=1e-5):
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.mean((h - mean) ** 2, axis=-1, keepdims=True)
    return scale * (h - mean) * jax.lax.rsqrt(var + eps) + bias


def _normal(key, shape, scale):
    return jax.random.normal(key, shape, dtype=jnp.float32) * jnp.asarray(scale, dtype=jnp.float32)


def init_params(key, scale=0.1):
    """Initialize the tiny corrector.

    W_g is zero and b_g is -2, so the initial gate is sigmoid(-2) ≈ 0.12
    on any input. Context-net and delta-head weights are random so a
    nonzero context changes the output and receives gradient.
    """
    keys = jax.random.split(key, 12)
    d = D_HID
    return {
        "in_proj": {
            "W": _normal(keys[0], (D_IN, d), scale),
            "b": jnp.zeros((d,), dtype=jnp.float32),
        },
        "block": {
            "ln1_scale": jnp.ones((d,), dtype=jnp.float32),
            "ln1_bias": jnp.zeros((d,), dtype=jnp.float32),
            "W_q": _normal(keys[1], (d, d), scale),
            "W_k": _normal(keys[2], (d, d), scale),
            "W_v": _normal(keys[3], (d, d), scale),
            "W_o": _normal(keys[4], (d, d), scale),
            "ln2_scale": jnp.ones((d,), dtype=jnp.float32),
            "ln2_bias": jnp.zeros((d,), dtype=jnp.float32),
            "W1": _normal(keys[5], (d, FFN_WIDTH), scale),
            "b1": jnp.zeros((FFN_WIDTH,), dtype=jnp.float32),
            "W2": _normal(keys[6], (FFN_WIDTH, d), scale),
            "b2": jnp.zeros((d,), dtype=jnp.float32),
        },
        "film": {
            "W": _normal(keys[7], (D_CTX, D_CTX_HID), scale),
            "b": jnp.zeros((D_CTX_HID,), dtype=jnp.float32),
            "W_gamma": _normal(keys[8], (D_CTX_HID, d), scale),
            "b_gamma": jnp.zeros((d,), dtype=jnp.float32),
            "W_beta": _normal(keys[9], (D_CTX_HID, d), scale),
            "b_beta": jnp.zeros((d,), dtype=jnp.float32),
        },
        "heads": {
            "W_delta": _normal(keys[10], (d, D_IN), scale),
            "b_delta": jnp.zeros((D_IN,), dtype=jnp.float32),
            "W_g": jnp.zeros((d, D_IN), dtype=jnp.float32),
            "b_g": jnp.full((D_IN,), -2.0, dtype=jnp.float32),
        },
    }


def context_to_film(film_params, context):
    """c (8,) -> tanh hidden 8 -> gamma, beta in R^{16}."""
    hidden = jnp.tanh(context @ film_params["W"] + film_params["b"])
    gamma = hidden @ film_params["W_gamma"] + film_params["b_gamma"]
    beta = hidden @ film_params["W_beta"] + film_params["b_beta"]
    return gamma, beta


def multihead_attention(h, block):
    """2-head attention. h has shape (H, d_hid)."""
    length, d_hid = h.shape
    head_dim = d_hid // N_HEADS
    q = (h @ block["W_q"]).reshape(length, N_HEADS, head_dim).transpose(1, 0, 2)
    k = (h @ block["W_k"]).reshape(length, N_HEADS, head_dim).transpose(1, 0, 2)
    v = (h @ block["W_v"]).reshape(length, N_HEADS, head_dim).transpose(1, 0, 2)
    logits = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.asarray(head_dim, dtype=h.dtype))
    weights = jax.nn.softmax(logits, axis=-1)
    mixed = jnp.matmul(weights, v).transpose(1, 0, 2).reshape(length, d_hid)
    return mixed @ block["W_o"]


def transformer_block(h, block):
    """One pre-norm block: LayerNorm, 2-head attention, residual, LayerNorm, GELU FFN, residual."""
    h = h + multihead_attention(layer_norm(h, block["ln1_scale"], block["ln1_bias"]), block)
    normed = layer_norm(h, block["ln2_scale"], block["ln2_bias"])
    hidden = jax.nn.gelu(normed @ block["W1"] + block["b1"])
    h = h + (hidden @ block["W2"] + block["b2"])
    return h


def encode(params, x):
    """Project (H, 3) -> (H, 16), add sinusoidal positions, run one block."""
    x = jnp.asarray(x, dtype=jnp.float32)
    h = x @ params["in_proj"]["W"] + params["in_proj"]["b"]
    h = h + sinusoidal_position_encoding(x.shape[0], h.shape[-1])
    return transformer_block(h, params["block"])


def heads(params, h_film):
    """Per-step delta in R^3 and gate in (0, 1)^3."""
    delta = h_film @ params["heads"]["W_delta"] + params["heads"]["b_delta"]
    gate = jax.nn.sigmoid(h_film @ params["heads"]["W_g"] + params["heads"]["b_g"])
    return delta, gate


def apply(params, x, context=None):
    """Functional forward. context=None uses gamma=0, beta=0 (FiLM identity)."""
    x = jnp.asarray(x, dtype=jnp.float32)
    h = encode(params, x)
    if context is None:
        gamma = jnp.zeros((h.shape[-1],), dtype=h.dtype)
        beta = jnp.zeros((h.shape[-1],), dtype=h.dtype)
    else:
        context = jnp.asarray(context, dtype=jnp.float32)
        gamma, beta = context_to_film(params["film"], context)
    h_film = film(h, gamma, beta)
    delta, gate = heads(params, h_film)
    z_hat = x + gate * delta
    return ForwardOut(z_hat=z_hat, delta=delta, gate=gate)


def forward(x, context=None, params=None):
    """``forward(x, context=None)`` / ``forward(x, context)``.

    ``params`` is required (see ``init_params``). Prefer
    ``FiLMCorrector(params).forward(x, context)`` for the bound form.
    Passing ``context=None`` skips FiLM. A provided context is mapped to
    gamma, beta and applied as ``(1 + gamma) * h + beta``.
    """
    if params is None:
        raise TypeError("forward(x, context=None) requires params= from init_params")
    return apply(params, x, context)


class FiLMCorrector:
    """Bound corrector whose ``forward(x, context=None)`` matches the spec."""

    def __init__(self, params):
        self.params = params

    def forward(self, x, context=None):
        return apply(self.params, x, context)

    def __call__(self, x, context=None):
        return self.forward(x, context)
