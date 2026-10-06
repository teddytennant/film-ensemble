"""Ensemble-covariance context (Algorithm 2), one time slice.

Given member states of shape (M, 3), or trajectories (T, M, 3) from which
the last time is taken:

- mu: mean over members, shape (3,)
- s: population std over members, shape (3,)
- covariance: population covariance, 3x3 (divide by M, matching ``s**2``)
- lambda1, lambda2: two leading eigenvalues of that covariance
  (``jnp.linalg.eigh``, which returns eigenvalues in ascending order, so
  lambda1 is the largest)
- v1: leading eigenvector

``context_mean_std_eigs`` returns (mu, s, lambda1, lambda2), length 8.
``context_eigvec`` returns the leading eigenvector, length 3.

OTD modes are not computed. The ensemble leading eigenvector is the
Jacobian-free proxy used as a conditioning signal.
"""

from __future__ import annotations

import jax.numpy as jnp


def _members(ensemble):
    ensemble = jnp.asarray(ensemble, dtype=jnp.float32)
    if ensemble.ndim == 3:
        # (T, M, 3) -> snapshot at the last time
        ensemble = ensemble[-1]
    if ensemble.ndim != 2 or ensemble.shape[-1] != 3:
        raise ValueError(
            f"expected ensemble shape (M, 3) or (T, M, 3), got {ensemble.shape}"
        )
    return ensemble


def ensemble_moments(ensemble):
    """Return mu (3,), s (3,), covariance (3, 3), evals (3,), evecs (3, 3).

    Eigenvalues are ascending. Eigenvector columns of ``evecs`` match that
    order, so the leading eigenvector is ``evecs[:, -1]``.
    """
    members = _members(ensemble)
    mu = jnp.mean(members, axis=0)
    centered = members - mu
    m = members.shape[0]
    cov = (centered.T @ centered) / jnp.asarray(m, dtype=members.dtype)
    cov = 0.5 * (cov + cov.T)
    s = jnp.std(members, axis=0)
    evals, evecs = jnp.linalg.eigh(cov)
    return mu, s, cov, evals, evecs


def context_mean_std_eigs(ensemble):
    """Length-8 context: mean (3,), std (3,), lambda1, lambda2."""
    mu, s, _cov, evals, _evecs = ensemble_moments(ensemble)
    lambda1 = evals[-1]
    lambda2 = evals[-2]
    return jnp.concatenate([mu, s, jnp.stack([lambda1, lambda2])])


def context_eigvec(ensemble, r=1):
    """Leading eigenvector when r=1. r=2 is the second-leading mode.

    The vector is a unit vector. Its sign is the sign returned by eigh.
    """
    if r < 1:
        raise ValueError("r starts at 1 for the leading eigenvector")
    _mu, _s, _cov, _evals, evecs = ensemble_moments(ensemble)
    return evecs[:, -int(r)]
