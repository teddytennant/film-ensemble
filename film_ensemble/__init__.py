"""FiLM ensemble conditioning for the 3D system in arXiv 2609.30746."""

from film_ensemble.ensemble import context_eigvec, context_mean_std_eigs
from film_ensemble.model import FiLMCorrector, apply, film, forward, init_params
from film_ensemble.system import coarse_params, reference_params, rk4_step, rollout
from film_ensemble.train import total_loss, train_step

__all__ = [
    "FiLMCorrector",
    "apply",
    "coarse_params",
    "context_eigvec",
    "context_mean_std_eigs",
    "film",
    "forward",
    "init_params",
    "reference_params",
    "rk4_step",
    "rollout",
    "total_loss",
    "train_step",
]
