import functools
from typing import Callable

import jax
import jax.numpy as jnp
from jumanji.types import TimeStep

from memento.utils.acting_utils import Information


def compute_advantages(returns):
    """Compute advantages"""
    # compute baseline
    baseline = returns.mean(-1, keepdims=True)

    # compute advantage
    if returns.shape[-1] > 1:
        advantages = returns - baseline
    else:
        advantages = returns

    return advantages


def max_improvement(traj: TimeStep, info: Information, loss_extra: dict, sp_spec: str):
    """Compute max improvement loss"""

    # compute episode returns
    returns = traj.reward.sum(-1)  # [N, K, M, t] --> [N, K, M]

    # R*
    R_star = loss_extra["best_return"]

    # update the returns to be the relu of difference with best return
    if sp_spec:
        offset = jnp.repeat(
            R_star[:, None, None],
            repeats=returns.shape[-1],
            axis=-1,
        )
    else:
        offset = R_star[:, None, :]

    # substract the offset
    returns = jax.nn.relu(returns - offset)  # loss_extra["best_return"]

    # get the logprob
    logprob_traj = info.extras["logprob"].sum(-1)  # [N, K, M, t] --> [N, K, M]

    # compute advantages
    advantages = compute_advantages(returns=returns)

    # compute loss
    loss = -jnp.mean(advantages * logprob_traj)

    return loss


def log_rectified_max_improvement(
    traj: TimeStep, info: Information, loss_extra: dict, sp_spec: str
):
    """Compute skewed sum loss"""

    # compute episode returns
    returns = traj.reward.sum(-1)  # [N, K, M, t] --> [N, K, M]

    # R*
    R_star = loss_extra["best_return"]

    # update the returns to be the relu of difference with best return
    if sp_spec:
        offset = jnp.repeat(
            R_star[:, None, None],
            repeats=returns.shape[-1],
            axis=-1,
        )
    else:
        offset = R_star[:, None, :]

    returns = jax.lax.cond(
        loss_extra["first_step"],
        lambda r, o: r,
        lambda r, o: jax.nn.relu(r - o),  # loss_extra["best_return"]
        *(returns, offset),
    )

    # get the logprob
    logprob_traj = info.extras["logprob"].sum(-1)  # [N, K, M, t] --> [N, K, M]

    # compute advantages
    advantages = compute_advantages(returns=returns)

    # compute loss
    loss = -jnp.mean(advantages * logprob_traj)

    loss = loss_extra["rectified_sum_weight"] * loss

    return loss


def pomo(
    traj: TimeStep,
    info: Information,
    loss_extra: dict,
):
    """Compute POMO loss"""

    # compute episode returns
    returns = traj.reward.sum(-1)  # [N, K, M, t] --> [N, K, M]

    # get the logprob
    logprob_traj = info.extras["logprob"].sum(-1)  # [N, K, M, t] --> [N, K, M]

    # compute advantages
    advantages = compute_advantages(returns=returns)

    # compute loss
    loss = -jnp.mean(advantages * logprob_traj)

    return loss


def get_loss_fn(cfg) -> Callable:
    """returns loss function given config"""
    # log_rectified_max_improvement
    # relu(R-R*) = R at step 0
    if cfg.type == "LRMI":
        print("loss fn in use: LRMI")
        return functools.partial(log_rectified_max_improvement, sp_spec=cfg.sp_spec)

    elif cfg.type == "max_improvement":
        print("loss fn in use: max_improvement")
        return functools.partial(max_improvement, sp_spec=cfg.sp_spec)

    else:  # cfg.type == "POMO":
        print("loss fn in use: POMO")
        return pomo


def get_rectified_sum_weights(cfg) -> jnp.ndarray:
    # num_ = cfg.optimizer.num_gradient_accumulation_steps  # 100
    num_ = cfg.budget  # 100
    c = cfg.loss.weight_offset
    d = cfg.loss.weight_scale

    get_w = lambda x: d * jnp.log(x + (1 + c))

    weights = []
    for i in range(0, num_):
        w_i = get_w(i)
        weights.append(w_i)

    weights = jnp.array(weights)

    # normalise weights
    sum_of_weights = weights.sum()
    normaliser = cfg.budget / sum_of_weights

    weights = weights * normaliser

    return weights
