import functools
from typing import Any, Optional, Tuple

import chex
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import jax.random as random
import omegaconf

import memento.trainers.trainer as trainer
from memento.environments.poppy_env import PoppyEnv
from memento.networks import Networks
from memento.utils.acting_utils import generate_trajectory
from memento.utils.checkpoint import get_params
from memento.utils.data import get_instances
from memento.utils.devices import spread_over_devices
from memento.utils.metrics import get_metrics


def rollout(
        cfg: omegaconf.DictConfig,
        environment: PoppyEnv,
        params: chex.ArrayTree,
        networks: Networks,
        problems: jnp.ndarray,
        start_positions: jnp.ndarray,
        acting_keys: jnp.ndarray,
        memory_state: Optional[chex.ArrayTree],
) -> Tuple[Any, Tuple[Any, Any, Any]]:
    """Rollout a batch of agents on a batch of problems and starting points.

    Args:
        cfg: The rollout config.
        environment: The environment to rollout.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
          across all agents, decoder params are assumed to have a leading dimension of shape K.
        networks: The required networks.
        problems: A batch of N problems ([N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair ([N, K, M]).
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
        memory_state: External memory to be used by the decoder.

    Returns:
        # TODO
    """

    # initialise the embeddings for each problem
    encoder_params, decoder_params = hk.data_structures.partition(
        lambda m, n, p: "encoder" in m, params
    )

    # get the embeddings for each problem - done once
    embeddings = jax.vmap(networks.encoder_fn.apply, in_axes=(None, 0))(
        encoder_params, problems
    )

    @functools.partial(jax.vmap, in_axes=(0, 0, None, 0, 0, 0))  # over N problems
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, None))  # over K agents
    @functools.partial(
        jax.vmap, in_axes=(None, None, None, 0, 0, None)
    )  # M starting pos. (vmap over start positions)
    def generate_trajectory_fn(
            problem, embeddings, decoder_params, start_position, acting_key, memory_state
    ):

        return generate_trajectory(
            networks.decoder_fn.apply,
            cfg.rollout.policy.temperature,
            environment,
            problem,
            embeddings,
            decoder_params,
            start_position,
            acting_key,
            memory_state,
        )

    # generate the traj
    acting_state, (traj, info) = generate_trajectory_fn(
        problems,
        embeddings,
        decoder_params,
        start_positions,
        acting_keys,
        memory_state,
    )

    info.metrics = jax.tree_map(lambda x: x.mean(), info.metrics)

    # compute the memory entries
    external_memory = hydra.utils.instantiate(cfg.memory)
    data = external_memory.entries_from_trajs(traj, info, embeddings)

    return acting_state, (traj, info, data)


def validate(
        cfg: omegaconf.DictConfig,
        params: chex.ArrayTree = None,
) -> dict:
    """Run validation on input problems.

    Args:
        cfg: The config for validation.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
          across all agents, decoder params are assumed to have a leading dimension of shape K.

    Returns:
        metrics: A dictionary of metrics from the validation.
    """
    if cfg.rollout.decoder_pmap_axis == "pop":
        # TODO: Handle metric collection in this case.
        raise NotImplementedError

    @functools.partial(jax.pmap, axis_name="i")
    def run_validate(problems, start_positions, acting_keys, memory_state):
        """Run the rollout on a batch of problems and return the episode return.

        Args:
            problems: A batch of N problems ([N, problem_size, 2]).
            start_positions: M starting positions for each problem-agent pair ([N, K, M]).
            acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).

        Returns:
            episode_return: The total return matrix for each N problem, K agent, M starting position
            with size [N, K, M].
        """
        # 1. Split problems, start_positions and acting_keys into chunks of size batch_size.
        # 2. Zip batches into list of inputs:
        #   [(problems[0],start_positions[0],acting_keys[0]),
        #    (problems[1],start_positions[1],acting_keys[1]),
        #    ...]
        num_batches = int(round(len(problems) / cfg.batch_size, 0))

        problems = jnp.stack(jnp.split(problems, num_batches, axis=0), axis=0)
        start_positions = jnp.stack(jnp.split(start_positions, num_batches, axis=0))
        acting_keys = jnp.stack(jnp.split(acting_keys, num_batches, axis=0))

        # reshape the memory
        memory_state = jax.tree_util.tree_map(
            lambda x: jnp.stack(jnp.split(x, num_batches, axis=0), axis=0),
            memory_state,
        )

        num_problems = problems.shape[1]

        if cfg.use_augmentations:
            problems = jax.vmap(jax.vmap(environment.get_augmentations))(problems)
            problems = problems.reshape(
                num_batches, num_problems * 8, environment.get_problem_size(), -1
            )
            # Note, the starting positions and acting keys are duplicated here.
            start_positions = jnp.repeat(start_positions, 8, axis=1)
            acting_keys = jnp.repeat(acting_keys, 8, axis=1)

        def body(_, x):
            problems, start_positions, acting_keys, memory_state = x
            _, (traj, info, _) = rollout(
                cfg=cfg,
                environment=environment,
                params=params,
                networks=networks,
                problems=problems,
                start_positions=start_positions,
                acting_keys=acting_keys,
                memory_state=memory_state,
            )
            info.metrics["rewards"] = traj.reward
            return None, info.metrics

        _, metrics = jax.lax.scan(
            body,
            init=None,
            xs=(problems, start_positions, acting_keys, memory_state),
        )

        if cfg.use_augmentations:
            num_agents, num_start_positions = (
                start_positions.shape[-2],
                start_positions.shape[-1],
            )
            metrics = jax.tree_map(
                lambda x: x.reshape(
                    num_batches,
                    num_problems,
                    8,
                    num_agents,
                    num_start_positions,
                    -1,
                ).max(2),
                metrics,
            )

        # Flatten batch dimension of metrics.
        metrics = jax.tree_map(lambda x: x.reshape(*(-1,) + x.shape[2:]), metrics)
        episode_return = metrics["rewards"].sum(-1)  # [N, K, M]
        return episode_return

    networks = trainer.get_networks(cfg.networks)
    environment = hydra.utils.instantiate(cfg.environment)
    if not params:
        params = get_params(cfg.checkpointing)

    key = random.PRNGKey(cfg.problem_seed)
    problems, start_positions, acting_keys = get_instances(
        cfg.problems, key, environment, params, cfg.num_starting_points
    )
    cfg.problems.num_problems = problems.shape[0]  # override in case of loaded data

    # spread over devices
    problems = spread_over_devices(problems)
    start_positions = spread_over_devices(start_positions)
    acting_keys = spread_over_devices(acting_keys)

    memory_state = hydra.utils.instantiate(cfg.memory).init_state(budget=cfg.budget)

    # replicate the memory to have on for each node

    if cfg.num_starting_points == -1:
        cfg.num_starting_points = environment.get_episode_horizon()

    memory_state = jax.tree_map(
        lambda x: jnp.repeat(
            x[None, ...], repeats=environment.get_episode_horizon(), axis=0
        ),
        memory_state,
    )

    # replicate the memory to have one for each problem
    memory_state = jax.tree_map(
        lambda x: jnp.repeat(x[None, ...], repeats=cfg.problems.num_problems, axis=0),
        memory_state,
    )

    # reshape to spread over devices
    memory_state = spread_over_devices(memory_state)

    # run the validation
    episode_return = run_validate(
        problems, start_positions, acting_keys, memory_state
    )

    episode_return = jnp.concatenate(episode_return, axis=0)

    if environment.is_reward_negative():
        ret_sign = -1
    else:
        ret_sign = 1
    return_str = environment.get_reward_string()

    # Make new metrics dictionary which will be all the returned statistics.
    metrics = {
        f"{return_str}": ret_sign * episode_return.max((-1, -2)).mean(),
        f"{return_str}_rand_agent": ret_sign * episode_return.max(-1).mean(),
        f"{return_str}_rand_start": ret_sign * episode_return.max(-2).mean(),
        f"{return_str}_rand_agent+start": ret_sign * episode_return.mean(),
    }

    metrics = get_metrics(
        metrics, episode_return, compute_expensive_metrics=cfg.compute_expensive_metrics
    )

    return metrics
