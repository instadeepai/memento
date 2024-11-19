import functools
from typing import Any, Dict, Tuple

import chex
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import jax.random as random
import omegaconf

from memento.environments.poppy_env import PoppyEnv
from memento.memory.external_memory import ExternalMemoryState, update_memory
from memento.networks import Networks
from memento.utils.acting_utils import generate_trajectory
from memento.utils.checkpoint import get_params
from memento.utils.data import get_instances
from memento.utils.devices import fetch_from_devices, spread_over_devices
from memento.utils.networks import get_networks


def slowrl_rollout(
    cfg: omegaconf.DictConfig,
    environment: PoppyEnv,
    params: chex.ArrayTree,
    networks: Networks,
    problems: jnp.ndarray,
    start_positions: jnp.ndarray,
    acting_keys: jnp.ndarray,
    memory_state: ExternalMemoryState,
) -> Tuple[Any, Tuple[Any, Any, Any]]:
    """Rollout a batch of agents on a batch of problems and starting points.

    RMK: slightly different from trainer rollout because of the mapping of the
    agents (see vmap in_axes in the generate_trajectory_fn).

    Args:
        cfg: The rollout config.
        environment: The environment to rollout.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
            across all agents. There is only one decoder in the case of conditioned decoder. A population
            is implicitely created by the use of several behavior markers as input to the decoder.
        networks: The required networks.
        problems: A batch of N problems ([N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair ([N, K, M]).
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
        memory_state: The external memory containing information about past episodes.

    Returns:
        Transitions of the rollout.
    """

    # split the params in encoder and decoder - those a merged in the training state
    encoder_params, decoder_params = hk.data_structures.partition(
        lambda m, n, p: "encoder" in m, params
    )

    # initialise the embeddings for each problem
    embeddings = jax.vmap(networks.encoder_fn.apply, in_axes=(None, 0))(
        encoder_params, problems
    )

    # TODO: vmap on external memory over start positions
    @functools.partial(jax.vmap, in_axes=(0, 0, None, 0, 0, 0))  # over N problems
    @functools.partial(
        jax.vmap, in_axes=(None, None, 0, 0, 0, None)
    )  # over K agents - behaviors
    @functools.partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0))  # M starting pos.
    def generate_trajectory_fn(
        problem,
        embeddings,
        decoder_params,
        start_position,
        acting_key,
        memory_state,
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


def slowrl_validate(
    cfg: omegaconf.DictConfig,
    params: chex.ArrayTree = None,
    logger: Any = None,
) -> Dict:
    """Run validation on input problems.

    Args:
        cfg: The config for validation.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
          across all agents, decoder params are assumed to have a leading dimension of shape K.
        logger: The logger to use.

    Returns:
        metrics: A dictionary of metrics from the validation.
    """

    print("Config : ", cfg)

    if cfg.num_devices < 0:
        cfg.num_devices = len(jax.local_devices())

    def log(metrics, used_budget, logger, key=None):
        metrics["used_budget"] = used_budget
        if logger:
            if key:
                metrics = {f"{key}/{k}": v for (k, v) in metrics.items()}
            logger.write(metrics)

    def run_validate(
        problems,
        start_positions,
        acting_keys,
        memory_state,
    ):
        """Run the rollout on a batch of problems and return the episode return.

        Args:
            problems: A batch of N problems ([N, problem_size, 2]).
            start_positions: M starting positions for each problem-agent pair ([N, K, M]).
            acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
            memory_state: The external memory containing information about past episodes.

        Returns:
            episode_return: The total return matrix for each N problem, K agent, M starting position
            with size [N, K, M].
        """

        _, (traj, info, data) = slowrl_rollout(
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

        metrics = info.metrics

        episode_return = metrics["rewards"].sum(-1)  # [N, K, M]
        episode_logprob = info.extras["logprob"]  # [N, K, M, t]

        return episode_return, data, episode_logprob

    # instantiate networks and environments
    networks = get_networks(cfg.networks)
    environment = hydra.utils.instantiate(cfg.environment)

    # get the params (encoder and decoder)
    if not params:
        params = get_params(cfg.checkpointing)

    # split encoder, decoder
    encoder_params, decoder_params = hk.data_structures.partition(
        lambda m, n, p: "encoder" in m, params
    )

    # duplicate the decoder params pop size time
    decoder_params = jax.tree_map(
        lambda x: jnp.repeat(x, cfg.pop_size, axis=0), decoder_params
    )

    # merge the encoder and decoder params
    params = hk.data_structures.merge(encoder_params, decoder_params)

    # define the number of starting points
    if cfg.num_starting_points < 0:
        num_starting_points = environment.get_problem_size()
    else:
        num_starting_points = cfg.num_starting_points

    def attempt_solutions_batch(
        problems, start_positions, acting_keys, memory_state, used_budget
    ):
        """Get one batch of solutions to one batch of problem instances.
        Will be scanned through to get full attempt on the batch of instances.

        Args:
            problems: instances to be solved.
            start_positions: starting positions for each instance.
            acting_keys: acting keys (will be updated).
            memory_state: external memory.
        # out: metrics

        """

        num_agents = acting_keys.shape[1]

        # run the validation episodes
        episode_return, data, episode_logprob = run_validate(
            problems, start_positions, acting_keys, memory_state
        )

        # update key - pb x agent x sp
        acting_keys = jax.vmap(jax.vmap(jax.vmap(lambda y: jax.random.split(y)[1])))(
            acting_keys
        )

        # update the external memory
        data = jax.tree_map(lambda x: x.squeeze(1), data)  # Remove the agent dimension

        memory_state = jax.vmap(jax.vmap(jax.vmap(update_memory)))(
            memory_state,
            data,
        )

        # get sign of the return
        if environment.is_reward_negative():
            ret_sign = -1
        else:
            ret_sign = 1
        return_str = environment.get_reward_string()

        # get best perf (aggr. over starting points)
        latest_batch_best_sp = episode_return.max(-1)

        # get latest batch min, mean, max and std
        latest_batch_max = latest_batch_best_sp.max(-1)
        latest_batch_min = latest_batch_best_sp.min(-1)
        latest_batch_mean = latest_batch_best_sp.mean(-1)
        latest_batch_std = latest_batch_best_sp.std(-1)

        # update the used budget
        used_budget += num_agents * num_starting_points

        # Make new metrics dictionary which will be all the returned statistics.
        metrics = {
            f"{return_str}_latest_max": ret_sign * latest_batch_max,
            f"{return_str}_latest_min": ret_sign * latest_batch_min,
            f"{return_str}_latest_mean": ret_sign * latest_batch_mean,
            f"{return_str}_latest_std": latest_batch_std,
            "budget": used_budget * jnp.ones_like(latest_batch_max),
        }

        return metrics, acting_keys, memory_state, used_budget

    def run_full_budget(problems, start_positions, acting_keys):
        """Run full budget but only on a subset of the problem instances.

        Do a scan on the attempt_solutions_batch function.

        Args:
            problems: batch of instances to be solved.
            start_positions: start positions for each instance.
            acting_keys: initial acting keys (will be updated at each step).

        Returns:
            Metrics of the batch of runs.
        """

        num_agents = acting_keys.shape[
            2
        ]  # acting_keys: (num_devices, N/num_devices, K, M, 2)
        num_solution_batches = int(cfg.budget / num_agents)

        @functools.partial(jax.pmap, axis_name="i")
        def full_pmapped_rollout(problems, start_positions, acting_keys, memory_state):
            used_budget = 0

            def scan_body(carry, _):
                """Scan body for the full budget."""
                acting_keys, memory_state, used_budget = carry

                (
                    metrics,
                    acting_keys,
                    memory_state,
                    used_budget,
                ) = attempt_solutions_batch(
                    problems, start_positions, acting_keys, memory_state, used_budget
                )

                return (acting_keys, memory_state, used_budget), metrics

            # run the scan
            (_, _, _), metrics = jax.lax.scan(
                scan_body,
                init=(acting_keys, memory_state, used_budget),
                xs=None,
                length=num_solution_batches,
            )

            return metrics

        # create the external memory
        memory_state = hydra.utils.instantiate(cfg.memory).init_state(budget=cfg.budget)

        # replicate the memory to have one for each node-sp pair
        memory_state = jax.tree_map(
            lambda x: jnp.repeat(
                x[None, ...], repeats=cfg.memory.num_node_buckets, axis=0
            ),
            memory_state,
        )

        memory_state = jax.tree_map(
            lambda x: jnp.repeat(x[None, ...], repeats=cfg.num_starting_points, axis=0),
            memory_state,
        )

        # TODO: end to-do

        # get the number of problems currently being solved - #devices x #problems per devices
        num_problems = problems.shape[0] * problems.shape[1]

        cfg.num_devices = len(jax.local_devices())

        # replicate the memory to have one for each problem
        memory_state = jax.tree_map(
            lambda x: jnp.repeat(
                x[None, ...], repeats=num_problems // cfg.num_devices, axis=0
            ),
            memory_state,
        )

        memory_state = jax.device_put_replicated(memory_state, jax.local_devices())

        # main fn: run the full budget, outputs the metrics of the runs
        metrics = full_pmapped_rollout(
            problems, start_positions, acting_keys, memory_state
        )

        return metrics

    if int(cfg.num_starting_points) == -1:
        cfg.num_starting_points = environment.get_episode_horizon()

    # get a set of instances - (spread over devices)
    key = random.PRNGKey(cfg.problem_seed)
    problems, start_positions, acting_keys = get_instances(
        cfg.problems,
        key,
        environment,
        params,
        cfg.num_starting_points,
    )

    # spread over devices
    problems = spread_over_devices(problems)
    start_positions = spread_over_devices(start_positions)
    acting_keys = spread_over_devices(acting_keys)

    # get the problems batch size
    instances_batch_size = cfg.instances_batch_size

    num_problems = problems.shape[1]

    # compute the number of batches
    num_instance_batches = int(num_problems / instances_batch_size)

    # split the problems, starting_positions, acting_keys into batches
    problems = jnp.stack(jnp.split(problems, num_instance_batches, axis=1), axis=1)
    start_positions = jnp.stack(
        jnp.split(start_positions, num_instance_batches, axis=1), axis=1
    )
    acting_keys = jnp.stack(
        jnp.split(acting_keys, num_instance_batches, axis=1), axis=1
    )

    # revert dimension 0 and 1
    problems = problems.transpose((1, 0, *range(len(problems.shape))[2:]))
    start_positions = start_positions.transpose(
        (1, 0, *range(len(start_positions.shape))[2:])
    )
    acting_keys = acting_keys.transpose((1, 0, *range(len(acting_keys.shape))[2:]))

    metrics = {}

    # loop over the batches - use for in order not to jit
    for i in range(num_instance_batches):
        # run the full budget
        metrics_batch = run_full_budget(problems[i], start_positions[i], acting_keys[i])

        if cfg.num_devices > 1:
            metrics_batch = fetch_from_devices(metrics_batch)

        # create the tour length from the latest tour length max
        latest_tour_length_max = metrics_batch["tour_length_latest_max"]  # N x budget
        tour_length = jax.lax.cummin(latest_tour_length_max, axis=1)
        metrics_batch["tour_length"] = tour_length

        # for each field, get the mean - devices x budget x problems
        for key in metrics_batch.keys():
            metrics_batch[key] = metrics_batch[key].mean((0, 2))

        # store the latest metrics in our metrics dict
        for key in metrics_batch.keys():
            if i == 0:
                metrics[key] = metrics_batch[key][None, ...]
            else:
                metrics[key] = jnp.concatenate(
                    [metrics[key], metrics_batch[key][None, ...]], axis=0
                )

    # aggregate the metrics - to get mean over the problems
    for key in metrics.keys():
        metrics[key] = metrics[key].mean(0)

    # log the final metrics (aggreg. on all problems)

    accumulated_budget = metrics["budget"]
    tmp_metrics = {}

    for i, budget in enumerate(accumulated_budget):
        for key in metrics.keys():
            tmp_metrics[key] = metrics[key][i]

        log(tmp_metrics, budget, logger, "slowrl")

    # extra interesting metrics
    tmp_metrics["improvement_delta"] = (
        metrics["tour_length"][0] - metrics["tour_length"][-1]
    )
    tmp_metrics["improvement_ratio"] = (
        tmp_metrics["improvement_delta"] / metrics["tour_length"][0] * 100
    )

    # add the tour length you got on the first shot
    tmp_metrics["tour_length_first_shot"] = metrics["tour_length"][0]

    # return the latest one
    return tmp_metrics
