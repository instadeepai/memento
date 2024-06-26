import functools
from typing import Any, Dict, Optional, Tuple

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
from memento.utils.emitter_pool import CMAPoolEmitter
from memento.utils.networks import get_networks


def slowrl_rollout(
    cfg: omegaconf.DictConfig,
    environment: PoppyEnv,
    params: chex.ArrayTree,
    behavior_markers: chex.Array,
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
        # TODO
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
    @functools.partial(jax.vmap, in_axes=(0, 0, None, 0, 0, 0, 0))  # over N problems
    @functools.partial(
        jax.vmap, in_axes=(None, None, None, 0, 0, 0, None)
    )  # over K agents - behaviors
    @functools.partial(
        jax.vmap, in_axes=(None, None, None, None, 0, 0, 0)
    )  # M starting pos.
    def generate_trajectory_fn(
        problem,
        embeddings,
        decoder_params,
        behavior_marker,
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
            behavior_marker,
            start_position,
            acting_key,
            memory_state,
        )

    # generate the traj
    acting_state, (traj, info) = generate_trajectory_fn(
        problems,
        embeddings,
        decoder_params,
        behavior_markers,
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
    behavior_dim: Optional[int] = None,
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
        behavior_markers,
        memory_state,
    ):
        """Run the rollout on a batch of problems and return the episode return.

        Args:
            problems: A batch of N problems ([N, problem_size, 2]).
            start_positions: M starting positions for each problem-agent pair ([N, K, M]).
            acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
            behavior_markers: latent vector for COMPASS.
            memory_state: The external memory containing information about past episodes.

        Returns:
            episode_return: The total return matrix for each N problem, K agent, M starting position
            with size [N, K, M].
        """

        _, (traj, info, data) = slowrl_rollout(
            cfg=cfg,
            environment=environment,
            params=params,
            behavior_markers=behavior_markers,
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

    # specific to POMO
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
        random_key,
        problems,
        start_positions,
        acting_keys,
        emitter_state,
        memory_state,
        used_budget,
    ):
        """Get one batch of solutions to one batch of problem instances.
        Will be scanned through to get full attempt on the batch of instances.

        Args:
            problems: instances to be solved.
            start_positions: starting positions for each instance.
            acting_keys: acting keys (will be updated).
            behavior_markers: latent vector for COMPASS.
            emitter_state: state of the emitter.
            memory_state: external memory.
        # out: metrics

        """

        num_agents = acting_keys.shape[1]

        # generate the behavior markers
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, num=problems.shape[0])

        behavior_markers, _random_keys = jax.vmap(emitter.sample)(
            emitter_state, subkeys
        )

        # run the validation episodes
        episode_return, data, episode_logprob = run_validate(
            problems, start_positions, acting_keys, behavior_markers, memory_state
        )

        # update key - pb x agent x sp
        acting_keys = jax.vmap(jax.vmap(jax.vmap(lambda y: jax.random.split(y)[1])))(
            acting_keys
        )

        # sort behavior markers based on the perf we got
        fitnesses = -episode_return.max(-1)

        # only take the actor (hence behavior marker) into account
        sorted_indices = jnp.argsort(fitnesses, axis=-1)

        data = jax.tree_map(
            lambda x: x.transpose((0, 2, 3, 1, 4)), data
        )  # [pbs, bd_dim, sp, nodes, feat_dim] -> [pbs, sp, nodes, bd_dim, feat_dim]

        # add data to the memory
        memory_state = jax.vmap(jax.vmap(jax.vmap(update_memory)))(
            memory_state,
            data,
        )

        # sort the behaviors accordingly
        sorted_behavior_markers = jax.vmap(functools.partial(jnp.take, axis=0))(
            behavior_markers, sorted_indices
        )

        # use it to update the state
        emitter_state = jax.vmap(emitter.update_state)(
            emitter_state,
            sorted_candidates=sorted_behavior_markers[
                :, : cfg.validation_pop_size // 2
            ],
        )

        # compute the metrics

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

        return (
            metrics,
            random_key,
            acting_keys,
            emitter_state,
            memory_state,
            used_budget,
        )

    def run_full_budget(
        random_key, problems, start_positions, acting_keys, emitter_state
    ):
        """Run full budget but only on a subset of the problem instances.

        Do a scan on the attempt_solutions_batch function.

        Args:
            problems: batch of instances to be solved.
            start_positions: start positions for each instance.
            acting_keys: initial acting keys (will be updated at each step).
            emitter_state: will be used to generate the behavior markers.

        Returns:
            Metrics of the batch of runs.
        """

        num_agents = acting_keys.shape[
            2
        ]  # acting_keys: (num_devices, N/num_devices, K, M, 2)
        num_solution_batches = int(cfg.budget / num_agents)
        # print("solutions", num_agents, num_solution_batches)

        @functools.partial(jax.pmap, axis_name="i")
        def full_pmapped_rollout(
            random_key,
            problems,
            start_positions,
            acting_keys,
            emitter_state,
            memory_state,
        ):
            used_budget = 0

            def scan_body(carry, _):
                """Scan body for the full budget."""
                (
                    random_key,
                    acting_keys,
                    emitter_state,
                    memory_state,
                    used_budget,
                ) = carry

                print("Random key: ", random_key.shape)

                (
                    metrics,
                    random_key,
                    acting_keys,
                    emitter_state,
                    memory_state,
                    used_budget,
                ) = attempt_solutions_batch(
                    random_key,
                    problems,
                    start_positions,
                    acting_keys,
                    emitter_state,
                    memory_state,
                    used_budget,
                )

                return (
                    random_key,
                    acting_keys,
                    emitter_state,
                    memory_state,
                    used_budget,
                ), metrics

            # run the scan
            (_, _, _, _, _), metrics = jax.lax.scan(
                scan_body,
                init=(
                    random_key,
                    acting_keys,
                    emitter_state,
                    memory_state,
                    used_budget,
                ),
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
        # PMAPPED function
        metrics = full_pmapped_rollout(
            random_key,
            problems,
            start_positions,
            acting_keys,
            emitter_state,
            memory_state,
        )

        return metrics

    if int(cfg.num_starting_points) == -1:
        cfg.num_starting_points = environment.get_episode_horizon()

    print("nb problems: ", cfg.problems.num_problems)

    # get a set of instances - (spread over devices)
    key = random.PRNGKey(cfg.problem_seed)
    problems, start_positions, acting_keys = get_instances(
        cfg.problems,
        key,
        environment,
        params,
        cfg.num_starting_points,
        pop_size=cfg.validation_pop_size,
    )

    # extract the num problems
    (problems, start_positions, acting_keys) = jax.tree_map(
        lambda x: x[: cfg.problems.num_problems],
        (problems, start_positions, acting_keys),
    )

    # spread over devices
    problems = spread_over_devices(problems)
    start_positions = spread_over_devices(start_positions)
    acting_keys = spread_over_devices(acting_keys)

    random_key = jax.random.PRNGKey(cfg.problem_seed)

    random_key, subkey = jax.random.split(random_key)
    emitter = CMAPoolEmitter(
        num_states=cfg.num_cmaes_states,
        population_size=cfg.validation_pop_size,
        num_best=cfg.validation_pop_size // 2,
        search_dim=behavior_dim,
        init_sigma=float(cfg.cmaes_sigma),
        delay_eigen_decomposition=False,
        init_minval=-cfg.behavior_amplification * jnp.ones((behavior_dim,)),
        init_maxval=cfg.behavior_amplification * jnp.ones((behavior_dim,)),
        random_key=subkey,
    )

    # this has to be done out of the jit
    emitter_state = jax.tree_util.tree_map(
        lambda x: jnp.repeat(
            jnp.expand_dims(x, axis=0), repeats=cfg.problems.num_problems, axis=0
        ),
        emitter.init(),
    )

    # spread amongst devices
    emitter_state = spread_over_devices(emitter_state)

    # print shapes of the emitter state
    print("Emitter state shapes: ", jax.tree_map(lambda x: x.shape, emitter_state))

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

    # need to do the same manip for the initial emitter state
    emitter_state = jax.tree_util.tree_map(
        lambda x: jnp.stack(jnp.split(x, num_instance_batches, axis=1), axis=1),
        emitter_state,
    )

    # revert dimension 0 and 1
    emitter_state = jax.tree_util.tree_map(
        lambda x: x.transpose((1, 0, *range(len(x.shape))[2:])), emitter_state
    )

    # split the random key
    random_keys = jax.random.split(
        random_key, num=cfg.num_devices * num_instance_batches
    )
    random_keys = spread_over_devices(random_keys)

    # split and stack the random keys
    random_keys = jnp.stack(
        jnp.split(random_keys, num_instance_batches, axis=1), axis=1
    )

    # transpose what needs to
    random_keys = random_keys.transpose((1, 0, *range(len(random_keys.shape))[2:]))
    random_keys = random_keys.squeeze(2)

    metrics = {}

    # loop over the batches - use for in order not to jit
    for i in range(num_instance_batches):

        emitter_state_i = jax.tree_map(lambda x: x[i], emitter_state)

        # run the full budget
        metrics_batch = run_full_budget(
            random_keys[i],
            problems[i],
            start_positions[i],
            acting_keys[i],
            emitter_state_i,
        )

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
