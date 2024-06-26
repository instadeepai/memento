import pickle
from typing import Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax import random
from jax.random import PRNGKey

from memento.environments.poppy_env import PoppyEnv


def get_start_positions(
    environment, start_key, num_start_positions, num_problems, num_agents
):
    """Generate the starting positions for each problem-agent pair.

    Args:
        environment: The environment to prepare problems for.
        start_key: The key for generating the starting positions.
        num_start_positions: The number of start positions per problem (M).  If <0
          then all possible positions are used, i.e. M=N.
        num_problems: The number of problems to generate (N).
        num_agents: The number of different agents that will each have unique starting points
          and acting keys on the same problem (K).

    Returns:
        num_start_positions: The number of start positions per problem.
        starting_positions: M starting positions for each problem-agent pair ([N, K, M]).
    """
    if num_start_positions < 0:
        start_positions = jnp.arange(
            environment.get_min_start(), environment.get_max_start() + 1
        )
        start_positions = (
            start_positions[None, None].repeat(num_problems, 0).repeat(num_agents, 1)
        )
        num_start_positions = environment.get_problem_size()
    else:
        start_positions = random.randint(
            start_key,
            (num_problems, 1, num_start_positions),
            minval=environment.get_min_start(),
            maxval=environment.get_max_start() + 1,
        ).repeat(
            num_agents, axis=1
        )  # make sure agents have same starting keys

    return num_start_positions, start_positions


def get_acting_keys(act_key, num_start_positions, num_problems, num_agents):
    """Get the acting keys

    Args:
        act_key: The key for generating the acting keys.
        num_start_positions: The number of start positions per problem.
        num_problems: The number of problems to generate (N).
        num_agents: The number of different agents that will each have unique starting points
          and acting keys on the same problem (K).

    Returns:
        acting_key: M acting keys for each problem-agent pair ([N, K, M, 2]).
    """
    acting_keys = random.split(
        act_key, num_problems * num_agents * num_start_positions
    ).reshape((num_problems, num_agents, num_start_positions, -1))

    return acting_keys


def prepare_problem_batch(
    problem_key: PRNGKey,
    start_act_key: PRNGKey,
    environment: PoppyEnv,
    num_problems: int,
    num_agents: int,
    num_start_positions: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Prepare a batch of problems.

    Args:
        prng_key: The key for generating this problem set.
        environment: The environment to prepare problems for.
        num_problems: The number of problems to generate (N).
        num_agents: The number of different agents that will each have unique starting points
          and acting keys on the same problem (K).
        num_start_positions: The number of start positions per problem (M).  If <0
          then all possible positions are used, i.e. M=N.

    Returns:
        problems: A batch of N problems ([N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair ([N, K, M]).
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
    """
    # start_key, act_key = random.split(start_act_key, 2)

    # WARNING: bad practice here, but quick trick for an experiment
    problem_key, start_key = jax.random.split(problem_key, 2)

    problems = jax.vmap(environment.generate_problem, in_axes=(0, None))(
        random.split(problem_key, num_problems), environment.get_problem_size()
    )

    # WARNING: use the problem key to fix the start positions
    num_start_positions, start_positions = get_start_positions(
        environment, start_key, num_start_positions, num_problems, num_agents
    )

    acting_keys = get_acting_keys(
        start_act_key, num_start_positions, num_problems, num_agents
    )

    return problems, start_positions, acting_keys


def load_instances(cfg, key, environment, num_start_positions, num_agents):
    """Load problems instances from the given file and generate start positions and acting keys.

    Args:
        cfg: The config containing the dataset loading information.
        key: The PRNGKey for generating the starting positions and acting keys.
        environment: The environment to generate the starting positions on.
        num_start_positions: The number of starting positions to generate.
        num_agents: The number of different agents that will each have unique starting points
          and acting keys on the same problem (K).

    Returns:
        problems: A batch of N problems ([N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair ([N, K, M]).
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
    """
    with open(cfg.load_path, "rb") as f:
        problems = jnp.array(pickle.load(f))

    start_key, act_key = random.split(key, 2)
    num_start_positions, start_positions = get_start_positions(
        environment, start_key, num_start_positions, problems.shape[0], num_agents
    )
    acting_keys = get_acting_keys(
        act_key, num_start_positions, problems.shape[0], num_agents
    )

    return problems, start_positions, acting_keys


def get_instances(cfg, key, environment, params, num_start_positions, pop_size):
    """Get the problem instances, start positions, and acting keys.

    Args:
        cfg: The config containing the dataset loading information.
        key: A PRNGKey.
        environment: The environment to generate the starting positions on.
        params: The encoder and decoder parameters.
        num_start_positions: The number of starting positions to generate.

    TODO: parts of load_instances and prepare_problem_batch are duplicated here.

    Returns:
        problems: A batch of N problems divided over D devices ([D, N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair divided over D devices
        ([D, N, K, M]).
        acting_keys: M acting keys for each problem-agent pair divided over D devices
        ([D, N, K, M, 2]).
    """

    # # get the decoder
    # _, decoder = hk.data_structures.partition(lambda m, n, p: "encoder" in m, params)

    # # deduce the number of agents - TODO: can be dangerous
    # num_agents = jax.tree_util.tree_leaves(decoder)[0].shape[0]

    num_agents = pop_size

    print(f"Number of agents: {num_agents}")

    if cfg.load_problem:
        problems, start_positions, acting_keys = load_instances(
            cfg, key, environment, num_start_positions, num_agents
        )
    else:
        key, problem_key = jax.random.split(key)
        problems, start_positions, acting_keys = prepare_problem_batch(
            key,
            problem_key,
            environment,
            cfg.num_problems,
            num_agents,
            num_start_positions,
        )

    return problems, start_positions, acting_keys


def generate_zeros_from_spec(spec: jnp.ndarray) -> jnp.ndarray:
    zeros: jnp.ndarray = jnp.zeros(spec.shape, spec.dtype)
    return zeros
