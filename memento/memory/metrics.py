from typing import Union

import jax.numpy as jnp
from chex import Array

from memento.environments.cvrp.types import Observation as CVRPObservation
from memento.environments.knapsack.types import Observation as KnapsackObservation
from memento.environments.tsp.types import Observation as TSPObservation
from memento.memory.types import CVRPMemoryDataPoint, TSPMemoryDataPoint


def compute_visited_overlap(
    observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
    memory_data: Union[TSPMemoryDataPoint, CVRPMemoryDataPoint],
) -> Array:
    """Computes the overlap between the visited nodes and the memory.

    Args:
        observation: The current observation.
        memory_data: external memory data.

    Returns:
        The overlap between the visited nodes and the memory.
    """

    # Step 1: Find non-overlapping elements for each row
    non_overlap = jnp.logical_xor(observation.action_mask, memory_data.visited_mask)

    # Step 2: Count non-overlapping elements per row
    non_overlap_counts = jnp.sum(non_overlap, axis=1)

    # Step 3: Compute mean, max, and min
    mean_count = jnp.mean(non_overlap_counts)
    max_count = jnp.max(non_overlap_counts)
    min_count = jnp.min(non_overlap_counts)

    num_exact_same = jnp.sum(
        jnp.all(observation.action_mask == memory_data.visited_mask, axis=1)
    )

    return {
        "mean_overlap": mean_count,
        "max_overlap": max_count,
        "min_overlap": min_count,
        "num_same_overlap": num_exact_same,
    }


def compute_timestep_diff_cvrp(
    observation: Union[CVRPObservation],
    memory_data: Union[TSPMemoryDataPoint, CVRPMemoryDataPoint],
) -> Array:
    """Computes the difference between the current timestep and the memory.

    Args:
        observation: The current observation.
        memory_data: external memory data.

    Returns:
        The difference between the current timestep and the memory.
    """

    timestep_obs = observation.num_visited
    timestep_memory = memory_data.visited_mask.sum(axis=-1)

    metrics = {
        "abs_diff_max": jnp.max(jnp.abs(timestep_obs - timestep_memory)),
        "abs_diff_mean": jnp.mean(jnp.abs(timestep_obs - timestep_memory)),
        "abs_diff_min": jnp.min(jnp.abs(timestep_obs - timestep_memory)),
        "diff_max": jnp.max(timestep_obs - timestep_memory),
        "diff_mean": jnp.mean(timestep_obs - timestep_memory),
        "diff_min": jnp.min(timestep_obs - timestep_memory),
    }

    return metrics


def compute_timestep_diff_tsp(
    observation: Union[TSPObservation],
    memory_data: Union[TSPMemoryDataPoint, CVRPMemoryDataPoint],
) -> Array:
    """Computes the difference between the current timestep and the memory.

    Args:
        observation: The current observation.
        memory_data: external memory data.

    Returns:
        The difference between the currenct timestep and the memory.
    """

    timestep_obs = (observation.trajectory > -1).sum()
    timestep_memory = memory_data.visited_mask.sum(axis=-1)

    metrics = {
        "abs_diff_max": jnp.max(jnp.abs(timestep_obs - timestep_memory)),
        "abs_diff_mean": jnp.mean(jnp.abs(timestep_obs - timestep_memory)),
        "abs_diff_min": jnp.min(jnp.abs(timestep_obs - timestep_memory)),
        "diff_max": jnp.max(timestep_obs - timestep_memory),
        "diff_mean": jnp.mean(timestep_obs - timestep_memory),
        "diff_min": jnp.min(timestep_obs - timestep_memory),
    }

    return metrics
