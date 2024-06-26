"""Definition of the external memory used to store past data as k, v entries. """

from abc import ABC, abstractmethod
from typing import Dict, Union

import jax
import jax.numpy as jnp
from flax import struct

from memento.environments.cvrp.types import Observation as CVRPObservation
from memento.environments.tsp.types import Observation as TSPObservation
from memento.memory.metrics import (
    compute_timestep_diff_cvrp,
    compute_timestep_diff_tsp,
    compute_visited_overlap,
)
from memento.memory.types import CVRPMemoryDataPoint, TSPMemoryDataPoint
from memento.utils.acting_utils import Information


@struct.dataclass
class ExternalMemoryState:
    """State of the external memory.

    Attributes:
        data: data stored in the external memory
        current_size: current size of the external memory
        budget: total budget
        current_budget: current budget
    """

    data: Union[TSPMemoryDataPoint, CVRPMemoryDataPoint]
    current_size: jnp.ndarray
    budget: jnp.ndarray
    current_budget: jnp.ndarray


class ExternalMemory(struct.PyTreeNode, ABC):
    """External memory"""

    memory_size: jnp.ndarray

    @abstractmethod
    def init_state(self, budget: int) -> ExternalMemoryState:
        """Create an empty external memory.

        Args:
            budget: total budget
        Returns:
            An empty external memory
        """
        pass

    @abstractmethod
    def entries_from_trajs(
        self,
        traj: jnp.ndarray,
        info: Information,
        embeddings: jnp.array,
    ):
        """
        Convert trajectories to data points for external memory.

        Args:
            traj: Trajectories to be converted.
            info: Information about the problem.

        Returns:
            Data points for external memory.
        """
        pass

    @staticmethod
    def create_key_value(memory_state, current_node: int):
        """Retrieve the k, v entries corresponding to the n nearest neighbors of a given query point.

        Args:
            memory_state: An instance of ExternalMemoryState.
            current_node: The query point.

        Returns:
            k, v entries corresponding to the n nearest neighbors of the query point
        """

        keys = memory_state.data.context[current_node]
        values = memory_state.data.values[current_node]

        return keys, values

    @staticmethod
    @abstractmethod
    def retrieve_all(
        input: Union[jnp.array, TSPMemoryDataPoint, CVRPMemoryDataPoint],
        current_node: int,
    ):
        pass

    @staticmethod
    @abstractmethod
    def get_retrieval_metrics(
        observation: TSPObservation, memory_state: ExternalMemoryState
    ) -> Dict:
        """
        Compute the retrieval metrics.
        Args:
            observation:
            memory_state:

        Returns:
            Retrieval metrics.
        """
        pass


class TSPExternalMemory(ExternalMemory):
    """External memory used to store past data as k, v entries.

    This dataclass needs to:
    - store the external memory
    - have a mechanism to retrieve the k, v entries corresponding to
    the n nearest neighbors of a given query point.
    - have a mechanism to update the external memory with new k, v entries.

    Inspired by the replay buffer from Brax for the creation of the buffer
    and for the insertion mechanism.

    For the retrieval mechanism, we use jax approx top k.

    """

    memory_size: int
    num_nodes: int
    num_node_buckets: int
    context_size: int
    value_size: int
    disable_returns: bool
    behavior_dim: int

    def init_state(self, budget: int) -> ExternalMemoryState:
        """Create an empty external memory.

        Args:
            budget: total budget

        Returns:
            An empty external memory
        """
        memory_size = self.memory_size

        empty_datapoint = TSPMemoryDataPoint(
            position=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            action=jnp.zeros((memory_size, 1), dtype=jnp.int32),  # Shape (memory_size,)
            returns=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            logprob=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            mem_logprob=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            attn_logprob=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            traj_logprob=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            end_traj_logprob=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            age=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
        )

        empty_memory_state = ExternalMemoryState(
            data=empty_datapoint,
            current_size=jnp.array(0),
            budget=jnp.array(budget),
            current_budget=jnp.array(0),
        )

        return empty_memory_state

    def entries_from_trajs(
        self,
        traj: jnp.ndarray,
        info: Information,
        embeddings: jnp.array,
    ):
        """
        Convert trajectories to data points for external memory.

        Args:
            traj: Trajectories to be converted.
            info: Information about the problem.
            embeddings: Embeddings of the nodes.

        Returns:
            Data points for external memory.
        """

        def sort_entries(data, nodes):
            i = jnp.argsort(nodes, axis=-1)[..., None]
            data = jax.tree_map(lambda x: jnp.take_along_axis(x, i, axis=3), data)
            return data

        # use info to update the memory
        actions = info.extras["action"]
        returns = traj.reward.sum(-1)
        returns = returns[..., None].repeat(actions.shape[-1], axis=-1)
        nodes = info.extras["current_node"]

        if self.disable_returns:
            returns = jnp.zeros_like(returns)

        age = 1 - (info.extras["current_budget"] / info.extras["budget"])

        # get the logprobs for each actoin in the trajectory
        traj_logprobs = info.extras["logprob"]

        # compute the logprob for the whole trajectory: sum
        # and retranch the logprob for the current action
        traj_logprob = jnp.sum(traj_logprobs, axis=-1, keepdims=True) - traj_logprobs

        # compute the logprob for the rest of the trajectory
        def compute_rest_traj_logprob(logprob):
            b = logprob[..., ::-1]
            c = jnp.cumsum(b, axis=-1) - b
            traj_logprob = c[..., ::-1]

            remaining_length = jnp.arange(b.shape[-1], 0, -1)

            # divide by the number of remaining steps
            traj_logprob = traj_logprob / remaining_length

            return traj_logprob  # / b.shape[-1]

        end_traj_logprob = compute_rest_traj_logprob(traj_logprobs)

        # create the data point
        data = TSPMemoryDataPoint(  # type: ignore
            position=nodes[..., None],
            action=actions[..., None],
            returns=returns[..., None],
            logprob=info.extras["logprob"][..., None],
            mem_logprob=info.extras["mem_logprob"][..., None],
            attn_logprob=info.extras["attn_logprob"][..., None],
            traj_logprob=traj_logprob[..., None],
            end_traj_logprob=end_traj_logprob[..., None],
            age=age[..., None],
        )

        data = sort_entries(data, nodes)

        return data

    @staticmethod
    def retrieve_all(
        input: Union[jnp.array, TSPMemoryDataPoint, CVRPMemoryDataPoint],
        current_node: int,
    ):
        """Retrieve the entries corresponding to the n nearest neighbors of a given query point.

        Args:
            input: An instance of DataPoint or a jnp.array.
            current_node: The query point.

        Returns:
            entries corresponding to the n nearest neighbors of the query point
        """

        return jax.tree_map(lambda x: x[current_node], input)

    @staticmethod
    def get_retrieval_metrics(
        observation: TSPObservation, memory_state: ExternalMemoryState
    ):

        # return metrics
        return {}


class CVRPExternalMemory(ExternalMemory):
    """External memory used to store past data as k, v entries.

    This dataclass needs to:
    - store the external memory
    - have a mechanism to retrieve the k, v entries corresponding to
    the n nearest neighbors of a given query point.
    - have a mechanism to update the external memory with new k, v entries.

    Inspired by the replay buffer from Brax for the creation of the buffer
    and for the insertion mechanism.

    For the retrieval mechanism, we use jax approx top k.

    """

    memory_size: int
    num_nodes: int
    num_node_buckets: int
    context_size: int
    value_size: int
    disable_returns: bool
    behavior_dim: int

    def init_state(self, budget: int) -> ExternalMemoryState:
        """Create an empty external memory.

        Args:
            budget: total budget

        Returns:
            An empty external memory
        """
        memory_size = self.memory_size

        empty_datapoint = CVRPMemoryDataPoint(
            position=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            capacity=jnp.zeros((memory_size, 1)),
            action=jnp.zeros((memory_size, 1), dtype=jnp.int32),  # Shape (memory_size,)
            returns=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            logprob=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            mem_logprob=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            attn_logprob=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            traj_logprob=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            end_traj_logprob=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
            age=jnp.zeros((memory_size, 1)),  # Shape (memory_size,)
        )

        empty_memory_state = ExternalMemoryState(
            data=empty_datapoint,
            current_size=jnp.array(0),
            budget=jnp.array(budget),
            current_budget=jnp.array(0),
        )

        return empty_memory_state

    def entries_from_trajs(
        self,
        traj: jnp.ndarray,
        info: Information,
        embeddings: jnp.array,
    ):
        """
        Convert trajectories to data points for external memory.

        Args:
            traj: Trajectories to be converted.
            info: Information about the problem.
            embeddings: Embeddings of the nodes.

        Returns:
            Data points for external memory.
        """

        def sort_and_select_entries(data, nodes):
            i = jnp.argsort(nodes, axis=-1)[..., None]
            data = jax.tree_map(lambda x: jnp.take_along_axis(x, i, axis=3), data)

            num_nodes = nodes.shape[-1] // 2
            data = jax.tree_map(
                lambda x: jnp.concatenate(
                    [x[..., :1, :], x[..., -num_nodes:, :]], axis=3
                ),
                data,
            )
            return data

        # use info to update the memory
        actions = info.extras["action"]
        returns = traj.reward.sum(-1)
        returns = returns[..., None].repeat(actions.shape[-1], axis=-1)
        nodes = info.extras["current_node"]

        # it's a sequence of size 1, so remove this dimension
        # intermediate_context = jnp.squeeze(intermediate_context, axis=-2)
        actions_embeddings = jax.vmap(lambda x, y: x[y])(embeddings, actions)

        if self.disable_returns:
            returns = jnp.zeros_like(returns)

        values = jnp.concatenate([actions_embeddings, returns[..., None]], axis=-1)

        age = 1 - (info.extras["current_budget"] / info.extras["budget"])

        # get the logprobs for each actoin in the trajectory
        traj_logprobs = info.extras["logprob"]

        # compute the logprob for the whole trajectory: sum
        # and retranch the logprob for the current action
        traj_logprob = jnp.sum(traj_logprobs, axis=-1, keepdims=True) - traj_logprobs

        # compute the logprob for the rest of the trajectory
        def compute_rest_traj_logprob(logprob):
            b = logprob[..., ::-1]
            c = jnp.cumsum(b, axis=-1) - b

            traj_logprob = c[..., ::-1]

            remaining_length = jnp.arange(b.shape[-1], 0, -1)

            # divide by the number of remaining steps
            traj_logprob = traj_logprob / remaining_length

            return traj_logprob

        end_traj_logprob = compute_rest_traj_logprob(traj_logprobs)

        # create the data point
        data = CVRPMemoryDataPoint(
            position=nodes[..., None],
            capacity=info.extras["capacity"][..., None],
            action=actions[..., None],
            returns=returns[..., None],
            logprob=info.extras["logprob"][..., None],
            mem_logprob=info.extras["mem_logprob"][..., None],
            attn_logprob=info.extras["attn_logprob"][..., None],
            traj_logprob=traj_logprob[..., None],
            end_traj_logprob=end_traj_logprob[..., None],
            age=age[..., None],
        )

        data = sort_and_select_entries(data, nodes)

        return data

    @staticmethod
    def get_retrieval_metrics(
        observation: CVRPObservation, memory_state: ExternalMemoryState
    ):
        memory_data = retrieve(memory_state.data, observation.position)

        metrics = {}
        metrics.update(compute_timestep_diff_cvrp(observation, memory_data))
        metrics.update(compute_visited_overlap(observation, memory_data))

        return metrics

    @staticmethod
    def retrieve_all(
        input: Union[jnp.array, TSPMemoryDataPoint, CVRPMemoryDataPoint],
        current_node: int,
    ):
        """Retrieve the entries corresponding to the n nearest neighbors of a given query point.

        Args:
            input: An instance of DataPoint or a jnp.array.
            current_node: The query point.

        Returns:
            entries corresponding to the n nearest neighbors of the query point
        """

        return jax.tree_map(lambda x: x[current_node], input)


def reinitialize_memory(memory_state: ExternalMemoryState):
    """Reinitialize the external memory.

    Args:
        memory_state: An instance of ExternalMemoryState

    Returns:
        Reinitialized external memory.
    """

    empty_data = jax.tree_map(lambda x: jnp.zeros_like(x), memory_state.data)
    empty_state = ExternalMemoryState(
        data=empty_data,
        current_size=jnp.zeros_like(memory_state.current_size),
        budget=memory_state.budget,
        current_budget=jnp.zeros_like(memory_state.current_budget),
    )
    return empty_state


def retrieve(
    input: Union[jnp.array, TSPMemoryDataPoint, CVRPMemoryDataPoint],
    current_node: int,
):
    """Retrieve the entries corresponding to the n nearest neighbors of a given query point.

    Args:
        input: An instance of DataPoint or a jnp.array.
        current_node: The query point.

    Returns:
        entries corresponding to the n nearest neighbors of the query point
    """

    return jax.tree_map(lambda x: x[current_node], input)


def update_memory(
    memory_state: ExternalMemoryState,
    new_data: Union[TSPMemoryDataPoint, CVRPMemoryDataPoint],
):
    """
    Update the external memory with new data using concatenation.

    Args:
        memory_state: An instance of ExternalMemoryState.
        new_data: New data to be added, either TSPMemoryDataPoint or CVRPMemoryDataPoint.

    Returns:
        Updated external memory.
    """

    current_size = memory_state.current_size
    memory_size = memory_state.data.position.shape[0]

    # Concatenate new data with existing data
    concatenated_data = jax.tree_map(
        lambda x, y: jnp.concatenate([y, x], axis=0),
        memory_state.data,
        jax.tree_map(lambda x: x, new_data),
    )

    # Select the most recent entries, up to the memory size
    updated_data = jax.tree_map(lambda x: x[:memory_size], concatenated_data)

    updated_size = jnp.minimum(current_size + 1, memory_size)
    updated_budget = memory_state.current_budget + new_data.position.shape[0]

    updated_state = ExternalMemoryState(
        data=updated_data,
        current_size=updated_size,
        current_budget=updated_budget,
        budget=memory_state.budget,
    )

    return updated_state
