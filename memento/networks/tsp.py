import haiku as hk
import hydra.utils
import jax.numpy as jnp
from chex import Array

from memento.environments.tsp.types import Observation as TSPObservation
from memento.memory.external_memory import ExternalMemoryState
from memento.networks.base import DecoderBase, EncoderBase, MemoryConditionedDecoderBase


class TSPEncoder(EncoderBase):
    def get_problem_projection(self, problem: Array) -> Array:
        proj = hk.Linear(self.model_size, name="encoder")
        return proj(problem)


class TSPDecoder(DecoderBase):
    def get_context(self, observation: TSPObservation, embeddings: Array) -> Array:  # type: ignore[override]
        return jnp.concatenate(
            [
                embeddings.mean(0),
                embeddings[observation.position],
                embeddings[observation.start_position],
            ],
            axis=0,
        )[
            None
        ]  # [1, 3*128=384,]

    def get_transformed_attention_mask(self, attention_mask: Array) -> Array:
        return attention_mask


class TSPMemoryConditionedDecoder(MemoryConditionedDecoderBase):
    def get_context(self, observation: TSPObservation, embeddings: Array) -> Array:  # type: ignore[override]
        return jnp.concatenate(
            [
                embeddings.mean(0),
                embeddings[observation.position],
                embeddings[observation.start_position],
            ],
            axis=0,
        )[
            None
        ]  # [1, 3*128=384,]

    def get_transformed_attention_mask(self, attention_mask: Array) -> Array:
        return attention_mask

    def retrieve(
        self,
        memory_state: ExternalMemoryState,
        current_node,
    ):
        keys, values = self.memory.create_key_value(memory_state, current_node)
        return keys, values

    def get_retrieval_metrics(
        self, observation: TSPObservation, memory_state: ExternalMemoryState
    ):
        metrics = self.memory.get_retrieval_metrics(observation, memory_state)
        return metrics
