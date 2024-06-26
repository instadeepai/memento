from typing import Union

import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array

from memento.environments.cvrp.types import Observation as CVRPObservation
from memento.environments.knapsack.types import Observation as KnapsackObservation
from memento.environments.tsp.types import Observation as TSPObservation
from memento.environments.cvrp.utils import DEPOT_IDX
from memento.networks.base import DecoderBase, EncoderBase
from memento.networks.efficient_mha import EfficientMultiHeadAttention


class CVRPEncoder(EncoderBase):
    # Modified for CVRP according to original source code: https://github.com/yd-kwon/POMO/blob/master/NEW_py_ver/CVRP/POMO/CVRPModel.py (~line 116)
    def get_problem_projection(self, problem: Array) -> Array:
        proj_depot = hk.Linear(self.model_size, name="depot_encoder")
        proj_nodes = hk.Linear(self.model_size, name="nodes_encoder")
        return jnp.where(
            jnp.zeros((problem.shape[0], 1)).at[DEPOT_IDX].set(1),
            proj_depot(problem),
            proj_nodes(problem),
        )


class CVRPDecoder(DecoderBase):
    def get_context(self, observation: CVRPObservation, embeddings: Array) -> Array:  # type: ignore[override]
        return jnp.concatenate(
            [
                embeddings.mean(0),
                embeddings[observation.position],
                observation.capacity[None],
            ],
            axis=0,
        )[
            None
        ]  # [1, 2*128+1=257,]

    def get_transformed_attention_mask(self, attention_mask: Array) -> Array:
        return jnp.where(attention_mask, 0, 1)


class CVRPConditionedDecoder(CVRPDecoder):
    def __init__(
        self,
        num_heads,
        key_size,
        model_size=128,
        name="decoder",
        embedding_size=16,
        eas_training=True,
        key_chunk_size=100,
        query_chunk_size=100,
    ):
        super().__init__(num_heads, key_size, model_size, name)

        self.embedding_size = embedding_size
        self.eas_training = eas_training
        self.key_chunk_size = key_chunk_size
        self.query_chunk_size = query_chunk_size

    def __call__(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        embeddings: Array,
        behavior_marker: Array,
        condition_query: bool = False,
        condition_key: bool = False,
        condition_value: bool = False,
    ) -> Array:
        context = self.get_context(observation, embeddings)  # [1, 3*128=384,]

        if self.eas_training:
            context = jax.lax.stop_gradient(context)

        if condition_query:
            context = jnp.concatenate(
                [context, jnp.expand_dims(behavior_marker, axis=0)], axis=1
            )  # with bd: [1, 384 + bd_dim]

        mha = EfficientMultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            model_size=self.model_size,
            w_init_scale=1,
            name="mha_dec",
            precision=jax.lax.Precision.DEFAULT,
            query_chunk_size=self.query_chunk_size,
            key_chunk_size=self.key_chunk_size,
        )

        mha_value = embeddings
        if condition_value:
            repeated_behavior_marker = jnp.repeat(
                jnp.expand_dims(behavior_marker, axis=0),
                repeats=embeddings.shape[0],
                axis=0,
            )

            mha_value = jnp.concatenate([embeddings, repeated_behavior_marker], axis=1)

        mha_key = embeddings
        if condition_key:
            mha_key = mha_value

        attention_mask = jnp.expand_dims(observation.action_mask, (0, 1))
        context = mha(
            query=context,
            key=mha_key,
            value=mha_value,
            mask=self.get_transformed_attention_mask(attention_mask),
        )  # --> [128]

        attn_logits = (
            embeddings @ context.squeeze() / jnp.sqrt(self.model_size)
        )  # --> [num_cities/items]
        attn_logits = 10 * jnp.tanh(attn_logits)  # clip to [-10,10]

        return attn_logits
