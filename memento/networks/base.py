from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import haiku as hk
import hydra.utils
import jax
import jax.numpy as jnp
from chex import Array

from memento.environments.cvrp.types import Observation as CVRPObservation
from memento.environments.knapsack.types import Observation as KnapsackObservation
from memento.environments.tsp.types import Observation as TSPObservation
from memento.networks.efficient_mha import EfficientMultiHeadAttention

if TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from chex import dataclass

from memento.memory.external_memory import ExternalMemoryState


@dataclass
class Networks:  # type: ignore
    encoder_fn: hk.Transformed
    decoder_fn: hk.Transformed


class EncoderBase(ABC, hk.Module):
    """Transformer-based encoder.

    By default, this is the encoder used by Kool et al. (arXiv:1803.08475) and
    Kwon et al. (arXiv:2010.16011).
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        key_size,
        model_size=None,
        expand_factor=4,
        name="encoder",
        key_chunk_size=100,
        query_chunk_size=100,
    ):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
        self.expand_factor = expand_factor
        self.key_chunk_size = key_chunk_size
        self.query_chunk_size = query_chunk_size

    def __call__(self, problem: Array) -> Array:
        x = self.get_problem_projection(problem)

        for i in range(self.num_layers):
            mha = EfficientMultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=self.model_size,
                w_init_scale=1 / self.num_layers,
                name=f"mha_{i}",
                precision=jax.lax.Precision.DEFAULT,
                query_chunk_size=self.query_chunk_size,
                key_chunk_size=self.key_chunk_size,
            )
            norm1 = hk.LayerNorm(
                axis=-1,  # should be batch norm according to Kool et al.
                create_scale=True,
                create_offset=True,
                name=f"norm_{i}a",
            )

            x = norm1(x + mha(query=x, key=x, value=x))

            mlp = hk.nets.MLP(
                output_sizes=[self.expand_factor * self.model_size, self.model_size],
                activation=jax.nn.relu,
                activate_final=False,
                name=f"mlp_{i}",
            )
            # should be batch norm according to POMO
            norm2 = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name=f"norm_{i}b"
            )
            x = norm2(x + mlp(x))

        return x

    @abstractmethod
    def get_problem_projection(self, problem: Array) -> Array:
        pass


class DecoderBase(ABC, hk.Module):
    """
    Decoder module.
    By default, this is the decoder used by Kool et al. (arXiv:1803.08475) and Kwon et al. (arXiv:2010.16011).
    """

    def __init__(
        self,
        num_heads,
        key_size,
        model_size=128,
        name="decoder",
        key_chunk_size=100,
        query_chunk_size=100,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
        self.key_chunk_size = key_chunk_size
        self.query_chunk_size = query_chunk_size

    def __call__(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        embeddings: Array,
    ) -> Array:
        context = self.get_context(observation, embeddings)
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

        attention_mask = jnp.expand_dims(observation.action_mask, (0, 1))
        context = mha(
            query=context,
            key=embeddings,
            value=embeddings,
            mask=self.get_transformed_attention_mask(attention_mask),
        )  # --> [128]

        attn_logits = (
            embeddings @ context.squeeze() / jnp.sqrt(self.model_size)
        )  # --> [num_cities/items]
        attn_logits = 10 * jnp.tanh(attn_logits)  # clip to [-10,10]

        return attn_logits, context

    @abstractmethod
    def get_context(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        embeddings: Array,
    ) -> Array:
        pass

    @abstractmethod
    def get_transformed_attention_mask(self, attention_mask: Array) -> Array:
        pass


class MemoryConditionedDecoderBase(ABC, hk.Module):
    """
    Decoder module.

    This decoder has an additional layer that uses an external memory to retrieve
    data and uses this data to update the context vector.
    """

    def __init__(
        self,
        num_heads,
        key_size,
        model_size=128,
        name="decoder",
        memory=None,
        memory_processing=None,
        normalizer="mean/std",
        additional_data=False,
        interaction_terms=False,
        key_chunk_size=100,
        query_chunk_size=100,
        memory_usage_flags=None,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
        self.memory = memory
        self.memory_processing_cfg = memory_processing
        self.normalizer = normalizer
        self.additional_data = additional_data
        self.interaction_terms = interaction_terms
        self.key_chunk_size = key_chunk_size
        self.query_chunk_size = query_chunk_size
        self.memory_usage_flags = memory_usage_flags

    def __call__(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        embeddings: Array,
        behavior_marker: Array,
        memory_state: ExternalMemoryState,
    ) -> Array:
        context = self.get_context(observation, embeddings)

        # add the behavior marker to the context
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

        # adding the behavior marker to the key/value as well
        mha_value = embeddings
        repeated_behavior_marker = jnp.repeat(
            jnp.expand_dims(behavior_marker, axis=0),
            repeats=embeddings.shape[0],
            axis=0,
        )

        mha_value = jnp.concatenate([embeddings, repeated_behavior_marker], axis=1)
        mha_key = mha_value

        attention_mask = jnp.expand_dims(observation.action_mask, (0, 1))

        context = mha(
            query=context,
            key=mha_key,
            value=mha_value,
            mask=self.get_transformed_attention_mask(attention_mask),
        )  # --> [128]

        # retrieve data
        data = self.memory.retrieve_all(memory_state.data, observation.position)
        current_size = jnp.maximum(
            memory_state.current_size[0], 1
        )  # +1 to avoid division by zero

        mask = (jnp.arange(data.logprob.shape[-2]) < current_size)[:, None]
        epsilon = 1e-5

        # # create additional data to be used
        # remaining budget
        remaining_budget = 1 - memory_state.current_budget[0] / memory_state.budget[0]

        def normalize(x, type="mean/std"):
            if type == "mean/std":
                return (x - jnp.mean(x, axis=-2, where=mask)) / (
                    jnp.std(x, axis=-2, where=mask) + epsilon
                )
            elif type == "min/max":
                return (x - jnp.min(x, axis=-2, where=mask, initial=0)) / (
                    (
                        jnp.max(x, axis=-2, where=mask, initial=0)
                        - jnp.min(x, axis=-2, where=mask, initial=0)
                    )
                    + epsilon
                )
            else:
                return x

        (
            normalized_logp,
            normalized_mem_logp,
            normalized_traj_logp,
            normalized_end_traj_logp,
            normalized_returns,
            normalized_age,
        ) = jax.tree_util.tree_map(
            lambda x: normalize(x, self.normalizer),
            (
                data.logprob,
                data.mem_logprob,
                data.traj_logprob,
                data.end_traj_logprob,
                data.returns,
                data.age,
            ),
        )

        # # re-normalise with respect to the memory usage start
        if False and self.memory_usage_flags.budget_trick:
            remaining_budget_input = (
                remaining_budget
            ) / self.memory_usage_flags.remaining_budget_start
        else:
            remaining_budget_input = remaining_budget

        # prepare the data to be used as input to the MLP
        data_weight = jnp.concatenate(
            [
                normalized_logp,
                normalized_returns,
                jnp.ones_like(normalized_returns) * remaining_budget_input,
            ],
            axis=-1,
        )

        if self.additional_data:
            additional_data_weight = jnp.concatenate(
                [
                    normalized_age,
                    normalized_mem_logp,
                    normalized_traj_logp,
                    normalized_end_traj_logp,
                ],
                axis=-1,
            )

            data_weight = jnp.concatenate(
                [data_weight, additional_data_weight], axis=-1
            )

        if hasattr(observation, "capacity"):
            normalized_capacity_delta = jnp.abs(observation.capacity - data.capacity)
            normalized_capacity_delta = (
                normalized_capacity_delta
                - jnp.mean(normalized_capacity_delta, axis=-2, where=mask)
            ) / (jnp.std(normalized_capacity_delta, axis=-2, where=mask) + epsilon)

            data_weight = jnp.concatenate(
                [data_weight, normalized_capacity_delta], axis=-1
            )

        if self.interaction_terms:
            n_features = data_weight.shape[-1]
            interaction_terms = jnp.array(
                [
                    data_weight[:, i] * data_weight[:, j]
                    for i in range(n_features)
                    for j in range(i, n_features)
                ]
            ).T

            # extended_data_weight = interaction_terms
            data_weight = jnp.concatenate([data_weight, interaction_terms], axis=-1)

        # design the neural network based on the config
        output_sizes = (
            self.memory_processing_cfg.mlp.hidden_size,
        ) * self.memory_processing_cfg.mlp.num_layers

        output_sizes += (1,)

        if self.memory_processing_cfg.mlp.activation == "gelu":
            activation = jax.nn.gelu
        elif self.memory_processing_cfg.mlp.activation == "relu":
            activation = jax.nn.relu
        elif self.memory_processing_cfg.mlp.activation == "tanh":
            activation = jnp.tanh
        else:
            raise ValueError(
                f"Activation {self.memory_processing_cfg.mlp.activation} not supported"
            )

        if self.memory_processing_cfg.mlp.init_zero:
            w_init = hk.initializers.Constant(0)
            b_init = hk.initializers.Constant(0)
        else:
            w_init = None  # will trigger default initialization
            b_init = None

        mlp = hk.nets.MLP(
            output_sizes=list(output_sizes),
            w_init=w_init,
            b_init=b_init,
            activation=activation,
            activate_final=False,
            name="weight_mlp_memory",
        )

        data_weight = mlp(data_weight).squeeze(-1)

        data_weight = jnp.where(mask.squeeze(-1), data_weight, 0)

        attn_logits = (
            embeddings @ context.squeeze() / jnp.sqrt(self.model_size)
        )  # --> [num_cities/items]

        # combine the attention logits with the data weights using data.action
        one_hot_actions = jax.nn.one_hot(
            data.action.squeeze(), num_classes=attn_logits.shape[-1]
        )
        weighted_one_hot = one_hot_actions * data_weight[..., None]
        new_logits = jnp.mean(weighted_one_hot, axis=0)

        if self.memory_usage_flags.budget_trick:
            mem_logits = new_logits * (
                remaining_budget < self.memory_usage_flags.remaining_budget_start
            )
        elif self.memory_usage_flags.steps_trick:
            usage_condition = (memory_state.current_size[0] > 0) * (
                ((remaining_budget > 0.6) * (remaining_budget < 0.8))
                + ((remaining_budget > 0.4) * (remaining_budget < 0.5))
                + (remaining_budget < 0.2)
            )
            mem_logits = new_logits * usage_condition
        else:
            mem_logits = new_logits * (memory_state.current_size[0] > 0)

        if self.memory_usage_flags.use_memory:
            logits = attn_logits + mem_logits
        else:
            logits = attn_logits  # + mem_logits

        logits = 10 * jnp.tanh(logits)  # clip to [-10,10]

        metrics = {}

        return logits, attn_logits, mem_logits, context, metrics

    @abstractmethod
    def get_context(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        embeddings: Array,
    ) -> Array:
        pass

    @abstractmethod
    def get_transformed_attention_mask(self, attention_mask: Array) -> Array:
        pass

    @abstractmethod
    def retrieve(
        self,
        memory_state: ExternalMemoryState,
        current_node: int,
    ):
        pass

    @abstractmethod
    def get_retrieval_metrics(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        memory_state: ExternalMemoryState,
    ):
        pass
