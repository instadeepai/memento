import chex
import haiku as hk
import hydra
import jax.numpy as jnp
from memento.memory.external_memory import ExternalMemory
from memento.networks import Networks
from memento.utils.acting_utils import Observation


def get_networks(cfg) -> Networks:
    """Get networks from the config"""

    def encoder_fn(problem: chex.Array):
        encoder = hydra.utils.instantiate(cfg.encoder, name="shared_encoder")
        return encoder(problem)

    def decoder_fn(
        observation: Observation,
        embeddings: chex.Array,
        external_memory: ExternalMemory,
    ):
        decoder = hydra.utils.instantiate(cfg.decoder, name="decoder")
        return decoder(observation, embeddings, external_memory)

    return Networks(
        encoder_fn=hk.without_apply_rng(hk.transform(encoder_fn)),
        decoder_fn=hk.without_apply_rng(hk.transform(decoder_fn)),
    )
