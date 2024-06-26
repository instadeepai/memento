from typing import Optional, Tuple

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
        behavior_marker: chex.Array,
        external_memory: ExternalMemory,
    ):
        decoder = hydra.utils.instantiate(cfg.decoder, name="decoder")
        return decoder(observation, embeddings, behavior_marker, external_memory)

    return Networks(
        encoder_fn=hk.without_apply_rng(hk.transform(encoder_fn)),
        decoder_fn=hk.without_apply_rng(hk.transform(decoder_fn)),
    )


def get_layers_with_offset_names(cfg):
    base_name = "decoder/mha_dec/"
    layers = ()
    if cfg.query:
        layers += (base_name + "query",)
    if cfg.key:
        layers += (base_name + "key",)
    if cfg.value:
        layers += (base_name + "value",)

    return layers


def sync_params_and_offset(
    current_params: hk.Params,
    target_params: hk.Params,
    offset_size: int,
    layers_names: Optional[Tuple[str, ...]] = None,
):
    """Init the decoder params from another decoder params.
    This suppose the same structure but potentially different shapes.

    The given layers are updated with the given params by adding an
    offset matrix with zeros to match the shape of the conditioned
    decoder.

    Args:
        params: params used to init
        layers_names: layers to update with an offset
        offset_size: size of the offset - often equal to the pop_size
    """
    if layers_names is None:
        layers_names = ()

    # first: merge the params - normal decoder last to override!
    params = hk.data_structures.merge(current_params, target_params)

    # cond_decoder_shapes = jax.tree_util.tree_map(
    #     lambda x: x.shape, conditioned_decoder_params
    # )

    # print("COnd decoder params: ", cond_decoder_shapes)

    new_params = {}

    for layer_name in layers_names:

        layer_matrix = target_params[layer_name]["w"]

        # print("Query matrix shape: ", layer_matrix.shape)
        # print("Query matrix: ", layer_matrix)

        offset_shape = (offset_size,) + layer_matrix.shape[1:]
        offset_matrix = jnp.zeros(shape=offset_shape)

        print("Offset matrix: ", offset_matrix.shape)

        new_layer_matrix = jnp.concatenate([layer_matrix, offset_matrix], axis=0)

        # print("New query matrix shape: ", new_layer_matrix.shape)
        # print("New query matrix: ", new_layer_matrix)

        new_params[layer_name] = {"w": new_layer_matrix}

    params = hk.data_structures.merge(params, new_params)

    return params
