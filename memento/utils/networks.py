from typing import Optional, Tuple

import haiku as hk
import jax.numpy as jnp


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
    
    new_params = {}

    for layer_name in layers_names:

        layer_matrix = target_params[layer_name]["w"]

        offset_shape = (offset_size,) + layer_matrix.shape[1:]
        offset_matrix = jnp.zeros(shape=offset_shape)

        print("Offset matrix: ", offset_matrix.shape)

        new_layer_matrix = jnp.concatenate([layer_matrix, offset_matrix], axis=0)

        new_params[layer_name] = {"w": new_layer_matrix}

    params = hk.data_structures.merge(params, new_params)

    return params
