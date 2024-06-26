### This is a one-line change from the original attention.py file from Haiku
### The goal is to replace the original attention module with the efficient attention module


# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""(Multi-Head) Attention module for use in Transformer architectures."""

import types
import warnings
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku._src import basic, initializers, module

from memento.networks.efficient_attention_utils import efficient_dot_product_attention


class EfficientMultiHeadAttention(hk.Module):
    """Multi-headed attention (MHA) module.

    This module is intended for attending over sequences of vectors.

    Rough sketch:
    - Compute keys (K), queries (Q), and values (V) as projections of inputs.
    - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
    - Output is another projection of WV^T.

    For more detail, see the original Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762.

    Glossary of shapes:
    - T: Sequence length.
    - D: Vector (embedding) size.
    - H: Number of attention heads.
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        # TODO(b/240019186): Remove `w_init_scale`.
        w_init_scale: Optional[float] = None,
        *,
        w_init: Optional[hk.initializers.Initializer] = None,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        name: Optional[str] = None,
        precision=jax.lax.Precision.HIGHEST,
        query_chunk_size=100,
        key_chunk_size=100,
    ):
        """Initialises the module.

        Args:
          num_heads: Number of independent attention heads (H).
          key_size: The size of keys (K) and queries used for attention.
          w_init_scale: DEPRECATED. Please use w_init instead.
          w_init: Initialiser for weights in the linear map.
          value_size: Optional size of the value projection (V). If None, defaults
            to the key size (K).
          model_size: Optional size of the output embedding (D'). If None, defaults
            to the key size multiplied by the number of heads (K * H).
          name: Optional name for this module.
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.precision = precision
        self.query_chunk_size = query_chunk_size
        self.key_chunk_size = key_chunk_size

        # Backwards-compatibility for w_init_scale.
        if w_init_scale is not None:
            warnings.warn(
                "w_init_scale is deprecated; please pass an explicit weight "
                "initialiser instead.",
                DeprecationWarning,
            )
        if w_init and w_init_scale:
            raise ValueError("Please provide only `w_init`, not `w_init_scale`.")
        if w_init is None and w_init_scale is None:
            raise ValueError("Please provide a weight initializer: `w_init`.")
        if w_init is None:
            w_init = hk.initializers.VarianceScaling(w_init_scale)
        self.w_init = w_init

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.

        Args:
          query: Embeddings sequence used to compute queries; shape [..., T', D_q].
          key: Embeddings sequence used to compute keys; shape [..., T, D_k].
          value: Embeddings sequence used to compute values; shape [..., T, D_v].
          mask: Optional mask applied to attention weights; shape [..., H=1, T', T].

        Returns:
          A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """

        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        *leading_dims, sequence_length, _ = query.shape
        projection = self._linear_projection

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = projection(query, self.key_size, "query")  # [T', H, Q=K]
        key_heads = projection(key, self.key_size, "key")  # [T, H, K]
        value_heads = projection(value, self.value_size, "value")  # [T, H, V]

        attn = efficient_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            mask,
            precision=self.precision,
            query_chunk_size=self.query_chunk_size,
            key_chunk_size=self.key_chunk_size,
        )

        attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = hk.Linear(self.model_size, w_init=self.w_init)
        return final_projection(attn)  # [T', D']

    @hk.transparent
    def _linear_projection(
        self,
        x: jnp.ndarray,
        head_size: int,
        name: Optional[str] = None,
    ) -> jnp.ndarray:
        y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))
