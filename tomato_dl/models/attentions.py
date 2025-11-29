import typing as tp
import tensorflow as tf
import jaxtyping as ttf
from keras import layers


class MultiHeadBaseAttention(layers.Layer):
    """Multi-head Base-Attention Layer
      NOTE: embed_dims should be divisible by num_heads
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, keep_attention_scores: tp.Optional[bool] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_scores = None
        # When set to bool would override the passed value in `call()`
        self.keep_attention_scores = keep_attention_scores
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, query: ttf.Float[tf.Tensor, "B T D"], context: ttf.Float[tf.Tensor, "B S D"],
             return_attention_scores: bool = False, training: tp.Optional[bool] = None) -> ttf.Float[tf.Tensor, "B T D"]:
        # NOTE: the key = value = context;
        # This ensures that previous attention_scores is always removed to keep memory
        self.attention_scores = None
        return_attention_scores = return_attention_scores if self.keep_attention_scores is None else self.keep_attention_scores
        attn_output = self.mha(query=query, value=context,
                               return_attention_scores=return_attention_scores, training=training)
        if return_attention_scores:
            attn_output, attn_scores = attn_output
            self.attention_scores = attn_scores
        skip_conn_output = layers.add([query, attn_output])
        return self.layernorm(skip_conn_output)

    def get_config(self) -> dict[str, tp.Any]:
        config = super().get_config()
        config.update({"num_heads": self.num_heads,
                      "embed_dim": self.embed_dim, "dropout": self.dropout})
        return config

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape


class MultiHeadSelfAttention(MultiHeadBaseAttention):
    """Multi-head Self-Attention Layer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, x: ttf.Float[tf.Tensor, "B T D"], *args, **kwargs) -> ttf.Float[tf.Tensor, "B T D"]:
        return super().call(x, x, *args, **kwargs)


class MultiHeadCrossAttention(MultiHeadBaseAttention):
    """Multi-head Self-Attention Layer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
