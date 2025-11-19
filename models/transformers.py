import typing as tp
import tensorflow as tf
import jaxtyping as ttf
from keras import layers

from .attentions import MultiHeadCrossAttention, MultiHeadSelfAttention
from .feed_forward import FeedForwardNetwork


class BaseTransformerLayer(layers.Layer):
    """Transformer Baseder Layer"""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.self_attention = MultiHeadSelfAttention(
            embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, mlp_dim, dropout)

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def get_config(self) -> dict[str, tp.Any]:
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads,
                      "mlp_dim": self.mlp_dim, "dropout": self.dropout})
        return config


class TransformerEncoder(BaseTransformerLayer):
    """Transformer Encoder Layer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs: ttf.Float[tf.Tensor, "B T D"],
             return_attention_scores: bool = False,
             training: tp.Optional[bool] = None) -> ttf.Float[tf.Tensor, "B T D"]:
        x = self.self_attention(
            inputs, return_attention_scores=return_attention_scores, training=training)
        x = self.ffn(x, training=training)
        return x


class TransformerDecoder(BaseTransformerLayer):
    """Transformer Decoder Layer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_attention = MultiHeadCrossAttention(
            self.embed_dim, self.num_heads, self.dropout)

    def call(self, inputs: ttf.Float[tf.Tensor, "B T D"],
             context: ttf.Float[tf.Tensor, "B S D"],
             return_attention_scores: bool = False,
             training: tp.Optional[bool] = None) -> ttf.Float[tf.Tensor, "B T D"]:
        x = self.self_attention(
            inputs, return_attention_scores=return_attention_scores, training=training)
        x = self.cross_attention(
            x, context, return_attention_scores=return_attention_scores, training=training)
        x = self.ffn(x, training=training)
        return x
