import typing as tp
import tensorflow as tf
import jaxtyping as ttf
from keras import layers


class FeedForwardNetwork(layers.Layer):
    """Feed Forward Network Layer"""

    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense1 = layers.Dense(mlp_dim, activation=tf.nn.gelu)
        self.dense2 = layers.Dense(embed_dim)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_layer = layers.Dropout(dropout)

    def call(self, inputs: ttf.Float[tf.Tensor, "B T D"], training: tp.Optional[bool] = None) -> ttf.Float[tf.Tensor, "B T D"]:
        ffn_output = self.dense1(inputs)
        ffn_output = self.dropout_layer(ffn_output, training=training)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout_layer(ffn_output, training=training)
        skip_conn_output = layers.add([inputs, ffn_output])
        return self.layernorm(skip_conn_output)

    def get_config(self) -> dict[str, tp.Any]:
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim,
                      "mlp_dim": self.mlp_dim, "dropout": self.dropout})
        return config

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape
