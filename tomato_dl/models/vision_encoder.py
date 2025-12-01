import typing as tp
import tensorflow as tf
import jaxtyping as ttf
from keras import Model
import keras
from .patch_embedding import PatchEmbedding
from .transformers import TransformerEncoder


@keras.saving.register_keras_serializable()
class VisionTransformerEncoder(Model):
    """Vision TransformerEncoder Block"""
    patch_embedding: PatchEmbedding
    encoder_layers: list[TransformerEncoder]

    def __init__(self, *, patch_size: tuple[int, int],
                 embed_dim: int, num_heads: int,
                 mlp_dim: int, num_layers: int,
                 dropout: float, use_proj_conv: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.block_config = dict(
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_proj_conv=use_proj_conv
        )
        self.patch_embedding = PatchEmbedding(
            patch_size, embed_dim, use_proj_conv)
        self.encoder_layers = [
            TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ]

    def call(self, inputs: ttf.Float[tf.Tensor, "B H W C"], training: tp.Optional[bool] = False) -> ttf.Float[tf.Tensor, "B T D"]:
        # Extract patches and embed them
        x0 = self.patch_embedding(inputs)
        # Apply Transformer layers
        for layer in self.encoder_layers[:-1]:
            x0 = layer(x0, training=training)
        # Ensures that last TansformerEncoder block keeps the attention matrix
        x0 = self.encoder_layers[-1](
            x0, return_attention_scores=training or False, training=training)
        return x0

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        output_shape = self.patch_embedding.compute_output_shape(input_shape)
        return output_shape

    def get_config(self) -> dict[str, tp.Any]:
        config = super().get_config()
        config.update(self.block_config)
        return config
