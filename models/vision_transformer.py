import typing as tp
import tensorflow as tf
import jaxtyping as ttf
from keras import Model, layers
import keras
from .vision_encoder import VisionTransformerEncoder


@keras.saving.register_keras_serializable()
class VisionTransformer(Model):
    """Vision Transformer Model"""

    def __init__(self, num_classes: int, dropout: float, vte_config: dict[str, tp.Any] = {}, **kwargs):
        super().__init__(**kwargs)
        self.model_config = dict(
            num_classes=num_classes,
            dropout=dropout,
            vte_config=vte_config
        )
        # VIT Encoder
        self.encoder = VisionTransformerEncoder(
            **dict(dropout=dropout, **vte_config))
        self.dropout = layers.Dropout(dropout)

        # Classifier
        self.mlp_head = layers.Dense(num_classes, name="classifier")

        # self.stacked = keras.Sequential([
        #     self.encoder,
        #     layers.Lambda(lambda inputs: inputs[:, 0]),
        #     self.dropout,
        #     self.mlp_head,
        #     layers.Softmax()
        # ])

    def call(self, inputs: ttf.Float[tf.Tensor, "B H W C"], training: tp.Optional[bool] = False):
        # output = self.stacked(inputs, training=training)
        # return output

        x0 = self.encoder(inputs, training=training)
        # Use the CLS token's representation for classification
        cls_output = x0[:, 0]
        x = self.dropout(cls_output, training=training)
        output = self.mlp_head(x)
        output = tf.nn.softmax(output)
        return output

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, int]:
        batch_size = input_shape[0]
        return (batch_size, self.model_config.num_classes)

    def get_config(self):
        config = super().get_config()
        config.update(self.model_config)
        return config
