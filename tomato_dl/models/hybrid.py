import typing as tp
import jaxtyping as ttf
import tensorflow as tf
import keras
from keras import Model, layers
from .transformers import TransformerDecoder


@keras.saving.register_keras_serializable()
class HybridModel(Model):
    """Hybrid Model"""

    def __init__(self,
                 num_classes: int,
                 dropout: float, num_layers: int,
                 decoder_config: dict[str, tp.Any],
                 cnn_extractor: keras.Model,
                 vit_extractor: keras.Model,
                 use_vit_as_key=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.model_config = dict(
            use_vit_as_key=use_vit_as_key,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            decoder_config=decoder_config,
        )

        # Extractor
        self.cnn_extractor = cnn_extractor
        self.vit_extractor = vit_extractor
        self.dropout = layers.Dropout(dropout)

        embed_dim = decoder_config['embed_dim']

        # CNN Part
        self.cnn_cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, embed_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True)

        self.cnn_project_patches = layers.Dense(embed_dim)

        # Decoder Part
        self.decoder_layers = [
            TransformerDecoder(
                **dict(dropout=dropout, **decoder_config)
            ) for _ in range(num_layers)
        ]

        # Classifier
        self.mlp_head = layers.Dense(
            num_classes, name="classifier", activation="softmax")

    def call(self, inputs: ttf.Float[tf.Tensor, "B H W C"], training: tp.Optional[bool] = False):
        # Extract global features
        x0 = self.vit_extractor(inputs, training=training)

        # Extract local features
        x1 = self.cnn_extractor(inputs, training=training)
        # Project Extracted local fateaures
        x1 = self.cnn_project_patches(x1)
        cnn_feature_shape = tf.shape(x1)
        batch_size, embed_size = cnn_feature_shape[0], cnn_feature_shape[-1]
        # Reshape features
        # shape (batch, seq, embed)
        x1 = tf.reshape(x1, (batch_size, -1, embed_size))
        # Concat cls_token
        repeat_cls_token = tf.repeat(self.cnn_cls_token, batch_size, axis=0)
        x1 = tf.concat([repeat_cls_token, x1], axis=1)

        if self.model_config['use_vit_as_key']:
            context = x0
            inputs = x1
        else:
            context = x1
            inputs = x0
        for layer in self.decoder_layers[:-1]:
            inputs = layer(inputs, context, training=training)
        x = self.decoder_layers[-1](inputs, context, training=training)
        # Use the first token's representation for classification
        cls_output = x[:, 0]
        x = self.dropout(cls_output, training=training)
        output = self.mlp_head(x)
        # output = tf.nn.softmax(output)
        return output

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, int]:
        batch_size = input_shape[0]
        return (batch_size, self.model_config['num_classes'])

    def get_config(self):
        config = super().get_config()
        config['vit_extractor'] = keras.saving.serialize_keras_object(
            self.vit_extractor)
        config['cnn_extractor'] = keras.saving.serialize_keras_object(
            self.cnn_extractor)
        config.update(self.model_config)
        return config

    @classmethod
    def from_config(cls, config):
        config["vit_extractor"] = keras.saving.deserialize_keras_object(
            config["vit_extractor"])
        config["cnn_extractor"] = keras.saving.deserialize_keras_object(
            config["cnn_extractor"])
        config["vit_extractor"].trainable = False
        config["cnn_extractor"].trainable = False
        return cls(**config)
