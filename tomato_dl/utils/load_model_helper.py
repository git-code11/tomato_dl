from dataclasses import dataclass
import typing as tp
from keras import optimizers, losses, Input
from ..models.vision_transformer import VisionTransformer
from ..models.effecient import EfficientNetV2
from ..models.inception import InceptionV3
from ..models.xception import Xception


@dataclass
class VitConfig:
    learning_rate: float
    weight_decay: float
    image_size: tuple[int, int]
    patch_size: tuple[int, int]
    embed_dim: int
    num_heads: int
    mlp_dim: int
    num_layers: int
    num_classes: int
    dropout: int


def load_vit(config: VitConfig,
             weights: tp.Optional[str] = None) -> VisionTransformer:

    vit_model = VisionTransformer(
        vte_config=dict(
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            num_layers=config.num_layers,
        ),
        num_classes=config.num_classes,
        dropout=config.dropout,
    )

    # Compile the model
    vit_model.compile(
        optimizer=optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        # since softmax is already added
        metrics=['accuracy']
    )

    _ = vit_model(Input(shape=(*config.image_size, 3)))
    if weights:
        vit_model.load_weights(weights)

    return vit_model


@dataclass
class CNNConfig:
    learning_rate: float
    weight_decay: float
    image_size: tuple[int, int]
    num_classes: int
    dropout: int = 5e-2


def load_efficient(config: CNNConfig,
                   weights: tp.Optional[str] = None):
    model = EfficientNetV2(
        image_size=config.image_size,
        num_classes=config.num_classes, dropout=config.dropout)
    model.compile(
        optimizer=optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay),
        loss=losses.SparseCategoricalCrossentropy(
            from_logits=False),  # since softmax is already added
        metrics=['accuracy'])
    if weights:
        model.load_weights(weights)
    return model


def load_inception(config: CNNConfig,
                   weights: tp.Optional[str] = None):
    model = InceptionV3(
        image_size=config.image_size,
        num_classes=config.num_classes,
        dropout=config.dropout)
    model.compile(
        optimizer=optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay),
        loss=losses.SparseCategoricalCrossentropy(
            from_logits=False),  # since softmax is already added
        metrics=['accuracy'])
    if weights:
        model.load_weights(weights)
    return model


def load_xception(config: CNNConfig,
                  weights: tp.Optional[str] = None):
    model = Xception(
        image_size=config.image_size,
        num_classes=config.num_classes,
    )
    model.compile(
        optimizer=optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay),
        loss=losses.SparseCategoricalCrossentropy(
            from_logits=False),  # since softmax is already added
        metrics=['accuracy'])
    if weights:
        model.load_weights(weights)
    return model
