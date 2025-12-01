from dataclasses import dataclass
import typing as tp
from keras import optimizers, losses, Input, Model
from ..models.vision_transformer import VisionTransformer
from ..models.effecient import EfficientNetV2
from ..models.inception import InceptionV3
from ..models.xception import Xception
from ..models.hybrid import HybridModel


@dataclass
class TrainingConfig:
    learning_rate: float
    weight_decay: float


@dataclass
class VitModelConfig:
    image_size: tuple[int, int]
    patch_size: tuple[int, int]
    embed_dim: int
    num_heads: int
    mlp_dim: int
    num_layers: int
    num_classes: int
    dropout: int


@dataclass
class VitConfig(VitModelConfig, TrainingConfig):
    ...


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
class CNNModelConfig:
    image_size: tuple[int, int]
    num_classes: int
    dropout: int = 5e-2


@dataclass
class CNNConfig(CNNModelConfig, TrainingConfig):
    ...


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


@dataclass
class HybridModelConfig:
    vit: VitModelConfig
    cnn: CNNModelConfig
    cnn_kind: str
    num_classes: int
    dropout: int
    num_layers: int
    vit_weights: str | None = None
    cnn_weights: str | None = None
    use_vit_as_key: bool = True


@dataclass
class HybridConfig(HybridModelConfig, TrainingConfig):
    ...


all_cnns = dict(
    efficient=load_efficient,
    inception=load_inception,
    xception=load_xception
)


def load_hybrid(config: HybridConfig, weights: tp.Optional[str] = None,):
    # load vit
    vit_config = VitConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        **vars(config.vit)
    )
    vit = load_vit(vit_config, weights=config.vit_weights)
    vit_extractor = vit.encoder

    # load cnn
    cnn_config = CNNConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        **vars(config.cnn)
    )
    cnn_loader = all_cnns[config.cnn_kind]
    cnn = cnn_loader(cnn_config, weights=config.cnn_weights)
    if config.cnn_kind == "efficient":
        cnn_extractor = cnn.layers[0]
    else:
        cnn_extractor = Model(
            inputs=cnn.input,
            outputs=cnn.layers[-3].output,
            name="CNNXExtractor")

    # Freeze Feature Extractor
    vit_extractor.trainable = False
    cnn_extractor.trainable = False

    model = HybridModel(
        decoder_config=dict(
            embed_dim=config.vit.embed_dim,
            num_heads=config.vit.num_heads,
            mlp_dim=config.vit.mlp_dim,
        ),
        num_classes=config.num_classes,
        dropout=config.dropout,
        num_layers=config.num_layers,
        use_vit_as_key=config.use_vit_as_key,
        cnn_extractor=cnn_extractor,
        vit_extractor=vit_extractor
    )

    # Compile the model
    model.compile(
        optimizer=optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay),
        loss=losses.SparseCategoricalCrossentropy(
            from_logits=False),  # since softmax is already added
        metrics=['accuracy'])
    _ = model(Input(shape=(*config.vit.image_size, 3)))
    if weights:
        model.load_weights(weights)
    return model
