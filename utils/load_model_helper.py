import typing as tp
from keras import optimizers, losses, Input
from ..models.vision_transformer import VisionTransformer


class VitConfig(tp.TypedDict):
    learning_rate: float
    image_size: tuple[int, int]
    patch_size: tuple[int, int]
    embed_size: int
    num_heads: int
    mlp_dim: int
    num_layers: int
    num_classes: int
    dropout: int


def load_vit(config: VitConfig, weights: tp.Optional[str] = None, skip_init: bool = False) -> VisionTransformer:
    lr = config['learning_rate']
    image_size = config['image_size']

    vit_model = VisionTransformer(
        vte_config=dict(
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            mlp_dim=config['mlp_dim'],
            num_layers=config['num_layers'],
        ),
        num_classes=config['num_classes'],
        dropout=config['dropout_rate'],
    )

    # Compile the model
    vit_model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        # since softmax is already added
        metrics=['accuracy']
    )

    _ = vit_model(Input(shape=(*image_size, 3)))
    if weights:
        vit_model.load_weights(weights)
    return vit_model
