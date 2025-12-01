import keras
from keras import layers


def EfficientNetV2(image_size: tuple[int, int], num_classes: int, dropout: float) -> keras.models.Model:
    effnet_extractor = keras.applications.EfficientNetV2B0(
        input_shape=(*image_size, 3),
        include_top=False,
        weights=None,
        input_tensor=None,
        pooling=None,
        classifier_activation="softmax",
        include_preprocessing=False,
    )
    model = keras.Sequential([
        effnet_extractor,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model
