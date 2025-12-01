import keras


def InceptionV3(image_size: tuple[int, int], num_classes: int, dropout: float) -> keras.models.Model:
    model = keras.applications.inception_v3.InceptionV3(
        include_top=True,
        weights=None,
        classes=num_classes,
        pooling="avg",
        input_shape=(*image_size, 3)
    )

    return model
