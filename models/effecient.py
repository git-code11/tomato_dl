# @title Efficientnet V2
def EfficientNetV2(num_classes=3, dropout_rate=0.1):
    effnet_extractor = keras.applications.EfficientNetV2B0(
        input_shape=(None, None, 3),
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
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


def load_effnet(weight=None):
    effnet = EfficientNetV2(num_classes=num_classes)
    effnet.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                   loss=losses.SparseCategoricalCrossentropy(
                       from_logits=False),  # since softmax is already added
                   metrics=['accuracy'])
    if weight:
        effnet.load_weights(weight)
    return effnet
