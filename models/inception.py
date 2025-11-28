inception_model = keras.applications.inception_v3.InceptionV3(
    include_top=True,
    weights=None,
    classes=num_classes,
    pooling="avg",
    input_shape=(*image_size, 3)
)
inception_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                        loss=losses.SparseCategoricalCrossentropy(
                            from_logits=False),  # since softmax is already added
                        metrics=['accuracy'])
