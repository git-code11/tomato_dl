
# @title HybridModel
class HybridModel(Model):
    """Hybrid Model"""

    def __init__(self, patch_size, num_classes, embed_dim, num_heads, mlp_dim, num_layers, dropout, cnn_extractor, vit_extractor):
        super().__init__()

        # Extractor
        self.cnn_extractor = cnn_extractor
        self.vit_extractor = vit_extractor
        self.dropout = layers.Dropout(dropout_rate)

        # CNN Part
        self.cnn_cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, embed_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True)

        # self.cnn_project_patches = layers.Conv2D(
        #     filters=embed_dim,
        #     kernel_size=1,
        #     padding="valid")

        self.cnn_project_patches = layers.Dense(embed_dim)

        # Decoder Part
        self.decoder_layers = [
            TransformerDecoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(1)
        ]

        # Classifier
        self.mlp_head = layers.Dense(
            num_classes, name="classifier", activation="softmax")

    def call(self, inputs, training=False):
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

        # x0 = context, x1 = input
        for layer in self.decoder_layers[:-1]:
            x1 = layer(x1, x0, training=training)
        x = self.decoder_layers[-1](x1, x0, training=training)
        # Use the first token's representation for classification
        cls_output = x[:, 0]
        x = self.dropout(cls_output, training=training)
        output = self.mlp_head(x)
        # output = tf.nn.softmax(output)
        return output


def load_hybrid(*, cnn, vit, weights=None,):
    model = HybridModel(
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        dropout=dropout_rate,
        cnn_extractor=cnn,
        vit_extractor=vit
    )

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                  loss=losses.SparseCategoricalCrossentropy(
                      from_logits=False),  # since softmax is already added
                  metrics=['accuracy'])
    _ = model(keras.Input(shape=(*image_size, 3)))
    if weights:
        model.load_weights(weights)
    return model
