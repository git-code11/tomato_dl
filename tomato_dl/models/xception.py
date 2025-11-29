
# @title Xception
class Xception:
    def __new__(self,  num_classes=3, input_shape=(None, None, 3), include_top=False):
        return self.create_model(input_shape, num_classes, include_top)

    @classmethod
    def entry_separable_block(cls, input, filters):
        x = layers.SeparableConv2D(
            filters, 3, strides=1, padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)  # tf.nn.relu(x)
        x = layers.SeparableConv2D(filters, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        return x

    @classmethod
    def middle_separable_block(cls, input, filters):
        x = input
        for i in range(2):  # 3
            x = layers.ReLU()(x)
            x = layers.SeparableConv2D(
                filters, 3, strides=1, padding='same')(input)
            x = layers.BatchNormalization()(x)
        x = layers.add([x, input])
        return x

    @classmethod
    def entry_flow(cls, input):
        # Block 1
        x = layers.Conv2D(32, 3, strides=2, padding='valid')(input)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)  # tf.nn.relu(x)
        x = layers.Conv2D(64, 3, strides=1, padding='valid')(x)
        x = layers.BatchNormalization()(x)
        x_block1 = layers.ReLU()(x)  # tf.nn.relu(x)
        x_block1_residual = layers.Conv2D(
            128, 1, strides=2, padding='same')(x_block1)

        # Block 2
        x = cls.entry_separable_block(x_block1, 128)
        x_block2 = layers.add([x, x_block1_residual])
        x_block2_residual = layers.Conv2D(
            256, 1, strides=2, padding='same')(x_block2)

        # Block 3
        x = cls.entry_separable_block(x_block2, 256)
        x_block3 = layers.add([x, x_block2_residual])
        x_block3_residual = layers.Conv2D(
            512, 1, strides=2, padding='same')(x_block3)

        # Block4
        x = cls.entry_separable_block(x_block3, 512)
        x_block4 = layers.add([x, x_block3_residual])
        return x_block4

    @classmethod
    def middle_flow(cls, input, inner_block_count=3):  # 4 #8
        x = input
        for i in range(inner_block_count):
            x = cls.middle_separable_block(x, 512)
        return x

    @classmethod
    def exit_flow(cls, input):
        # Block 1
        x = layers.ReLU()(input)
        x = layers.SeparableConv2D(512, 3, strides=1, padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.SeparableConv2D(768, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        residual = layers.Conv2D(768, 1, strides=2, padding='same')(input)
        x = layers.add([x, residual])

        # Block 2
        x = layers.SeparableConv2D(1024, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.SeparableConv2D(1536, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    @classmethod
    def classifier(cls, input, num_classes):
        x = layers.GlobalAveragePooling2D()(input)
        # x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(num_classes, activation="softmax")(x)
        # x = tf.nn.softmax(x)
        return x

    @classmethod
    def create_model(cls, input_shape, num_classes, include_top):
        input = layers.Input(shape=input_shape)
        x = cls.entry_flow(input)
        x = cls.middle_flow(x)
        x = cls.exit_flow(x)
        if include_top:
            x = cls.classifier(x, num_classes)
        model = Model(inputs=input, outputs=x, name="Xception")
        return model


def load_xception(weight=None):
    xmodel = Xception(include_top=True, num_classes=num_classes)
    xmodel.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                   loss=losses.SparseCategoricalCrossentropy(
                       from_logits=False),  # since softmax is already added
                   metrics=['accuracy'])
    if weight:
        xmodel.load_weights(weight)
    return xmodel
