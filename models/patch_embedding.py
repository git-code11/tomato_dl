import typing as tp
import tensorflow as tf
import jaxtyping as ttf
from keras import layers


class PatchEmbedding(layers.Layer):
    """Generate patch embedding from image
    STEP 1:
    Create patches from image using Convolution
    with kernel size (patch_size,) and stride (patch_size)
    and zero padding to allowing consistent output and project its channel to embed_dim

    STEP2:
    Reshape the output to have a shape  of (BATCH, SEQ, EMBED_DIMS)

    STEP3:
    Concatenate a learnable weight [CLS-TOKEN] of shape (1, EMBED_DIMS) to the each of the sequence

    STEP4:
    Add the positional embedding to the input combined embed and shape of (BATCH, SEQ + 1, EMBED_DIMS)

    """
    # using only a since convolution layer to extract information
    # which would lead to less features been extracted which would affect later layers
    # self.conv1 = layers.Conv2D(
    #       filters=embed_dim,
    #       kernel_size=patch_size,
    #       strides=patch_size,
    #       padding="valid")

    def __init__(self, patch_size: tuple[int, int], embed_dim: int, use_proj_conv: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_proj_conv = use_proj_conv

        if use_proj_conv:
            # for experimental purpose
            self.project_patches = layers.Conv2D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="valid")
        else:
            self.project_patches = layers.Dense(embed_dim)

    def seq_length(self, input_shape: tuple[int, int, int, int]) -> int:
        height, width = input_shape[1:3]
        return (height // self.patch_size[0]) * (width // self.patch_size[1])

    def _project_image(self, images: ttf.Float[tf.Tensor, "B H W C"]) -> ttf.Float[tf.Tensor, "B P D"]:
        # get patches from images and project channel to embed_dim
        batch_size = tf.shape(images)[0]

        if not self.use_proj_conv:
            images = tf.image.extract_patches(images=images,
                                              sizes=[1, *self.patch_size, 1],
                                              strides=[1, *self.patch_size, 1],
                                              rates=[1, 1, 1, 1],
                                              # (B, H//P, W//P, P*P*3)
                                              padding='VALID')
        patches = self.project_patches(images)
        # shape (batch, seq, embed)
        patches = tf.reshape(patches, (batch_size, -1, self.embed_dim))
        return patches

    def build(self, input_shape: tuple[int, ...]) -> tp.NoReturn:
        seq_length = self.seq_length(input_shape)
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,)
        self.position_embedding = layers.Embedding(
            input_dim=seq_length + 1, output_dim=self.embed_dim)
        super().build(input_shape)

    def call(self, images: ttf.Float[tf.Tensor, "B H W C"]) -> ttf.Float[tf.Tensor, 'B P+1 D']:
        # get patches from images and project channel to embed_dim
        batch_size = tf.shape(images)[0]
        patches = self._project_image(images)  # shape (batch, seq, embed)
        seq_length = tf.shape(patches)[1]

        # concat class_token to patches
        class_emb = tf.broadcast_to(
            self.cls_token, [batch_size, 1, self.embed_dim]
        )
        patches = tf.concat([class_emb, patches], axis=1)

        # Add the positional emebeding to the input combined embed
        positions = tf.range(start=0, limit=seq_length + 1)
        # same as tf.expand_dims(positions, axis=0)
        positions = positions[tf.newaxis, ...]
        pos_embed = self.position_embedding(positions)
        patches = patches + pos_embed
        return patches  # shape (batch_size, seq + 1, embed_dim)

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, int, int]:
        seq_length = self.seq_length(input_shape)
        return (input_shape[0], seq_length + 1, self.embed_dim)

    def get_config(self) -> dict[str, tp.Any]:
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "use_proj_conv": self.use_proj_conv
        })
        return config
