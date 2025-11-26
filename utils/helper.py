import typing as tp
import jaxtyping as ttf
import tensorflow as tf
import matplotlib.pyplot as plt
import math


def convert_image_to_patches(
    image: ttf.Float[tf.Tensor, "H W C"],
    patch_size: tuple[int, int]
) -> ttf.Float[tf.Tensor, "N Ph Pw C"]:
    # Convert image to patches helper
    # Extract patches from image and
    # returns result (patch_length, patch_size, patch_size, channel)
    image_size = tf.shape(image)[:2]
    num_patches = (image_size // patch_size)
    patches = [tf.split(image_row, int(num_patches[1]), axis=1)
               for image_row in tf.split(image, int(num_patches[0]), axis=0)]
    patches = tf.stack(patches)
    patches = tf.reshape(patches, [-1, *patch_size, 3])
    return patches


def display_patch(
        patches: ttf.Float[tf.Tensor, "N Ph Pw C"],
        fig: tp.Optional[plt.Figure] = None) -> plt.Figure:
    # Display image patches
    n2 = tf.shape(patches)[0]
    n = tf.math.sqrt(tf.cast(n2, tf.float32))
    n = int(n)
    if fig is None:
        fig = plt.figure(figsize=(4, 4))
    for i, patch_img in enumerate(patches):
        ax = plt.subplot(n, n, i + 1)
        ax.imshow(patch_img)
        ax.axis("off")
    return fig


def display_images(
        images: list[tf.Tensor],
        labels: list[int],
        class_labels: list[str],
        *,
        row_size: int | None = None,
        col_size: int | None = None,
        fig: tp.Optional[plt.Figure] = None
) -> plt.Figure:
    # @title Preview samples of datasets
    image_count = len(images)
    if row_size is None and col_size is None:
        col_size = 6

    if row_size is None:
        row_size = math.ceil(image_count / col_size)
    elif col_size is None:
        col_size = math.ceil(image_count / row_size)

    if fig is None:
        fig = plt.figure(figsize=(24, 24))

    for idx, img in enumerate(images):
        ax = plt.subplot(row_size, col_size, idx + 1)
        label = class_labels[labels[idx]]
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')

    return fig
