# Training model scripts
import os
import time
import pathlib
import pandas as pd
import keras
from utils.load_model_helper import VitConfig, load_vit
from base import BaseTrainer, DatasetDict

BASE_DIR = pathlib.Path.cwd()

DATASET_DIR = BASE_DIR / "datasets"

BASE_CHECKPOINT_DIR = BASE_DIR / "outputs/vit"

# @title Paramaters Definition
# Define specific parameters
batch_size = 32
image_size = (256, 256)     # Input image size (256x256)
patch_size = (32, 32)     # Size of patches to divide the image into
num_classes = 3    # Number of output classes Infered from the loaded dataset classes
embed_dim = 256     # Embedding dimension
num_heads = 2       # Number of attention heads
mlp_dim = 128      # Dimension of the MLP in the Transformer block
num_layers = 4     # Number of Transformer Encoder layers
dropout_rate = 0.05  # Dropout rate


# Declare the VitConfig Parameters
vit_config = VitConfig(
    learning_rate=1e-4,
    weight_decay=1e-2,
    image_size=image_size,
    patch_size=patch_size,
    embed_djm=embed_dim,
    mlp_dim=mlp_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout_rate=dropout_rate,
    num_classes=num_classes,
)


class VitTrainer(BaseTrainer):
    def __init__(self, base_checkpoint_dir: os.PathLike, dataset_dir: os.PathLike):
        super().__init__()
        self.base_dir = base_checkpoint_dir
        self.dataset_dir = dataset_dir

    def load_model(self):
        # Intialize the model
        return load_vit(vit_config)

    def load_datasets(self):
        (train_val_ds, test_ds) = keras.utils.image_dataset_from_directory(
            self.dataset_dir,
            shuffle=True,
            validation_split=0.1,
            label_mode='int',
            subset="both",
            seed=57,
            image_size=image_size,
            batch_size=batch_size)

        self._display_labels = train_val_ds.class_names
        return DatasetDict(
            train_ds=train_val_ds,
            test_ds=test_ds
        )

    @property
    def display_labels(self) -> list[str]:
        return self._display_labels

    @property
    def callbacks(self) -> list[keras.callbacks.Callback]:
        return [
            keras.callbacks.TensorBoard(log_dir=self.base_dir / "logs"),
            keras.callbacks.ModelCheckpoint(
                save_weights_only=True,
                save_best_only=True,
                filepath=self.base_dir/'checkpoints/{epoch:02d}-{val_loss:.2f}.weights.h5'),
            keras.callbacks.EarlyStopping(
                patience=5,
                verbose=1,
                restore_best_weights=True),
        ]


if __name__ == "__main__":
    vit_trainer = VitTrainer(BASE_CHECKPOINT_DIR, DATASET_DIR)
    vit_trainer.prepare()
    history = vit_trainer.run(1)
    # plot history graph
    timestamp = time.time_ns()
    vit_trainer.plot_history(
        0, file_path=BASE_CHECKPOINT_DIR / f"vit_train_history-{timestamp}.jpg"
    )
    metrics = vit_trainer.inference()
    metrics.save_fig(BASE_CHECKPOINT_DIR /
                     f"vit_train_confusion_matrix-{timestamp}.jpg")
    data = metrics.to_series()
    pd.to_csv(data, BASE_CHECKPOINT_DIR /
              f"vit_metrics-{timestamp}.jpg")
