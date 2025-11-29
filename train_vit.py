# Training model scripts
import os
import warnings
warnings.filterwarnings('ignore')
import time
import pathlib
import pandas as pd
import jaxtyping as ttf
import tensorflow as tf
import keras
from tomato_dl.utils.load_model_helper import VitConfig, load_vit
from tomato_dl.training.base import BaseTrainer, DatasetDict, DataSplit

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
    embed_dim=embed_dim,
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
        # Image Normalization Helper: Uses the Standardization formula (x - mean)/std
        create_normalize_image = lambda mean, std: keras.layers.Lambda(lambda x: (x - mean) / std)
        self.normalize_image = create_normalize_image(0., 255.)

    def load_model(self):
        # Intialize the model
        return load_vit(vit_config)

    def preprocess(self, data: ttf.Float[tf.Tensor, "..."]) -> ttf.Float[tf.Tensor, "..."]:
        return self.normalize_image(data)

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

        # Split train_val dataset into training and validation
        validation_size = int(train_val_ds.cardinality().numpy()*0.2)
        val_ds = train_val_ds.take(validation_size)
        train_ds = train_val_ds.skip(validation_size)

        return DatasetDict(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
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
    #history = vit_trainer.run(2)
    # plot history graph
    timestamp = time.time_ns()
    # vit_trainer.plot_history(
    #     0, file_path=BASE_CHECKPOINT_DIR / f"vit_train_history-{timestamp}.jpg"
    # )
    all_metrics = dict()
    metrics = vit_trainer.inference(kind=DataSplit.TEST)
    metrics.save_fig(BASE_CHECKPOINT_DIR /
                     f"vit_test_confusion_matrix-{timestamp}.jpg")
    all_metrics['test'] = metrics.to_series()
              
    metrics = vit_trainer.inference(kind=DataSplit.TRAIN)
    metrics.save_fig(BASE_CHECKPOINT_DIR /
                     f"vit_train_confusion_matrix-{timestamp}.jpg")
    all_metrics['train'] = metrics.to_series()

    df = pd.DataFrame(all_metrics)
    df.to_csv(BASE_CHECKPOINT_DIR /
              f"vit_metrics-{timestamp}.csv")
