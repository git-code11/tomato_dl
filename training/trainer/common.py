# Training model scripts
import typing as tp
import os
import jaxtyping as ttf
import tensorflow as tf
import keras
from tomato_dl.training.base import AbstractTrainer, DatasetDict


class BaseTrainer(AbstractTrainer):
    def __init__(self, config: dict[str, tp.Any],
                 base_checkpoint_dir: os.PathLike,
                 dataset_dir: os.PathLike):
        super().__init__()
        self.base_dir = base_checkpoint_dir
        self.dataset_dir = dataset_dir
        self.config = config
        # Image Normalization Helper:
        # Uses the Standardization formula (x - mean)/std

        def create_normalize_image(mean, std): return keras.layers.Lambda(
            lambda x: (x - mean) / std)
        self.normalize_image = create_normalize_image(0., 255.)

    def preprocess(self, data: ttf.Float[tf.Tensor, "..."]) \
            -> ttf.Float[tf.Tensor, "..."]:
        return self.normalize_image(data)

    def load_datasets(self, *, split: bool = True) \
            -> DatasetDict:
        image_size = tuple(self.config['model_params']['image_size'])
        batch_size = self.config['training_params']['batch_size']
        seed = self.config['training_params']['seed'] if split else None
        split_ratio = self.config['training_params'].get(
            'split_ratio', (0.7, 0.2, 0.1))

        if len(split_ratio) != 3:
            raise Exception(
                "Require len(split_ratio) == 2 (train, valid, test)")

        if not (all([x > 0 for x in split_ratio]) and sum(split_ratio) == 1):
            raise Exception(
                "Check split paramaters: Enure non-zero and sum to be 1")

        ds = keras.utils.image_dataset_from_directory(
            self.dataset_dir,
            shuffle=True if split else False,
            batch_size=batch_size,
            validation_split=split_ratio[2] if split else None,
            label_mode='int',
            subset="both" if split else None,
            seed=seed,
            image_size=image_size,
        )

        if not split:
            self._display_labels = ds.class_names
            return DatasetDict(
                train_ds=ds
            )

        (train_val_ds, test_ds) = ds

        self._display_labels = train_val_ds.class_names

        # Split train_val dataset into training and validation
        validation_size = (train_val_ds.cardinality() +
                           test_ds.cardinality())*split_ratio[1]
        val_ds = train_val_ds.take(validation_size)
        train_ds = train_val_ds.skip(validation_size)

        return DatasetDict(
            train_ds=train_ds,
            valid_ds=val_ds,
            test_ds=test_ds,
        )

    @property
    def display_labels(self) -> list[str]:
        return self._display_labels

    @property
    def callbacks(self) -> list[keras.callbacks.Callback]:
        patience = self.config['training_params']['patience']
        return [
            keras.callbacks.TensorBoard(log_dir=self.base_dir / "logs"),
            keras.callbacks.ModelCheckpoint(
                save_weights_only=True,
                save_best_only=True,
                filepath=self.base_dir/'checkpoints/{epoch:02d}-{val_loss:.2f}.weights.h5'),
            keras.callbacks.EarlyStopping(
                patience=patience,
                verbose=1,
                restore_best_weights=True),
        ]
