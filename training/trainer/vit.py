# Training model scripts
from tomato_dl.training.base import BaseTrainer, DatasetDict
from tomato_dl.utils.load_model_helper import VitConfig, load_vit
import keras
import tensorflow as tf
import jaxtyping as ttf
import typing as tp
import os
import warnings
warnings.filterwarnings('ignore')


class VitTrainer(BaseTrainer):
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

    def load_model(self, weights: str | None = None):
        # Intialize the model
        # Declare the VitConfig Parameters
        config = self.config['model_params']
        training_params = self.config['training_params']
        weights = training_params.get('weights')

        vit_config = VitConfig(
            learning_rate=training_params['learning_rate'],
            weight_decay=training_params['weight_decay'],
            image_size=tuple(config['image_size']),
            patch_size=tuple(config['patch_size']),
            embed_dim=config['embed_dim'],
            mlp_dim=config['mlp_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_classes=config['num_classes'],
        )
        return load_vit(vit_config, weights)

    def preprocess(self, data: ttf.Float[tf.Tensor, "..."]) \
            -> ttf.Float[tf.Tensor, "..."]:
        return self.normalize_image(data)

    def load_datasets(self):
        (train_val_ds, test_ds) = keras.utils.image_dataset_from_directory(
            self.dataset_dir,
            shuffle=True,
            validation_split=0.1,
            label_mode='int',
            subset="both",
            seed=57,
            image_size=tuple(self.config['model_params']['image_size']),
            batch_size=self.config['training_params']['batch_size'])

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
