# Training model scripts
from tomato_dl.utils.load_model_helper import CNNConfig, load_inception
from .common import BaseTrainer


class Trainer(BaseTrainer):

    def load_model(self, weights: str | None = None):
        # Intialize the model
        # Declare the VitConfig Parameters
        config = self.config['model_params']
        training_params = self.config['training_params']
        weights = training_params.get('weights')

        config = CNNConfig(
            learning_rate=training_params['learning_rate'],
            weight_decay=training_params['weight_decay'],
            image_size=tuple(config['image_size']),
            dropout=config['dropout'],
            num_classes=config['num_classes'],
        )
        return load_inception(config, weights)
