# Training model scripts
from tomato_dl.utils.load_model_helper import CNNModelConfig, VitModelConfig, HybridConfig, load_hybrid
from .common import BaseTrainer


class Trainer(BaseTrainer):

    def load_model(self, weights: str | None = None):
        # Intialize the model
        # Declare the VitConfig Parameters
        config = self.config['model_params']
        training_params = self.config['training_params']
        weights = training_params.get('weights')

        cnn_config = CNNModelConfig(
            image_size=tuple(config['cnn']['image_size']),
            dropout=config['cnn']['dropout'],
            num_classes=config['cnn']['num_classes'],
        )

        vit_config = VitModelConfig(
            image_size=tuple(config['vit']['image_size']),
            patch_size=tuple(config['vit']['patch_size']),
            embed_dim=config['vit']['embed_dim'],
            mlp_dim=config['vit']['mlp_dim'],
            num_heads=config['vit']['num_heads'],
            num_layers=config['vit']['num_layers'],
            dropout=config['vit']['dropout'],
            num_classes=config['vit']['num_classes'],
        )

        config = HybridConfig(
            cnn=cnn_config,
            vit=vit_config,
            cnn_kind=config['cnn']['kind'],
            vit_weights=config.get('vit_weights'),
            cnn_weights=config.get('cnn_weights'),
            learning_rate=training_params['learning_rate'],
            weight_decay=training_params['weight_decay'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_classes=config['num_classes'],
            use_vit_as_key=config['use_vit_as_key']
        )
        return load_hybrid(config, weights)
