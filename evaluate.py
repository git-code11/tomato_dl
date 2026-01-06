# Evaluate dataset based on specific model
import typing as tp
import os
import argparse
from dataclasses import dataclass
import importlib
import pathlib
from omegaconf import OmegaConf, MISSING
from tomato_dl.training.base import AbstractTrainer
# import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

CONFIG_PATH = r"training/conf"
CONFIG_NAME = "config"

BASE_CHECKPOINT_DIR = pathlib.cwd()


@dataclass
class ModelConfig:
    name: str = MISSING
    trainer: str = MISSING
    model_params: dict[str, tp.Any] = MISSING
    training_params: dict[str, tp.Any] = MISSING


def evaluate_model(trainer: AbstractTrainer, base_dir: os.PathLike) -> None:
    metrics = trainer.train_inference()
    metrics.save_fig(base_dir /
                     "train_confusion_matrix.jpg")
    data = metrics.to_series()
    df = pd.DataFrame(dict(train=data))
    df.to_csv(base_dir /
              "metrics.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("prefix", help="ID for the current evaluation")
    parser.add_argument("--config", required=True, help="Model config")
    parser.add_argument("--weights", required=True, help="Model weights")
    parser.add_argument("--dataset", required=True,
                        help="Dataset folder in class based arrangement")

    params = parser.parse_args()

    config_path = params['config']
    schema = OmegaConf.structured(ModelConfig)
    cfg = OmegaConf.load(config_path)
    cfg['training_params']['weights'] = params['weights']
    cfg = OmegaConf.merge(schema, cfg)

    base_dir = BASE_CHECKPOINT_DIR / params['prefix']
    dataset_dir = params['config']

    # Load Trainer class
    config = cfg['model']
    trainer_module = importlib.import_module(config['trainer'])
    Trainer = trainer_module.Trainer

    trainer: AbstractTrainer = Trainer(
        config, base_dir, dataset_dir)
    trainer.prepare()

    evaluate_model(trainer, base_dir)
