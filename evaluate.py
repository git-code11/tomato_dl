# Evaluate dataset based on specific model
import typing as tp
import os
import argparse
from dataclasses import dataclass
import importlib
import pathlib
from omegaconf import OmegaConf, MISSING
from training.trainer.common import BaseTrainer
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


def evaluate_model(trainer: BaseTrainer) -> None:
    metrics = trainer.train_inference()
    metrics.save_fig(trainer.base_dir /
                     "train_confusion_matrix.jpg")
    data = metrics.to_series()
    df = pd.DataFrame(dict(train=data))
    df.to_csv(trainer.base_dir /
              "metrics.csv")


class GetTrainerArgs(tp.TypedDict):
    basedir: str
    config: str
    weights: str
    dataset: str


def get_trainer(**params: GetTrainerArgs) -> BaseTrainer:
    config_path = params['config']
    schema = OmegaConf.structured(ModelConfig)
    cfg = OmegaConf.load(config_path)
    cfg['training_params']['weights'] = params['weights']
    cfg = OmegaConf.merge(schema, cfg)

    base_dir = params['base_dir']
    dataset_dir = params['config']

    # Load Trainer class
    config = cfg['model']
    trainer_module = importlib.import_module(config['trainer'])
    Trainer = trainer_module.Trainer

    trainer = Trainer(
        config, base_dir, dataset_dir)

    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("--base_dir", default=BASE_CHECKPOINT_DIR,
                        help="ID for the current evaluation")
    parser.add_argument("--config", required=True, help="Model config")
    parser.add_argument("--weights", required=True, help="Model weights")
    parser.add_argument("--dataset", required=True,
                        help="Dataset folder in class based arrangement")

    params = parser.parse_args()
    trainer = get_trainer(*params)
    trainer.prepare()

    evaluate_model(trainer)
