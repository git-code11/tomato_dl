import typing as tp
from dataclasses import dataclass
import importlib
import pathlib
from omegaconf import DictConfig, MISSING
import hydra
from hydra.core.config_store import ConfigStore
# import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

CONFIG_PATH = r"training/conf"
CONFIG_NAME = "config"


@dataclass
class ModelConfig:
    name: str = MISSING
    trainer: str = MISSING
    model_params: dict[str, tp.Any] = MISSING
    training_params: dict[str, tp.Any] = MISSING


@dataclass
class Config:
    model: ModelConfig = MISSING


cs = ConfigStore.instance()
cs.store(name=CONFIG_NAME, node=Config)


@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name=CONFIG_NAME)
def train_model(cfg: DictConfig) -> None:
    config = cfg['model']

    BASE_DIR = pathlib.Path.cwd()
    BASE_CHECKPOINT_DIR = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    EPOCH = config['training_params'].get('epoch', 10)
    # datasets directory with reference to base dir
    DATASET = config['training_params'].get('datasets', 'datasets')

    DATASET_DIR = BASE_DIR / DATASET

    print(f"{config=}")
    print(f"{BASE_CHECKPOINT_DIR}")
    print(f"{DATASET_DIR=}")

    # Load Trainer class
    trainer_module = importlib.import_module(config['trainer'])
    Trainer = trainer_module.Trainer

    vit_trainer = Trainer(config, BASE_CHECKPOINT_DIR, DATASET_DIR)
    vit_trainer.prepare()

    # -------TEST------
    # out = vit_trainer.model(np.random.rand(1, 256, 256, 3))
    # print(f"{out.shape=}")
    # return
    # -------TEST------

    vit_trainer.run(EPOCH)

    # plot history graph
    vit_trainer.plot_history(
        file_path=BASE_CHECKPOINT_DIR / "train_history.jpg")
    all_metrics = dict()
    metrics = vit_trainer.test_inference()
    metrics.save_fig(BASE_CHECKPOINT_DIR /
                     "test_confusion_matrix.jpg")
    all_metrics['test'] = metrics.to_series()

    metrics = vit_trainer.valid_inference()
    metrics.save_fig(BASE_CHECKPOINT_DIR /
                     "valid_confusion_matrix.jpg")
    all_metrics['valid'] = metrics.to_series()

    metrics = vit_trainer.train_inference()
    metrics.save_fig(BASE_CHECKPOINT_DIR /
                     "train_confusion_matrix.jpg")
    all_metrics['train'] = metrics.to_series()

    df = pd.DataFrame(all_metrics)
    df.to_csv(BASE_CHECKPOINT_DIR /
              "metrics.csv")


if __name__ == "__main__":
    import sys
    # sys.path.append('../')
    sys.path.append('./')
    train_model()
