import pathlib
from omegaconf import DictConfig
import hydra

CONFIG_PATH = r"conf"
CONFIG_NAME = "config"


@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name=CONFIG_NAME)
def train_model(cfg: DictConfig) -> None:
    import pandas as pd
    from trainer.vit import VitTrainer

    BASE_DIR = pathlib.Path.cwd()
    BASE_CHECKPOINT_DIR = pathlib.Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    config = cfg['model']

    EPOCH = config['training_params'].get('epoch', 10)
    # datasets directory with reference to base dir
    DATASET = config['training_params'].get('datasets', 'datasets')

    DATASET_DIR = BASE_DIR / DATASET

    print(f"{config=}")
    print(f"{DATASET_DIR=}")
    # return

    vit_trainer = VitTrainer(config, BASE_CHECKPOINT_DIR, DATASET_DIR)
    vit_trainer.prepare()
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
    sys.path.append('./')
    train_model()
