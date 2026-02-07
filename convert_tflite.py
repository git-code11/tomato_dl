from evaluate import get_trainer
from tomato_dl.utils.tflite import TfliteInference
import argparse
import pathlib
import warnings
warnings.filterwarnings('ignore')


BASE_CHECKPOINT_DIR = pathlib.Path.cwd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("--basedir", default=BASE_CHECKPOINT_DIR,
                        help="ID for the current evaluation")
    parser.add_argument("--config", required=True, help="Model config")
    parser.add_argument("--weights", required=True, help="Model weights")
    parser.add_argument("--output", required=True, help="Tflite Name")

    params = parser.parse_args()
    trainer = get_trainer(**vars(params))

    model = trainer.load_model()

    if not str.endswith(params['output'], ".tflite"):
        params['output'] += ".tflite"

    print(f"Converting Model to Tflite")
    tflite = TfliteInference.convert_to_tflite(
        model, model_path=params['output'])
