import typing as tp
import tensorflow as tf
import time


class InferenceResult(tp.TypedDict):
    outputs: list[tp.Any]
    elapsed_time: float


class TfliteInference:
    _interpreter: tp.Optional[tf.lite.Interepreter]

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._interpreter = None

    def load_model(self):
        self._interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self._interpreter.allocate_tensors()
        self._output_details = self._interpreter.get_output_details()
        self._input_details = self._interpreter.get_input_details()

    def inference(self, inputs: list[tp.Any]) -> InferenceResult:
        if self._interpreter is None:
            self.load_model()

        for idx, input_ in enumerate(self.inputs):
            self._interpreter.set_tensor(
                self._input_details[idx]['index'], input_)
        start_time = time.perf_counter_ns()
        self._interpreter.invoke()
        elapsed_time = time.perf_counter_ns() - start_time
        outputs = list(map(lambda output_detail: self._interpreter.get_tensor(
            output_detail['index']), self._output_details))
        result = InferenceResult(outputs=outputs, elapsed_time=elapsed_time)
        return result

    @classmethod
    def convert_to_tflite(cls, model: tf.keras.Model, model_path: str):
        # tensorflow lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        # converter.experimental_new_converter = True
        tflite_model_raw = converter.convert()
        # Save the model.
        with open(model_path, 'wb') as f:
            f.write(tflite_model_raw)
        return cls(model_path=model_path)
