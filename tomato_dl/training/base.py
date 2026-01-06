# A training abstract class
import typing as tp
import enum
from abc import ABC, abstractmethod
from functools import reduce
import operator
import jaxtyping as ttf
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from ..utils import plot
from ..utils.metrics import MetricsHelper, MetricsGroup


type OptionalDataset = tp.Optional[tf.data.Dataset]


class DatasetDict(tp.TypedDict):
    train_ds: OptionalDataset
    valid_ds: OptionalDataset
    test_ds: OptionalDataset


class DataSplit(enum.Enum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


class AbstractTrainer(ABC):
    model_train_history: list[keras.callbacks.History]
    ds: DatasetDict
    model: keras.Model

    def __init__(self):
        self.model_train_history = []

    def prepare(self, **kwargs):
        self.ds = self.load_datasets(**kwargs)
        self.model = self.load_model()

    @abstractmethod
    def load_model(self) -> keras.Model:
        ...

    @abstractmethod
    def load_datasets(self, *, split: bool = True) -> DatasetDict:
        ...

    @abstractmethod
    def preprocess(self, data: ttf.Float[tf.Tensor, "B ..."]) -> ttf.Float[tf.Tensor, "B ..."]:
        return data

    @property
    @abstractmethod
    def callbacks(self) -> list[keras.callbacks.Callback]:
        ...

    @property
    @abstractmethod
    def display_labels(self) -> tp.Optional[list[str]]:
        return None

    def run(self, epoch: int) -> keras.callbacks.History:
        train_ds = self.ds.get('train_ds')  # .take(1)
        val_ds = self.ds.get('valid_ds')  # .take(1)
        _train_ds = train_ds.map(lambda x, y: (self.preprocess(x), y))
        _val_ds = val_ds.map(lambda x, y: (self.preprocess(x), y))
        callbacks = self.callbacks
        history = self.model.fit(_train_ds, epochs=epoch,
                                 validation_data=_val_ds,
                                 callbacks=callbacks)
        self.model_train_history.append(history)
        return history

    def inference(self, ds: OptionalDataset = None, *, kind: DataSplit = DataSplit.TEST) -> MetricsGroup:
        if ds is None:
            ds = self.ds.get(f"{kind.value.lower()}_ds")  # .take(1)
        ds = ds.map(lambda x, y: (self.preprocess(x), y))
        metrics = MetricsHelper.get_metrics(
            ds, model=self.model, display_labels=self.display_labels, title=kind.value.upper())
        return metrics

    def test_inference(self, ds: OptionalDataset = None) -> MetricsGroup:
        return self.inference(ds, kind=DataSplit.TEST)

    def train_inference(self, ds: OptionalDataset = None) -> MetricsGroup:
        return self.inference(ds, kind=DataSplit.TRAIN)

    def valid_inference(self, ds: OptionalDataset = None) -> MetricsGroup:
        return self.inference(ds, kind=DataSplit.VALID)

    def _history_metrics(self, key: str, *idxs: list[int, ...]) -> list[float]:
        history = operator.itemgetter(*idxs)(self.model_train_history)
        if len(idxs) == 1:
            history = [history]
        metrics = reduce(
            lambda acc, x: [*acc, *x.history[key]], history, [])
        return metrics

    def plot_history(self, *idxs: list[int], file_path: tp.Optional[str] = None) -> plt.Figure:
        if len(idxs) == 0:
            idxs = list(range(len(self.model_train_history)))

        accuracy = self._history_metrics('accuracy', *idxs)
        val_accuracy = self._history_metrics('val_accuracy', *idxs)
        loss = self._history_metrics('loss', *idxs)
        val_loss = self._history_metrics('val_loss', *idxs)

        metrics_history = plot.GraphHistory(
            accuracy=accuracy,
            val_accuracy=val_accuracy,
            loss=loss,
            val_loss=val_loss
        )

        (fig, *_) = plot.plot_graph(metrics_history)
        if file_path:
            fig.savefig(file_path)
        return fig
