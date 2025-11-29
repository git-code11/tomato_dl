# A training abstract class
import typing as tp
from abc import ABC, abstractmethod
from functools import reduce
import operator
import pathlib
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


class BaseTrainer(ABC):
    model_train_history: list[keras.callbacks.History]
    ds: DatasetDict
    model: keras.models.Model

    def __init__(self):
        self.model_train_history = []

    def prepare(self):
        self.ds = self.load_datasets()
        self.model = self.load_model()

    @abstractmethod
    def load_model(self) -> keras.models.Model:
        ...

    @abstractmethod
    def load_datasets(self) -> DatasetDict:
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
        train_ds = self.ds.get('train_ds').take(1)
        val_ds = self.ds.get('val_ds').take(1)
        _train_ds = train_ds.map(lambda x, y: (self.preprocess(x), y))
        _val_ds = val_ds.map(lambda x, y: (self.preprocess(x), y))
        callbacks = self.callbacks
        history = self.model.fit(_train_ds, epochs=epoch,
                                 validation_data=_val_ds,
                                 callbacks=callbacks)
        self.model_train_history.append(history)
        return history

    def inference(self, ds: OptionalDataset = None, *, kind='test_ds') -> MetricsGroup:
        if ds is None:
            ds = self.ds.get(kind).take(1)
        ds = ds.map(lambda x, y: (self.preprocess(x), y))
        metrics = MetricsHelper.get_metrics(
            ds, model=self.model, display_labels=self.display_labels)
        return metrics

    def _history_metrics(self, key: str, *idxs: list[int, ...]) -> list[float]:
        history = operator.itemgetter(*idxs)(self.model_train_history)
        if len(idxs) == 1:
          history = [history]
        metrics = reduce(
            lambda acc, x: [*acc, *x.history[key]], history, [])
        return metrics

    def plot_history(self, *idxs: list[int], file_path: tp.Optional[str] = None) -> plt.Figure:
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
