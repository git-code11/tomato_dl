import typing as tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, roc_auc_score, f1_score,
    average_precision_score, ConfusionMatrixDisplay
)


class MetricsGroup(tp.NamedTuple):
    accuracy: float
    recall: float
    precision: float
    roc_auc: float
    average_precision: float
    f1: float
    confusion_matrix_figure: plt.Figure

    def to_series(self) -> pd.Series:
        items = {**self._asdict()}
        items.pop('confusion_matrix_figure')
        return pd.Series(items)

    def save_fig(self, img_path: str) -> tp.NoReturn:
        fig: plt.Figure = self.confusion_matrix_figure
        fig.savefig(img_path)


class MetricsHelper:

    def get_metrics(ds: tf.data.Dataset, *,
                    model: tf.keras.Model,
                    display_labels: tp.Optional[list[str]] = None,
                    title="Test") -> MetricsGroup:
        ds_target = []
        ds_pred = []
        ds_scores = []
        for (X, y) in ds.as_numpy_iterator():
            ds_target.append(y)
            pred_scores = model(X)
            pred = tf.argmax(pred_scores, axis=-1)
            ds_pred.append(pred)
            ds_scores.append(pred_scores)

        targets = tf.concat(ds_target, axis=0)
        preds = tf.concat(ds_pred, axis=0)
        preds = tf.cast(preds, tf.int32)
        scores = tf.concat(ds_scores, axis=0)

        targets = targets.numpy()
        preds = preds.numpy()
        scores = scores.numpy()

        class_size = np.shape(scores)[-1]

        assert (np.shape(preds) == np.shape(targets), "Shape is different")

        accuracy = accuracy_score(targets, preds)
        recall = recall_score(targets, preds, average='macro')
        precision = precision_score(targets, preds, average='macro')
        f1 = f1_score(targets, preds, average='macro')
        _target = tf.one_hot(tf.constant(targets, dtype=tf.int32), class_size).numpy()
        avg_precision = average_precision_score(
            _target, scores, average="macro")
        
        roc_auc = roc_auc_score(
            _target, scores, multi_class='ovr', average='macro')
        # print(f"AUC: {auc}")
        # print(f"Accuracy: {accuracy}")
        # print(f"F1 Score: {f1}")
        # print(f"Recall: {recall}")
        # print(f"Precision: {precision}")

        # total = tf.shape(targets)[0]
        # total_equal = tf.reduce_sum(tf.cast(targets == preds, tf.int32))
        # not_equal = total - total_equal
        # print(f"Total Equal: {total_equal}, Total Not Equal: {not_equal}")
        dp = ConfusionMatrixDisplay.from_predictions(
            targets, preds, display_labels=display_labels)
        dp.figure_.suptitle(f"Confusion Matrix on {title} Dataset")

        return MetricsGroup(
            accuracy=accuracy,
            recall=recall,
            precision=precision,
            roc_auc=roc_auc,
            f1=f1,
            average_precision=avg_precision,
            confusion_matrix_figure=dp.figure_
        )


# @title print prediction helper
# print prediciton

def print_prediction(ds: tf.data.Dataset,
                     model: tf.keras.Model,
                     display_labels: list[str],
                     col_size: int = 5) -> list[plt.Figure]:

    pages = []

    for (X, y) in ds.as_numpy_iterator():
        row_size = tf.math.ceil(tf.shape(y)[0] / col_size)
        row_size = int(row_size)
        fig = plt.figure(figsize=(col_size * 6, row_size * 6))
        pred_scores = model(X)
        preds = tf.argmax(pred_scores, -1)
        scores = tf.reduce_max(pred_scores, -1)
        # scores = tf.gather(pred_scores, preds)
        for idx in enumerate(X):
            ax = fig.subplot(row_size, col_size, idx + 1)
            label = display_labels[int(y[idx])]
            pred_label = display_labels[int(preds[idx])]
            img = tf.cast(X, dtype=tf.int8)
            ax.imshow(img)
            ax.set_title(f'True: {label}\nPredicited:{
                         pred_label} - {scores[idx].numpy()}')
            ax.axis('off')
        pages.append(fig)

    return pages
