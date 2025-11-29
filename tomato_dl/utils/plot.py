import typing as tp
import jaxtyping as ttf
import numpy as np
import matplotlib.pyplot as plt


type Array = tp.Union[list[float], ttf.Float[np.ndarray, "..."]]


class GraphHistory(tp.TypedDict):
    accuracy: Array
    loss: Array
    val_accuracy: Array
    val_loss: Array


def plot_graph(
    history: GraphHistory, fig: tp.Optional[plt.Figure] = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    # Create figure
    if fig is None:
        fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Training & Validation Metrics Visualization",
                 fontweight="bold", fontsize="x-large")
    # Plot both loss and accuracy
    (ax1, ax2) = fig.subplots(nrows=1, ncols=2)
    # Plot the training graph
    ax1.plot(history['accuracy'])
    ax1.plot(history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    # Plot the loss graph
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    return (fig, ax1, ax2)
