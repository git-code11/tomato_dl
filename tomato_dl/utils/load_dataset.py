from functools import reduce
from operator import itemgetter
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_dataset(dataset_dir: os.PathLike,
                 *,
                 image_size: tuple[int, int] = (256, 256),
                 seed: int | None = None,
                 batch_size: int | None = 32,
                 split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15)) \
        -> list[tf.data.Dataset]:

    if len(split_ratio) != 3:
        raise Exception(
            "Require len(split_ratio) == 2 (train, valid, test)")

    if not (all([x > 0 for x in split_ratio]) and sum(split_ratio) == 1):
        raise Exception(
            "Check split paramaters: Enure non-zero and sum to be 1")

    classes = tf.io.gfile.listdir(dataset_dir)
    classes.sort()

    def glob_dir(_cls):
        _dir = tf.io.gfile.join(dataset_dir, _cls)
        _fnames = tf.io.gfile.glob(tf.io.gfile.join(_dir, '*.jpg'))
        _fnames = map(lambda x: tf.io.gfile.join(_dir, x), _fnames)
        return list(_fnames)

    fnames = {x: glob_dir(x) for x in classes}

    print("Found ", end="")
    print(*[f"{x} = {len(fnames[x])}" for x in classes], sep=", ")

    labels = {
        x: tf.broadcast_to(
            idx, [len(fnames[x]), 1])
        for idx, x in enumerate(classes)
    }
   
    # Flatten
    fnames, labels = list(
        reduce(
            lambda y, x: [y[0]+fnames[x], tf.concat([y[1],labels[x]], axis=0)],
            classes[1:],
            [fnames[classes[0]], labels[classes[0]]]
        )
    )

    labels = labels.numpy().tolist()

    def qget_labels(idxs): return list(itemgetter(*idxs)(labels))

    def qget_fnames(idxs): return list(itemgetter(*idxs)(fnames))

    total_count = len(labels)
    data_count = [int(total_count*x) for x in split_ratio]

    print(f"Total Size = {total_count}")

    train_idx, test_idx = train_test_split(
       tf.range(total_count).numpy().tolist(),
        test_size=data_count[2],
        random_state=seed,
        stratify=labels
    )
  
    train_idx, valid_idx = train_test_split(
        train_idx,
        test_size=data_count[1],
        random_state=seed,
        stratify=qget_labels(train_idx)
    )

    def load_image(img_path: str):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        return img

    def make_dataset(idxs: list[int]):
        _fnames = qget_fnames(idxs)
        _labels = qget_labels(idxs)
        
        X = tf.data.Dataset.from_tensor_slices(_fnames) \
            .map(load_image)
        y = tf.data.Dataset.from_tensor_slices(_labels)
        out = tf.data.Dataset.zip(X, y)

        if batch_size:
            out = out.batch(batch_size)

        out.class_names = classes
        out.file_paths = _fnames
        return out

    dss = [make_dataset(idx)
           for idx in [train_idx, valid_idx, test_idx]]

    return dss
