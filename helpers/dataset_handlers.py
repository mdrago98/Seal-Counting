import tensorflow as tf
import pandas as pd
import numpy as np
import hashlib
import os
from pathlib import Path
from helpers.utils import default_logger


# def get_feature_map():
#     """
#     Get tf.train.Example features.
#
#     Returns:
#         features
#     """
#     features = {
#         'image_width': tf.io.FixedLenFeature([], tf.int64),
#         'image_height': tf.io.FixedLenFeature([], tf.int64),
#         'image_path': tf.io.FixedLenFeature([], tf.string),
#         'image_file': tf.io.FixedLenFeature([], tf.string),
#         'image_key': tf.io.FixedLenFeature([], tf.string),
#         'image_data': tf.io.FixedLenFeature([], tf.string),
#         'image_format': tf.io.FixedLenFeature([], tf.string),
#         'x_min': tf.io.VarLenFeature(tf.float32),
#         'y_min': tf.io.VarLenFeature(tf.float32),
#         'x_max': tf.io.VarLenFeature(tf.float32),
#         'y_max': tf.io.VarLenFeature(tf.float32),
#         'object_name': tf.io.VarLenFeature(tf.string),
#         'object_id': tf.io.VarLenFeature(tf.int64),
#     }
#     return features


def get_feature_map():
    """
    Get tf.train.Example features.

    Returns:
        features
    """
    features = {
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/source_id": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
    }
    return features


def create_example(separate_data, key, image_data):
    """
    Create tf.train.Example object.
    Args:
        separate_data: numpy tensor of 1 image data.
        key: output of hashlib.sha256()
        image_data: raw image data.

    Returns:
        tf.train.Example object.
    """
    [
        image,
        object_name,
        image_width,
        image_height,
        x_min,
        y_min,
        x_max,
        y_max,
        _,
        _,
        object_id,
    ] = separate_data
    image_file_name = os.path.split(image[0])[-1]
    image_format = image_file_name.split(".")[-1]
    features = {
        "image_height": tf.train.Feature(int64_list=tf.train.Int64List(value=[image_height[0]])),
        "image_width": tf.train.Feature(int64_list=tf.train.Int64List(value=[image_width[0]])),
        "image_path": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image[0].encode("utf-8")])
        ),
        "image_file": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_file_name.encode("utf8")])
        ),
        "image_key": tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode("utf8")])),
        "image_data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
        "image_format": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_format.encode("utf8")])
        ),
        "x_min": tf.train.Feature(float_list=tf.train.FloatList(value=x_min)),
        "y_min": tf.train.Feature(float_list=tf.train.FloatList(value=y_min)),
        "x_max": tf.train.Feature(float_list=tf.train.FloatList(value=x_max)),
        "y_max": tf.train.Feature(float_list=tf.train.FloatList(value=y_max)),
        "object_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=object_name)),
        "object_id": tf.train.Feature(int64_list=tf.train.Int64List(value=object_id)),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def read_example(
    example, feature_map, class_table, max_boxes, new_size=None, get_features=False,
):
    """
    Read single a single example from a TFRecord file.
    Args:
        example: nd tensor.
        feature_map: A dictionary of feature names mapped to tf.io objects.
        class_table: StaticHashTable object.
        max_boxes: Maximum number of boxes per image
        new_size: w, h new image size
        get_features: If True, features will be returned.

    Returns:
        x_train, y_train
    """
    # TODO: remove x pixel
    features = tf.io.parse_single_example(example, feature_map)
    x_train = tf.image.decode_png(features["image/encoded"], channels=3)
    if new_size:
        x_train = tf.image.resize(x_train, new_size)
    object_name = tf.sparse.to_dense(features["image/object/class/label"])
    label = tf.cast(object_name, tf.float32)
    y_train = tf.stack(
        [
            tf.sparse.to_dense(features[feature])
            for feature in [
                "image/object/bbox/xmin",
                "image/object/bbox/ymin",
                "image/object/bbox/xmax",
                "image/object/bbox/ymax",
            ]
        ]
        + [label],
        1,
    )
    padding = [[0, max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, padding)
    if get_features:
        return x_train, y_train, features
    return x_train, y_train


def write_tf_record(output_path, groups, data, trainer=None):
    """
    Write data to TFRecord.
    Args:
        output_path: Full path to save.
        groups: pandas GroupBy object.
        data: pandas DataFrame
        trainer: main.Trainer object.

    Returns:
        None
    """
    print(f"Processing {os.path.split(output_path)[-1]}")
    if trainer:
        if "train" in output_path:
            trainer.train_tf_record = output_path
        if "test" in output_path:
            trainer.valid_tf_record = output_path
    with tf.io.TFRecordWriter(output_path) as r_writer:
        for current_image, (image_path, objects) in enumerate(groups, 1):
            print(
                f"\rBuilding example: {current_image}/{len(groups)} ... "
                f"{os.path.split(image_path)[-1]} "
                f"{round(100 * (current_image / len(groups)))}% completed",
                end="",
            )
            separate_data = pd.DataFrame(objects, columns=data.columns).T.to_numpy()
            (image_width, image_height, x_min, y_min, x_max, y_max,) = separate_data[2:8]
            x_min /= image_width
            x_max /= image_width
            y_min /= image_height
            y_max /= image_height
            try:
                image_data = open(image_path, "rb").read()
                key = hashlib.sha256(image_data).hexdigest()
                training_example = create_example(separate_data, key, image_data)
                r_writer.write(training_example.SerializeToString())
            except Exception as e:
                default_logger.error(e)
    print()


def save_tfr(data, output_folder, dataset_name, test_size=None, trainer=None):
    """
    Transform and save dataset into TFRecord format.
    Args:
        data: pandas DataFrame with adjusted labels.
        output_folder: Path to folder where TFRecord(s) will be saved.
        dataset_name: str name of the dataset.
        test_size: relative test subset size.
        trainer: main.Trainer object

    Returns:
        None
    """
    data["Object Name"] = data["Object Name"].apply(lambda x: x.encode("utf-8"))
    data["Object ID"] = data["Object ID"].astype(int)
    data[data.dtypes[data.dtypes == "int64"].index] = data[
        data.dtypes[data.dtypes == "int64"].index
    ].apply(abs)
    data.to_csv(os.path.join("../yolo", "Data", "TFRecords", "full_data.csv"), index=False)
    groups = np.array(data.groupby("Image Path"))
    np.random.shuffle(groups)
    if test_size:
        assert 0 < test_size < 1, f"test_size must be 0 < test_size < 1 and {test_size} is given"
        separation_index = int((1 - test_size) * len(groups))
        training_set = groups[:separation_index]
        test_set = groups[separation_index:]
        training_frame = pd.concat([item[1] for item in training_set])
        test_frame = pd.concat([item[1] for item in test_set])
        training_frame.to_csv(
            os.path.join("../yolo", "Data", "TFRecords", "training_data.csv"), index=False,
        )
        test_frame.to_csv(
            os.path.join("../yolo", "Data", "TFRecords", "test_data.csv"), index=False,
        )
        training_path = str(
            Path(os.path.join(output_folder, f"{dataset_name}_train.tfrecord")).absolute().resolve()
        )
        test_path = str(
            Path(os.path.join(output_folder, f"{dataset_name}_test.tfrecord")).absolute().resolve()
        )
        write_tf_record(training_path, training_set, data, trainer)
        default_logger.info(f"Saved training TFRecord: {training_path}")
        write_tf_record(test_path, test_set, data, trainer)
        default_logger.info(f"Saved validation TFRecord: {test_path}")
        return
    tf_record_path = os.path.join(output_folder, f"{dataset_name}.tfrecord")
    write_tf_record(tf_record_path, groups, data, trainer)
    default_logger.info(f"Saved TFRecord {tf_record_path}")


def read_tfr(
    tf_record_file,
    classes_file,
    feature_map,
    max_boxes,
    classes_delimiter="\n",
    new_size=None,
    get_features=False,
):
    """
    Read and load dataset from TFRecord file.
    Args:
        tf_record_file: Path to TFRecord file.
        classes_file: file containing classes.
        feature_map: A dictionary of feature names mapped to tf.io objects.
        max_boxes: Maximum number of boxes per image.
        classes_delimiter: delimiter in classes_file.
        new_size: w, h new image size
        get_features: If True, features will be returned.

    Returns:
        MapDataset object.
    """
    tf_record_file = str(Path(tf_record_file).absolute().resolve())
    text_init = tf.lookup.TextFileInitializer(
        classes_file, tf.string, 0, tf.int64, -1, delimiter=classes_delimiter
    )
    class_table = tf.lookup.StaticHashTable(text_init, -1)
    files = tf.data.Dataset.list_files(tf_record_file)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    default_logger.info(f"Read TFRecord: {tf_record_file}")
    return dataset.map(
        lambda x: read_example(x, feature_map, class_table, max_boxes, new_size, get_features)
    )
