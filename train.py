import os
import pickle
import time
from functools import partial
from glob import glob
from random import sample

from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)

from kmeans import generate_anchors
from yolo.callbacks import GPUReport, TimeHistory
from yolo.config import anchor_size, masks, backbones, heads, down_sampling_factor
from yolo.models import yolo_loss
from yolo.utils import freeze_all, draw_outputs, get_flops
import yolo.dataset as dataset
from memory_profiler import profile


flags.DEFINE_string("dataset", "/data2/seals/tfrecords/1024/train", "path to dataset")
flags.DEFINE_float(
    "validation", 0.2, "the size of the validation set",
)
flags.DEFINE_boolean("tiny", False, "yolov3 or yolov3-tiny")
flags.DEFINE_string("classes", "data/data_files/seal.names", "path to classes file")
flags.DEFINE_enum(
    "mode",
    "fit",
    ["fit", "eager_fit", "eager_tf"],
    "fit: model.fit, " "eager_fit: model.fit(run_eagerly=True), " "eager_tf: custom GradientTape",
)
flags.DEFINE_integer("size", 416, "image size")
flags.DEFINE_integer("epochs", 20, "number of epochs")
flags.DEFINE_integer("batch_size", 8, "batch size")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")

flags.DEFINE_string("out_dir", "/home/md273/model_zoo/416/", "the model output")
flags.DEFINE_string("record_csv", "", "the all records csv frame to calculate the anchor boxes")
flags.DEFINE_enum(
    "backbone",
    "original",
    ["original", "dense"],
    "original: the original YOLOv3 with darknet 53, "
    "dense: Customised yolo v3 with a downsampling factor of 16, ",
)

flags.DEFINE_enum(
    "head",
    "yolo3",
    ["yolo3", "yolo3_dense_2", "yolo3_dense_1"],
    "original: the original YOLOv3 with darknet 53,"
    "yolo3_dense_2: Customised yolo v3 with a downsampling factor of 16, and two scales ",
)

flags.DEFINE_integer(
    "sample_train", -1, "The number of datapoints to take for smaller experiment runs", -1
)

flags.DEFINE_boolean(
    "disable_prefetch", False, "A flag that disables prefetch for memory estimation"
)

flags.DEFINE_boolean(
    "mem_eval",
    False,
    "A flag that disables gpu usage and activates cpu usage for memory evaluation",
)


def create_img_summary(records: tf.data.Dataset, file_writer, take: int = 10) -> None:
    with file_writer.as_default():
        for x, y in records.take(take):
            boxes = []
            scores = []
            classes = []
            for x1, y1, x2, y2, label in y:
                if x1 == 0 and x2 == 0:
                    continue

                boxes.append((x1, y1, x2, y2))
                scores.append(1)
                classes.append(label)
            nums = [len(boxes)]
            boxes = [boxes]
            scores = [scores]
            classes = [classes]
            img = cv2.cvtColor(x.numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), classes)
            rgb_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            tf.summary.image("Training data", rgb_tensor, step=0)
            file_writer.flush()


os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"


# @tf.function
def validation(images, labels, loss: list, model, pred_loss: list, total_loss: list):
    """
    A tf function to calculate the validation error 
    :param images: the validation images
    :param labels: the labels
    :param loss: the loss function
    :param model: the model to validate
    :param pred_loss: the training prediction loss
    :param total_loss: the training loss
    :return: 
    """
    outputs = model(images)
    regularization_loss = tf.reduce_sum(model.losses)
    pred_loss = []
    if len(loss) == 1:
        outputs = (outputs, None)
    for output, label, loss_fn in zip(outputs, labels, loss):
        pred_loss.append(loss_fn(label, output))
    total_loss = tf.reduce_sum(pred_loss) + regularization_loss
    return pred_loss, total_loss


# @tf.function
def train_batch(
    images, labels, loss: list, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer
) -> tuple:
    """
    A tf function to train the model on a batch.
    :param images: the images
    :param labels: the labels in the batch
    :param loss: the loss functions
    :param model: the model
    :param optimizer: the optimization technique
    :return: 
    """
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        # outputs = model(images, training=True, _checkpoint=True)
        outputs = model(images)
        regularization_loss = tf.reduce_sum(model.losses)
        pred_loss = []
        if len(loss) == 1:
            outputs = (outputs, None)
        for output, label, loss_fn in zip(outputs, labels, loss):
            pred_loss.append(loss_fn(label, output))
        total_loss = tf.reduce_sum(pred_loss) + regularization_loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return pred_loss, total_loss


@profile
def train_eagerly(model, train_dataset, val_dataset, loss, optimizer) -> None:
    """
    A function to train the yolo model using eager mode.
    :param model: the model
    :param train_dataset: the training dataset
    :param val_dataset: the validation dataset
    :param loss: the loss function
    :param optimizer: the optimizer
    :return: None
    """
    avg_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean("val_loss", dtype=tf.float32)
    for epoch in range(1, FLAGS.epochs + 1):
        n_samples = 0
        t1 = time.time()

        for batch, (images, labels) in enumerate(train_dataset):
            n_samples += 1
            pred_loss, total_loss = train_batch(images, labels, loss, model, optimizer)
            logging.info(
                "{}_train_{}, {}, {}".format(
                    epoch,
                    batch,
                    total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss)),
                )
            )
            avg_loss.update_state(total_loss)

        t2 = time.time()
        logging.info(f"Trained on {n_samples} in {t2 - t1}")

        for batch, (images, labels) in enumerate(val_dataset):
            pred_loss, total_loss = validation(images, labels, loss, model, pred_loss, total_loss)

            logging.info(
                "{}_val_{}, {}, {}".format(
                    epoch,
                    batch,
                    total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss)),
                )
            )
            avg_val_loss.update_state(total_loss)

        logging.info(
            "{}, train: {}, val: {}".format(
                epoch, avg_loss.result().numpy(), avg_val_loss.result().numpy()
            )
        )

        avg_loss.reset_states()
        avg_val_loss.reset_states()
        model.save_weights(os.path.join(FLAGS.out_dir, f"checkpoints/yolov3_train_{epoch}.tf"))


def memory_usage_psutil():
    """
    A util to get the cpu memory utilisation
    :return: the cpu mem usage
    """
    import psutil

    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem


def main(_argv):
    if FLAGS.mem_eval:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    log_dir = os.path.join(FLAGS.out_dir, "logs")
    writer = tf.summary.create_file_writer(log_dir)
    with open(FLAGS.classes, "r") as class_file:
        num_classes = len([class_name.strip() for class_name in class_file.readlines()])
    all_records = read_csv(FLAGS.record_csv)
    logging.info("Generating anchors")
    anchors, _, _ = generate_anchors(all_records, anchor_size[FLAGS.head])
    anchor_masks = masks[FLAGS.head]
    np.save(os.path.join(FLAGS.out_dir, "anchors.npy"), anchors)
    logging.info("Building model config")
    backbone = backbones[FLAGS.backbone]
    head = heads[FLAGS.head]
    model = head(
        FLAGS.size,
        training=True,
        classes=num_classes,
        anchors=anchors,
        masks=anchor_masks,
        backbone=backbone,
    )
    print(model.summary())

    all_files = glob(os.path.join(FLAGS.dataset, "*.tfrecord"))
    train_files, test_files = train_test_split(
        all_files, train_size=1 - FLAGS.validation, random_state=42,
    )
    train_dataset = dataset.load_tfrecord_dataset(train_files, FLAGS.classes, FLAGS.size)
    train_dataset = train_dataset.shuffle(buffer_size=512)

    train_dataset = train_dataset.batch(FLAGS.batch_size)

    for _, y in train_dataset.take(1):
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size, downsampling_factor=16)

    downsample_by = down_sampling_factor.get(FLAGS.backbone)

    train_dataset = train_dataset.map(
        lambda x, y: (
            dataset.transform_images(x, FLAGS.size),
            dataset.transform_targets(
                y, anchors, anchor_masks, FLAGS.size, downsampling_factor=downsample_by
            ),
        )
    )
    if FLAGS.sample_train > -1:
        train_dataset = train_dataset.take(FLAGS.sample_train)
    # train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if not FLAGS.disable_prefetch:
        train_dataset = train_dataset.prefetch(buffer_size=1)

    val_dataset = dataset.load_tfrecord_dataset(test_files, FLAGS.classes, FLAGS.size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(
        lambda x, y: (
            dataset.transform_images(x, FLAGS.size),
            dataset.transform_targets(
                y, anchors, anchor_masks, FLAGS.size, downsampling_factor=downsample_by
            ),
        )
    )
    if not FLAGS.disable_prefetch:
        val_dataset = val_dataset.prefetch(buffer_size=1)

    if FLAGS.sample_train > -1:
        val_dataset = val_dataset.take(FLAGS.sample_train)

    print(model.summary())
    optimizer = keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [yolo_loss(anchors[mask], classes=num_classes) for mask in anchor_masks]

    if FLAGS.mode == "eager_tf":
        # eager implementation
        train_eagerly(model, train_dataset, val_dataset, loss, optimizer)
    else:
        model.compile(optimizer=optimizer, loss=loss, run_eagerly=(FLAGS.mode == "eager_fit"))

        callbacks = [
            GPUReport(os.path.join(FLAGS.out_dir, "memory_usage.csv")),
            TimeHistory(os.path.join(FLAGS.out_dir, "epoch_time.csv")),
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint(
                os.path.join(FLAGS.out_dir, "checkpoints/yolov3_train_{epoch}.tf"),
                verbose=1,
                save_weights_only=True,
            ),
            TensorBoard(os.path.join(FLAGS.out_dir, "logs"), profile_batch="500,510"),
        ]

        history = model.fit(
            train_dataset, epochs=FLAGS.epochs, callbacks=callbacks, validation_data=val_dataset,
        )
        logging.info(history)
        pickle.dump(history, open(os.path.join(FLAGS.out_dir, "history.pickle"), "wb"))
        # model.save(os.path.join(FLAGS.out_dir, "model"))


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
