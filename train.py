import os
import pickle
from glob import glob
from random import sample

from absl import app, flags, logging
from absl.flags import FLAGS

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
from tensorflow.python.keras import Input

from kmeans import kmeans
from yolov3_tf2.models import (
    YoloV3,
    YoloV3Tiny,
    YoloLoss,
    yolo_anchors,
    yolo_anchor_masks,
    yolo_tiny_anchors,
    yolo_tiny_anchor_masks,
    Darknet53_Lite,
)
from yolov3_tf2.utils import freeze_all, draw_outputs
import yolov3_tf2.dataset as dataset


flags.DEFINE_string("dataset", "/data2/seals/tfrecords/1024/train", "path to dataset")
flags.DEFINE_float(
    "validation", 0.2, "path to validation dataset",
)
flags.DEFINE_boolean("tiny", False, "yolov3 or yolov3-tiny")
flags.DEFINE_string("weights", "./checkpoints/yolov3.tf", "path to weights file")
flags.DEFINE_string("classes", "data/data_files/seal.names", "path to classes file")
flags.DEFINE_enum(
    "mode",
    "eager_tf",
    ["fit", "eager_fit", "eager_tf"],
    "fit: model.fit, " "eager_fit: model.fit(run_eagerly=True), " "eager_tf: custom GradientTape",
)
flags.DEFINE_enum(
    "transfer",
    "none",
    ["none", "darknet", "no_output", "frozen", "fine_tune"],
    "none: Training from scratch, "
    "darknet: Transfer darknet, "
    "no_output: Transfer all but output, "
    "frozen: Transfer and freeze all, "
    "fine_tune: Transfer all and freeze darknet only",
)
flags.DEFINE_integer("size", 416, "image size")
flags.DEFINE_integer("epochs", 100, "number of epochs")
flags.DEFINE_integer("batch_size", 8, "batch size")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_integer(
    "weights_num_classes",
    None,
    "specify num class for `weights` file if different, "
    "useful in transfer learning with different number of classes",
)
flags.DEFINE_string("out_dir", "/home/md273/model_zoo/416/", "the model output")


def create_img_summary(records: tf.data.Dataset, file_writer, take: int = 10) -> None:
    with file_writer.as_default():
        for x, y in records.take(take):
            if x1 == 0 and x2 == 0:
                continue
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


def is_test(x, y):
    return x % 4 == 0


def is_train(x, y):
    return not is_test(x, y)


recover = lambda x, y: y

os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    num_classes = 0
    with open(FLAGS.classes, "r") as class_file:
        num_classes = len([class_name.strip() for class_name in class_file.readlines()])
    if FLAGS.tiny:
        # TODO: replace num_classes
        model = YoloV3Tiny(FLAGS.size, training=True, classes=num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        # backbone = Darknet(inputs=Input([1024, 1024, 3], batch_size=4, name="input"))
        # print(backbone.summary())
        model = YoloV3(FLAGS.size, training=True, classes=num_classes, lite=True)
        print(model.summary())
        # TODO plugin flag
        all_records = read_csv("/data2/seals/tfrecords/416/train/records.csv")
        all_records["x_pixel"] = all_records["x_pixel"].apply(lambda x: x / FLAGS.size)
        all_records["y_pixel"] = all_records["y_pixel"].apply(lambda y: y / FLAGS.size)
        logging.info("Generating anchors")
        anchors = kmeans(all_records[["x_pixel", "y_pixel"]].to_numpy(), 9)
        anchor_masks = yolo_anchor_masks

    all_files = glob(os.path.join(FLAGS.dataset, "*.tfrecord"))
    all_files = all_files[:20]
    train_files, test_files = train_test_split(
        all_files, train_size=1 - FLAGS.validation, random_state=42,
    )

    train_dataset = dataset.load_tfrecord_dataset(train_files, FLAGS.classes, FLAGS.size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(
        lambda x, y: (
            dataset.transform_images(x, FLAGS.size),
            dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size),
        )
    )
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # val_dataset = dataset.load_fake_dataset()
    val_dataset = dataset.load_tfrecord_dataset(test_files, FLAGS.classes, FLAGS.size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(
        lambda x, y: (
            dataset.transform_images(x, FLAGS.size),
            dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size),
        )
    )
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # Configure the model for transfer learning
    if FLAGS.transfer == "none":
        pass  # Nothing to do
    elif FLAGS.transfer in ["darknet", "no_output"]:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes,
            )
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or num_classes,
            )
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == "darknet":
            model.get_layer("yolo_darknet").set_weights(
                model_pretrained.get_layer("yolo_darknet").get_weights()
            )
            freeze_all(model.get_layer("yolo_darknet"))

        elif FLAGS.transfer == "no_output":
            for l in model.layers:
                if not l.name.startswith("yolo_output"):
                    l.set_weights(model_pretrained.get_layer(l.name).get_weights())
                    freeze_all(l)

    else:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == "fine_tune":
            # freeze darknet and fine tune other layers
            darknet = model.get_layer("yolo_darknet")
            freeze_all(darknet)
        elif FLAGS.transfer == "frozen":
            # freeze everything
            freeze_all(model)
    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=num_classes) for mask in anchor_masks]

    if FLAGS.mode == "eager_tf":
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean("val_loss", dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                logging.info(
                    "{}_train_{}, {}, {}".format(
                        epoch,
                        batch,
                        total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss)),
                    )
                )
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

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
            model.save_weights("checkpoints/yolov3_train_{}.tf".format(epoch))
    else:
        model.compile(optimizer=optimizer, loss=loss, run_eagerly=(FLAGS.mode == "eager_fit"))

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint(
                os.path.join(FLAGS.out_dir, "checkpoints/yolov3_train_{epoch}.tf"),
                verbose=1,
                save_weights_only=True,
            ),
            TensorBoard(os.path.join(FLAGS.out_dir, "logs")),
        ]

        history = model.fit(
            train_dataset, epochs=FLAGS.epochs, callbacks=callbacks, validation_data=val_dataset,
        )
        logging.info(history)
        pickle.dump(history, open(os.path.join(FLAGS.out_dir, "history.pickle"), "wb"))
        model.save(os.path.join(FLAGS.out_dir, "model"))


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
