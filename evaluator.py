import os

from absl import app, flags, logging

from yolov3_tf2 import dataset
from yolov3_tf2.models import YoloV3
import tensorflow as tf
import pickle
from time import time

FLAGS = flags.FLAGS

flags.DEFINE_string("classes", "./data/data_files/seal.names", "path to classes file")
flags.DEFINE_string(
    "weights",
    "/home/md273/model_zoo/416_eager/checkpoints/yolov3_train_7.tf",
    "path to weights file",
)
flags.DEFINE_boolean("tiny", False, "yolov3 or yolov3-tiny")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_string("image", "./data/girl.png", "path to input image")
flags.DEFINE_string(
    "tfrecord", "/data2/seals/tfrecords/416_all/train.tfrecord", "tfrecord instead of image"
)
flags.DEFINE_string(
    "output", "/home/md273/model_zoo/416_eager/results_test.pickle", "path to output image"
)
flags.DEFINE_integer("num_classes", 80, "number of classes in the model")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# def iou(y_pred, y_true) -> tf.Tensor:
#     I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
#     U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
#     return tf.reduce_mean(I / U)


def bb_intersection_over_union(boxA: tf.Tensor, boxB: tf.Tensor) -> tf.Tensor:
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def calc_mean(iterable: list) -> float:
    return sum(iterable) / len(iterable)


def omit_zero_vals(x: tf.Tensor) -> tf.Tensor:
    """
    A helper function to remove zero valued vectors from a tensor
    :param x: the tensor to prune
    :return: the pruned tensor
    """
    intermediate_tensor = tf.reduce_sum(tf.abs(x), 1)
    zero_vector = tf.zeros(shape=(1, 1), dtype=tf.float32)
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    return tf.boolean_mask(x, bool_mask)


def calculate_metrics(eval_data, model, sample: int = 100) -> dict:
    eval_scores = {"inference_times": [], "ious": []}
    ious = []
    for index, (image, labels) in enumerate(eval_data):
        t1 = time()
        boxes, scores, classes, nums = model(image)
        t2 = time()
        eval_scores["inference_times"].append(t2 - t1)
        # get first 4 cols from tensor and compute iou
        ious += [
            bb_intersection_over_union(prediction, truth).numpy()
            for prediction, truth in zip(boxes[0], labels[:, :4])
            if tf.reduce_sum(truth)[0] != 0
        ]
    return eval_scores


def main(_argv):
    yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info("Weights loaded")
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info("Classes loaded")
    eval_data = dataset.load_tfrecord_dataset(FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
    eval_data = eval_data.map(lambda x, y: (tf.expand_dims(x, 0), y,))
    eval_data = eval_data.map(lambda x, y: (dataset.transform_images(x, FLAGS.size), y,))
    results = calculate_metrics(eval_data, yolo)
    with open(FLAGS.output, "wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    app.run(main)
