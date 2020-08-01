import os
from glob import glob
from pathlib import Path

# from mean_average_precision import MeanAveragePrecision
from random import sample

import cv2
from absl import app, flags, logging
from mean_average_precision.detection_map import DetectionMAP
from pandas import DataFrame, concat
from tqdm import tqdm

from yolov3_tf2 import dataset
from yolov3_tf2.models import YoloV3
import tensorflow as tf
import pickle
from time import time
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

# from mean_average_precision import DetectionMAP

FLAGS = flags.FLAGS

flags.DEFINE_string("classes", "./data/data_files/seal.names", "path to classes file")
flags.DEFINE_string(
    "weights",
    "/home/md273/model_zoo/416_eager/checkpoints/yolov3_train_7.tf",
    "path to weights file",
)
flags.DEFINE_boolean("tiny", False, "yolov3 or yolov3-tiny")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_string(
    "tfrecord", "/data2/seals/tfrecords/416/train", "tfrecord instead of image",
)
flags.DEFINE_string("output", "/home/md273/model_zoo/416_eager/eval", "path to output image")
# flags.DEFINE_integer("num_classes", 80, "number of classes in the model")

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


def calculate_metrics(eval_data, model, mAP) -> tuple:
    false_negative = 0
    inference_times = []
    detections = DataFrame()

    for index, (image, labels) in enumerate(eval_data):
        t1 = time()
        boxes, scores, classes, nums = model(image)
        t2 = time()
        inference_times += [t2 - t1]
        # get first 4 cols from tensor and compute iou
        pred, pred_cls, pred_conf, gt_bb, gt_cls = [], [], [], [], []
        for i, truth in enumerate(labels[:, :5]):
            if (
                tf.reduce_sum(truth).numpy() != 0
                or tf.reduce_sum(boxes[0][i]) != 0
                or tf.reduce_sum(scores[0][i]) != 0
            ):
                pred += [boxes[0][i].numpy()]
                pred_cls += [classes[0][i].numpy()]
                pred_conf += [scores[0][i].numpy()]
                gt_bb += [truth[:4].numpy()]
                gt_cls += [truth[4:5].numpy()[0]]

                iou = bb_intersection_over_union(boxes[0][i], truth).numpy()
                detection = {
                    "image": [index],
                    "confidence": [scores[0][i].numpy()],
                }
                if scores[0][i] >= 0.5 and iou >= 0.5:
                    detection["TP"] = [1]
                    detection["FP"] = [0]
                elif iou < 0.5:
                    detection["TP"] = [0]
                    detection["FP"] = [1]
                elif tf.reduce_sum(truth).numpy() != 0:
                    false_negative += 1
                detections = detections.append(DataFrame(detection))
        mAP.evaluate(
            *[
                np.array(pred),
                np.array(pred_cls),
                np.array(pred_conf),
                np.array(gt_bb),
                np.array(gt_cls),
            ]
        )

    detections = detections.sort_values("confidence", ascending=False)
    detections["Acc TP"] = detections["TP"].cumsum()
    detections["Acc FP"] = detections["FP"].cumsum()
    detections["Precision"] = detections["Acc TP"] / (detections["Acc TP"] + detections["Acc FP"])
    detections["Recall"] = detections["Acc TP"] / (detections["Acc TP"] + false_negative)

    return detections, mAP


def main(_argv):
    # TODO: save anchors during training and load
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    # yolo = YoloV3(classes=len(class_names))
    yolo = YoloV3(classes=80)
    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info("Weights loaded")

    logging.info("Classes loaded")
    all_files = sample(glob(os.path.join(FLAGS.tfrecord, "*.tfrecord")), 10)
    mAP = DetectionMAP(len(class_names))
    all_results = DataFrame()
    fig_dir = Path(os.path.join(os.path.basename(FLAGS.output)), "figures")
    fig_dir.mkdir(exist_ok=True, parents=True)
    csv_out = Path(os.path.join(os.path.basename(FLAGS.output)), "csv")
    csv_out.mkdir(exist_ok=True, parents=True)
    for file in tqdm(all_files):
        eval_data = dataset.load_tfrecord_dataset([file], FLAGS.classes, FLAGS.size)
        eval_data = eval_data.map(lambda x, y: (tf.expand_dims(x, 0), y,))
        eval_data = eval_data.map(lambda x, y: (dataset.transform_images(x, FLAGS.size), y,))

        results, mAP = calculate_metrics(eval_data, yolo, mAP)
        results.to_csv(os.path.join(FLAGS.output, f"{os.path.basename(file)}.csv"))
        all_results = concat([all_results, results])
        mAP.plot()
        plt.show()
        ax = sns.lineplot(x="Recall", y="Precision", data=results)
        fig = ax.get_figure()
        fig.savefig(os.path.join(fig_dir, f"pr_{os.path.basename(file)}.png"))
    all_results.dropna().to_csv("all_results")
    ax = sns.lineplot(x="Recall", y="Precision", data=all_results)
    fig = ax.get_figure()
    fig.savefig(os.path.join(fig_dir, f"pr_{all}.png"))


if __name__ == "__main__":
    app.run(main)
