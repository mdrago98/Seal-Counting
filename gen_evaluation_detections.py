import os
from pathlib import Path

# from mean_average_precision import MeanAveragePrecision

import cv2
from absl import app, flags, logging
from pandas import DataFrame, read_csv
from mean_average_precision.detection_map import DetectionMAP

from data.generate_tfrecords import (
    extrapolate_patches,
    split_frame,
    split,
)
from yolo import dataset
from yolo.config import backbones, heads, masks
import tensorflow as tf
from time import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from yolo.utils import draw_outputs, COLORS

FLAGS = flags.FLAGS

flags.DEFINE_string("classes", "./data/data_files/seal.names", "path to classes file")

flags.DEFINE_integer("size", 416, "the patch size")
flags.DEFINE_integer("model_size", 416, "the model size")
flags.DEFINE_string(
    "weights",
    "/home/md273/model_zoo/416_darknet_1/checkpoints/yolov3_train_15.tf",
    "path to weights file",
)
flags.DEFINE_string(
    "output", "/home/md273/model_zoo/416_darknet_1/eval", "path to output the results"
)
flags.DEFINE_string(
    "anchor_path", "/home/md273/model_zoo/416_darknet_1/anchors.npy", "path to the anchor file",
)

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
    "dense: Customised yolo v3 with a downsampling factor of 16, and two scales ",
)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def bb_intersection_over_union(box_a: tf.Tensor, box_b: tf.Tensor) -> float:
    """
    Calculates the bounding box IOU over the the ground truth.
    :param box_a: the first bounding box
    :param box_b: the second bounding box
    :return: the iou
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    boxBArea = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def calc_mean(iterable: list) -> float:
    """
    Calculates the mean of an iterable
    :param iterable: the iterable of numeric types
    :return: the mean
    """
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


def zero_filter(x):
    """
    A function to remove zero vectors from a tensor
    :param x: the tensor to prune
    :return: the pruned tensor
    """
    return tf.boolean_mask(x, tf.cast(x, dtype=tf.bool))


def calculate_metrics(eval_data, model) -> tuple:
    mAP = DetectionMAP(1)
    false_negative = 0
    inference_times = []
    detections = DataFrame()

    for index, (image, labels) in enumerate(eval_data):
        t1 = time()
        boxes, scores, classes, nums = model.predict(image)
        t2 = time()
        inference_times += [t2 - t1]
        labels = zero_filter(labels)
        # get first 4 cols from tensor and compute iou
        pred, pred_cls, pred_conf, gt_bb, gt_cls = [], [], [], [], []
        for i, truth in enumerate(labels[:, :5]):
            if (
                tf.reduce_sum(truth).numpy() != 0
                or tf.reduce_sum(boxes[0][i]) != 0
                or tf.reduce_sum(scores[0][i]) != 0
            ):
                pred += [boxes[0][i].numpy()]
                pred_cls += [1]
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
        if len(pred) != 0:
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


def write_gt(ground_truth: DataFrame, output: str):
    ground_truth["layer_name"] = ground_truth["layer_name"].apply(lambda _: 0)
    ground_truth[["layer_name", "xmin", "ymin", "xmax", "ymax"]].to_csv(
        output, sep="\t", index=False, header=False
    )


def normalise(ground_truth, model_size):
    """
    Apply fractional representation for the bounding boxes to get the normalised size
    :param ground_truth: the ground truth files
    :param model_size: the model size
    :return: 
    """
    ground_truth["xmin"] = ground_truth["xmin"] / ground_truth["image_width"] * model_size
    ground_truth["xmax"] = ground_truth["xmax"] / ground_truth["image_width"] * model_size
    ground_truth["ymin"] = ground_truth["ymin"] / ground_truth["image_height"] * model_size
    ground_truth["ymax"] = ground_truth["ymax"] / ground_truth["image_height"] * model_size
    return ground_truth


def write_detections(boxes, conf, output):
    """
    A fucntion to write the detections to file.
    :param boxes: the boxes
    :param conf: the confidence
    :param output: the name of the file
    :return: 
    """
    with open(output, "w") as file:
        lines = [
            f"0 {conf[i]} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n"
            for i, box in enumerate(boxes)
            if boxes.size != 0 and np.sum(box) != 0
        ]
        file.writelines(lines)


def predict_on_patches(image_name: str, interval_size: tuple, seal_locations: DataFrame, model):
    """
    A function that executes the model on extracted patches from an aerial image
    :param downsample: an int indicating the final downsampling size. 0 means no downsampling
    :param image_name: the image name
    :param interval_size: the patch size
    :param seal_locations: the seal locations ground truth
    :param model: the model to evaluate
    :return: None
    """
    inference_times = []
    patches = extrapolate_patches(image_name, seal_locations, interval_size, ignore_extrema=False)
    # TODO: remove
    # patches = [patch for patch in patches if patch.object.size != 0]
    for i, (image_bytes, transformed, patch_region) in enumerate(patches):
        image = tf.image.decode_png(image_bytes)
        image = tf.expand_dims(image, 0)
        image = dataset.transform_images(image, FLAGS.model_size)
        t1 = time()
        boxes, scores, classes, nums = model(image)
        t2 = time()
        inference_times += [t2 - t1]
        transformed = normalise(transformed, FLAGS.model_size)

        if len(transformed) > 0:
            output = os.path.join(FLAGS.output, "detections", f"{i}.png")
            Path(output).parent.mkdir(exist_ok=True, parents=True)
            plot(
                image,
                (boxes, scores, classes, nums),
                transformed,
                os.path.join(FLAGS.output, "detections", f"{i}.png"),
            )
        # filtered_boxes = zero_filter(boxes)
        # if filtered_boxes.shape.rank == 1:
        #     filtered_boxes = tf.expand_dims(filtered_boxes, 0)
        # filtered_conf = zero_filter(scores)
        # if filtered_conf.shape.rank == 1:
        #     filtered_conf = tf.expand_dims(filtered_conf, 0)
        # filtered_classes = np.zeros(filtered_boxes.shape[0], dtype=int)
        name = f"{os.path.splitext(image_name)[0]}_{i}.txt"
        Path(os.path.join(FLAGS.output, "ground-truth", name)).parent.mkdir(
            exist_ok=True, parents=True
        )
        write_gt(transformed, os.path.join(FLAGS.output, "ground-truth", name))
        Path(os.path.join(FLAGS.output, "detection_results", name)).parent.mkdir(
            exist_ok=True, parents=True
        )
        write_detections(
            (boxes[0] * FLAGS.model_size).numpy(),
            scores[0].numpy(),
            os.path.join(FLAGS.output, "detection_results", name),
        )


def plot(image, predictions, transformations, output):
    img = cv2.cvtColor(image[0].numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, predictions, "green")
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for index, row in transformations[["xmin", "ymin", "xmax", "ymax"]].iterrows():
        patch = patches.Rectangle(
            (row["xmin"], row["ymin"]),
            row["xmax"] - row["xmin"],
            row["ymax"] - row["ymin"],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(patch)
    plt.savefig(output, bbox_inches="tight", dpi=800)
    return img


def main(_argv):
    # TODO: save anchors during training and load
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    anchors = np.load(FLAGS.anchor_path)

    backbone = backbones[FLAGS.backbone]
    head = heads[FLAGS.head]
    anchor_masks = masks[FLAGS.head]
    model = head(
        FLAGS.model_size,
        classes=len(class_names),
        anchors=anchors,
        masks=anchor_masks,
        backbone=backbone,
        training=False,
    )
    model.summary()
    # model = yolo_3(classes=len(class_names), anchors=anchors, training=False, backbone=backbones[FLAGS.backbone])
    model.load_weights(FLAGS.weights).expect_partial()
    logging.info("Weights loaded")
    logging.info(model.count_params())
    locations = read_csv("/data2/seals/tfrecords/all.csv")
    locations["y_pixel"] = locations["image_height"] - locations["y_pixel"]
    _, test = split_frame(locations)
    logging.info("Locations loaded")
    grouped = split(test, "tiff_file")
    for filename, locations, _ in grouped[:5]:
        logging.info(f"extracting {filename}")
        predict_on_patches(filename, (FLAGS.size, FLAGS.size), locations, model)
    # predict_on_patches(grouped_train)

    # all_files = sample(glob(os.path.join(FLAGS.tfrecord, "*.tfrecord")), 1)

    # all_results = DataFrame()
    # fig_dir = Path(os.path.join(os.path.basename(FLAGS.output)), "figures")
    # fig_dir.mkdir(exist_ok=True, parents=True)
    # csv_out = Path(os.path.join(os.path.basename(FLAGS.output)), "csv")
    # csv_out.mkdir(exist_ok=True, parents=True)
    # for file in tqdm(all_files):
    #     eval_data = dataset.load_tfrecord_dataset([file], FLAGS.classes, FLAGS.size)
    #     eval_data = eval_data.map(lambda x, y: (tf.expand_dims(x, 0), y,))
    #     eval_data = eval_data.map(lambda x, y: (dataset.transform_images(x, FLAGS.size), y,))
    #
    #     results, mAP = calculate_metrics(eval_data, yolo)
    #     # results.to_csv(os.path.join(FLAGS.output, f"{os.path.basename(file)}.csv"))
    #     all_results = concat([all_results, results])
    #     mAP.plot()
    #     plt.show()
    #     # ax = sns.lineplot(x="Recall", y="Precision", data=results)
    #     # fig = ax.get_figure()
    #     # fig.savefig(os.path.join(fig_dir, f"pr_{os.path.basename(file)}.png"))
    # all_results.dropna().to_csv("all_results")
    # ax = sns.lineplot(x="Recall", y="Precision", data=all_results)
    # fig = ax.get_figure()
    # fig.savefig(os.path.join(fig_dir, f"pr_{all}.png"))


if __name__ == "__main__":
    app.run(main)
