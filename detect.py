import os
import time
from glob import glob
from pathlib import Path

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from pandas import read_excel

from yolo import dataset
from yolo.models import YoloV3, YoloV3Tiny
from yolo.dataset import transform_images, load_tfrecord_dataset
from yolo.utils import draw_outputs
from matplotlib import pyplot as plt


flags.DEFINE_string("classes", "./data/data_files/seal.names", "path to classes file")
flags.DEFINE_string(
    "weights",
    "/home/md273/model_zoo/416_eager/checkpoints/yolov3_train_7.tf",
    "path to weights file",
)
flags.DEFINE_boolean("tiny", False, "yolov3 or yolov3-tiny")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_string("image", "./data/girl.png", "path to input image")
flags.DEFINE_string("tfrecord", "/data2/seals/tfrecords/416/test", "tfrecord instead of image")
flags.DEFINE_string("output", "/home/md273/model_zoo/416_eager/output", "path to output image")
flags.DEFINE_integer("num_classes", 80, "number of classes in the model")


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()

    # yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info("weights loaded")

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info("classes loaded")

    # if FLAGS.tfrecord:
    #     dataset = load_tfrecord_dataset(
    #         FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
    #     dataset = dataset.shuffle(20)
    #     img_raw, _label = next(iter(dataset.take(1)))
    # else:
    #     img_raw = tf.image.decode_image(
    #         open(FLAGS.image, 'rb').read(), channels=3)

    all_files = glob(os.path.join(FLAGS.tfrecord, "*.tfrecord"))[4:5]
    eval_data = dataset.load_tfrecord_dataset(all_files, FLAGS.classes, FLAGS.size)
    eval_data = eval_data.map(lambda x, y: (tf.expand_dims(x, 0), y,))
    eval_data = eval_data.map(lambda x, y: (dataset.transform_images(x, FLAGS.size), y,))
    imgs = [[]]
    file_props = read_excel(
        os.path.join(os.getcwd(), "data", "data_files", "pixel_coord.xlsx"),
        sheet_name="FileOverview",
    ).dropna()
    width = file_props[
        file_props["tiff_file"] == f"{os.path.splitext(os.path.basename(all_files[-1]))[0]}.tif"
    ]["image_width"].iloc[0]
    allocations_per_row = width / FLAGS.size
    dir = Path(os.path.join(FLAGS.output, "detections"))
    dir.mkdir(exist_ok=True, parents=True)
    for index, (image, labels) in enumerate(eval_data):
        boxes, scores, classes, nums = yolo(image)
        img = cv2.cvtColor(image[0].numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        # truth_score = [1 for _ in labels]
        # img = draw_outputs(img, (labels[:, :4], truth_score, labels[:, 4:5], nums), truth_score)
        imgs[-1].append(img) if len(imgs[-1]) + 1 <= allocations_per_row else imgs.append([img])
        cv2.imwrite(os.path.join(FLAGS.output, f"{index}.png"), img)
    # stitcher = cv2.Stitcher_create()
    # show_images(imgs, len(imgs[-1]))

    # img = tf.expand_dims(img_raw, 0)
    # img = transform_images(img, FLAGS.size)
    #
    # t1 = time.time()
    # boxes, scores, classes, nums = yolo(img)
    # t2 = time.time()
    # logging.info("time: {}".format(t2 - t1))
    #
    # logging.info("detections:")
    # for i in range(nums[0]):
    #     logging.info(
    #         "\t{}, {}, {}".format(
    #             class_names[int(classes[0][i])], np.array(scores[0][i]), np.array(boxes[0][i])
    #         )
    #     )
    #
    # img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    # cv2.imwrite(FLAGS.output, img)
    # logging.info("output saved to: {}".format(FLAGS.output))


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert (titles is None) or (len(images) == len(titles))
    n_images = len(images)
    fig, ax = plt.subplots(nrows=n_images, ncols=len(images[-1]), sharex=True, sharey=True,)
    for i, row in enumerate(images):
        for j, image in enumerate(row):
            ax[i, j].imshow(image)

    # for row in images:
    #     a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
    #     if image.ndim == 2:
    #         plt.gray()
    #     plt.imshow(image)
    #     a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.show()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
