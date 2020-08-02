import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from matplotlib import pyplot as plt
from yolo import dataset
import numpy as np
import cv2
import tensorflow as tf


def draw_labels(x, y, class_names):
    img = x[0].numpy()
    boxes, classes = tf.split(y[0], (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
    return img


train_data_eval = dataset.load_tfrecord_dataset(
    ["/data2/seals/tfrecords/512/train/StitchMICE_IHfl16_4_1022_CP_FINAL.tfrecord"],
    "../data/data_files/seal.names",
    512,
)
train_data_eval = train_data_eval.batch(1)
train_data_eval = train_data_eval.map(lambda x, y: (dataset.transform_images(x, 512), y,))

for i, (x, y) in enumerate(train_data_eval):
    if y[0][0][0] != 0 and y[0][0][2] != 0:
        img = draw_labels(x, y, {})
        plt.imshow(img)
        plt.title("my picture")
        plt.savefig(f"/data2/seals/tfrecords/512/sample/{i}.png")
