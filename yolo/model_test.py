import os
from functools import reduce
from glob import glob
from os import environ

from tensorflow.python.keras import Input

from yolo import darknet_backbone, dataset
import tensorflow as tf
import numpy as np
import seaborn as sns
from yolo.models import (
    darknet_conv,
    blocking_convolution,
    dense_darknet,
    yolo3_dense,
)
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K


size = 800
environ["CUDA_VISIBLE_DEVICES"] = "-1"

anchors = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_anchors = (
    np.array(
        [
            (10, 13),
            (16, 30),
            (33, 23),
            (30, 61),
            (62, 45),
            (59, 119),
            (116, 90),
            (156, 198),
            (373, 326),
        ],
        np.float32,
    )
    / 416
)

test = dense_darknet()
test.summary()

test = yolo3_dense(416, training=True)


def backbone(name=None, size=(None, None)):
    x = inputs = Input([size[0], size[0], 3])
    x = darknet_conv(x, 32, 3)
    x = blocking_convolution(x, 64, 1)
    x = blocking_convolution(x, 128, 2)  # skip connection
    x = x_36 = blocking_convolution(x, 256, 8)  # skip connection
    x = x_61 = blocking_convolution(x, 512, 8)
    x = blocking_convolution(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


all_files = glob(os.path.join("/data2/seals/tfrecords/608/train", "*.tfrecord"))
train_dataset = dataset.load_tfrecord_dataset(all_files, "../data/data_files/seal.names", size)


def stage_1(mult=1):
    x = inputs = Input([None, None, 3])
    x = darknet_conv(x, 32 * mult, 3)
    return tf.keras.Model(inputs, x)


def stage_2(mult=1):
    x = inputs = Input([None, None, 3])
    x = darknet_conv(x, 32 * mult, 3)
    x = blocking_convolution(x, 64 * mult, 1)
    return tf.keras.Model(inputs, x)


def stage_3(mult=1):
    x = inputs = Input([None, None, 3])
    x = darknet_conv(x, 32 * mult, 3)
    x = blocking_convolution(x, 64 * mult, 1)
    x = blocking_convolution(x, 128 * mult, 2)
    return tf.keras.Model(inputs, x)


def stage_4(mult=1):
    x = inputs = Input([None, None, 3])
    x = darknet_conv(x, 32 * mult, 3)
    x = blocking_convolution(x, 64 * mult, 1)
    x = blocking_convolution(x, 128 * mult, 2)
    x = x_36 = blocking_convolution(x, 256 * mult, 8)
    return tf.keras.Model(inputs, x)


def stage_5(mult=1):
    x = inputs = Input([None, None, 3])
    x = darknet_conv(x, 32 * mult, 3)
    x = blocking_convolution(x, 64 * mult, 1)
    x = blocking_convolution(x, 128 * mult, 2)
    x = x_36 = blocking_convolution(x, 256 * mult, 8)
    x = x_61 = blocking_convolution(x, 512 * mult, 8)
    return tf.keras.Model(inputs, x)


def stage_6(mult=1):
    x = inputs = Input([None, None, 3])
    x = darknet_conv(x, 32 * mult, 3)
    x = blocking_convolution(x, 64 * mult, 1)
    x = blocking_convolution(x, 128 * mult, 2)
    x = x_36 = blocking_convolution(x, 256 * mult, 8)
    x = x_61 = blocking_convolution(x, 512 * mult, 8)
    x = blocking_convolution(x, 10248 * mult, 4)
    return tf.keras.Model(inputs, x)


stage_1().summary()
stage_2().summary()
stage_3().summary()
stage_4().summary()
stage_5().summary()
stage_6().summary()


def multiply(args):
    return ()


size_alloc = []


def iterate_size(size, train_dataset, mult=1):
    train_dataset = train_dataset.batch(2)
    train_dataset = train_dataset.map(
        lambda x, y: (
            dataset.transform_images(x, size),
            dataset.transform_targets(y, yolo_anchors, anchors, size),
        )
    )
    size_alloc = []
    for x_train, _ in train_dataset.take(1):
        test(x_train)
        size_alloc += [tf.size(x_train)]
        size_alloc += [reduce(lambda x, y: x * y, stage_1(mult)(x_train).shape)]
        size_alloc += [reduce(lambda x, y: x * y, stage_2(mult)(x_train).shape)]
        size_alloc += [reduce(lambda x, y: x * y, stage_3(mult)(x_train).shape)]
        size_alloc += [reduce(lambda x, y: x * y, stage_4(mult)(x_train).shape)]
        size_alloc += [reduce(lambda x, y: x * y, stage_5(mult)(x_train).shape)]
        size_alloc += [reduce(lambda x, y: x * y, stage_6(mult)(x_train).shape)]
    return size_alloc, np.sum([K.count_params(w) for w in stage_6(mult).trainable_weights])


for size in [416, 512, 608, 800, 1024]:
    sizes, train_param = iterate_size(size, train_dataset)
    print(f"{size}: {sizes}")
    print(f"trainable params {train_param}")
    size_alloc += [sizes]

frame = pd.DataFrame(np.array(size_alloc).T, columns=[416, 512, 608, 800, 1024])
ax = sns.lineplot(data=frame)
ax.set_ylabel("Size")
ax.set_xlabel("darknet_backbone Blocks")
plt.show()
print(size_alloc)
