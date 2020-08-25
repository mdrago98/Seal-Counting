from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.models import Model
import tensorflow as tf
import numpy as np


def darknet_conv(x, filters, size, strides=1, batch_norm=True):
    """
    decalres the darknet convolution
    :param x: the input 
    :param filters: the number of filters
    :param size: the size of a filter
    :param strides: the number of strides 
    :param batch_norm: tru IFF batch normalisation is to be applied
    :return: the transformed input
    """
    if strides == 1:
        padding = "same"
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = "valid"
    x = Conv2D(
        filters=filters,
        kernel_size=size,
        strides=strides,
        padding=padding,
        use_bias=not batch_norm,
        kernel_regularizer=l2(0.0005),
    )(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def darknet_res(x, filters):
    """
    Defines the darknet residual block
    :param x: the input
    :param filters: the number of filters
    :return: the transformed residual 
    """
    prev = x
    x = darknet_conv(x, filters // 2, 1)
    x = darknet_conv(x, filters, 3)
    x = Add()([prev, x])
    return x


def blocking_convolution(x, filters, blocks):
    x = darknet_conv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = darknet_res(x, filters)
    return x


def out_conv(filters, name=None):
    """
    Declares the output convolution
    :param filters: the number of filters
    :param name: the name
    :return: the output conv function
    """

    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concatenate with skip
            x = darknet_conv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = darknet_conv(x, filters, 1)
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, filters, 1)
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def yolt_block(x, filters):
    x = darknet_conv(x, filters, 3)
    # prev = x
    x = darknet_conv(x, filters // 2, 1)
    x = darknet_conv(x, filters, 3)
    # x = Add()([prev, x])
    return x


def yolo_boxes(pred: tf.Tensor, anchors: np.array, classes: int) -> tuple:
    """
    A function to decode the yolo output into bounding boxes, confidence and the original record
    :param pred: the prediction tensor in the shape (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    :param anchors: the anchors of the relevant scale
    :param classes: the number of classes
    :return: a tuple of bbox, objectness, class_probs, pred_box
    """
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_out(filters, anchors: np.array, classes, name=None):
    """
    Defines the yolo output convolutions
    :param filters: the number of filters
    :param anchors: the anchors
    :param classes: the number of classes
    :param name: the name of the block
    :return: the transformed yolo output kernel
    """

    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(
            lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5))
        )(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


def yolt_block(x, filters):
    x = darknet_conv(x, filters, 3)
    # prev = x
    x = darknet_conv(x, filters // 2, 1)
    x = darknet_conv(x, filters, 3)
    # x = Add()([prev, x])
    return x


def yolo_nms(
    outputs, anchors, masks, classes, yolo_max_boxes, yolo_iou_threshold, yolo_score_threshold
):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold,
    )

    return boxes, scores, classes, valid_detections
