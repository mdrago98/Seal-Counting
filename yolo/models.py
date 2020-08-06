from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy

from .utils import broadcast_iou

flags.DEFINE_integer("yolo_max_boxes", 100, "maximum number of boxes per image")
flags.DEFINE_float("yolo_iou_threshold", 0.5, "iou threshold")
flags.DEFINE_float("yolo_score_threshold", 0.5, "score threshold")

# yolo_anchors = (
#     np.array(
#         [
#             (10, 13),
#             (16, 30),
#             (33, 23),
#             (30, 61),
#             (62, 45),
#             (59, 119),
#             (116, 90),
#             (156, 198),
#             (373, 326),
#         ],
#         np.float32,
#     )
#     / 416
# )
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = (
    np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)], np.float32) / 416
)
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def darknet_conv(x, filters, size, strides=1, batch_norm=True):
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


def darknet_backbone(name=None, size=(None, None)):
    x = inputs = Input([size[0], size[0], 3])
    x = darknet_conv(x, 32, 3)
    x = blocking_convolution(x, 64, 1)
    x = blocking_convolution(x, 128, 2)  # skip connection
    x = x_36 = blocking_convolution(x, 256, 8)  # skip connection
    x = x_61 = blocking_convolution(x, 512, 8)
    x = blocking_convolution(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def Darknet_deep(name=None, size=(None, None)):
    x = inputs = Input([size[0], size[0], 3])
    x = darknet_conv(x, 32 * 2, 3)
    x = blocking_convolution(x, 64 * 2, 1)
    x = blocking_convolution(x, 128 * 2, 2)  # skip connection
    x = x_36 = blocking_convolution(x, 256 * 2, 8)  # skip connection
    x = x_61 = blocking_convolution(x, 512 * 2, 8)
    x = blocking_convolution(x, 1024 * 2, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def out_conv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
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


def yolo_conv_tiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = darknet_conv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = darknet_conv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def yolo_out(filters, anchors, classes, name=None):
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
    x = darknet_conv(x, filters // 2, 1)
    x = darknet_conv(x, filters, 3)
    return x


def dense_darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = darknet_conv(x, 32, 3)
    x = MaxPool2D(2, 2, "same")(x)
    x = darknet_conv(x, 64, 3)
    x = MaxPool2D(2, 2, "same")(x)
    x = yolt_block(x, 124)
    x = MaxPool2D(2, 2, "same")(x)

    x = yolt_block(x, 256)
    x = MaxPool2D(2, 2, "same")(x)

    x = yolt_block(x, 512)
    x = MaxPool2D(2, 2, "same")(x)

    x = darknet_conv(x, 256, 1)
    x = darknet_conv(x, 512, 3)
    x = pass_through = darknet_conv(x, 1024, 3)

    x = darknet_conv(x, 1024, 3)
    x = darknet_conv(x, 1024, 1)
    x = darknet_conv(x, 32, 1)
    return tf.keras.Model(inputs, (pass_through, x), name=name)


def yolo3_dense(
    size=None,
    channels=3,
    anchors=yolo_tiny_anchors,
    masks=yolo_tiny_anchor_masks,
    classes=80,
    training=False,
):
    x = inputs = Input([size, size, channels], name="input")

    x_8, x = dense_darknet(name="yolo_darknet")(x)

    x = yolo_conv_tiny(256, name="yolo_conv_0")(x)
    output_0 = yolo_out(256, len(masks[0]), classes, name="yolo_output_0")(x)

    if training:
        return Model(inputs, output_0, name="yolov3")

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name="yolo_boxes_0")(
        output_0
    )
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name="yolo_nms")(boxes_0[:3])
    return Model(inputs, outputs, name="yolov3_tiny")


def yolo_boxes(pred: tf.Tensor, anchors: np.array, classes: int) -> tuple:
    """
    A fuinction to decode the yolo output into bounding boxes, confidence and the original record
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


def yolo_nms(outputs, anchors, masks, classes):
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
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold,
    )

    return boxes, scores, classes, valid_detections


def yolo_3(
    size=None,
    channels=3,
    anchors=None,
    masks=yolo_anchor_masks,
    classes=7,
    lite=False,
    training=False,
    backbone=darknet_backbone,
):
    x = inputs = Input([size, size, channels], name="input")

    x_36, x_61, x = backbone(name="yolo_darknet")(x)

    x = out_conv(512, name="yolo_conv_0")(x)
    output_0 = yolo_out(512, len(masks[0]), classes, name="yolo_output_0")(x)

    x = out_conv(256, name="yolo_conv_1")((x, x_61))
    output_1 = yolo_out(256, len(masks[1]), classes, name="yolo_output_1")(x)

    x = out_conv(128, name="yolo_conv_2")((x, x_36))
    output_2 = yolo_out(128, len(masks[2]), classes, name="yolo_output_2")(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name="yolov3")

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name="yolo_boxes_0")(
        output_0
    )
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name="yolo_boxes_1")(
        output_1
    )
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name="yolo_boxes_2")(
        output_2
    )

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name="yolo_nms")(
        (boxes_0[:3], boxes_1[:3], boxes_2[:3])
    )

    return Model(inputs, outputs, name="yolov3")


def yolo_loss(anchors: np.array, classes: int = 7, ignore_thresh: float = 0.5):
    """
    Calculates the loss function for Yolo
    :param anchors: the scale's anchor boxes
    :param classes: the number of classes
    :param ignore_thresh: the iou threashold to accept of reject a prediction
    :return: 
    """

    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(
                broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1
            ),
            (pred_box, true_box, obj_mask),
            tf.float32,
        )
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss
