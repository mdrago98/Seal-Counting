from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy

from .layers import darknet_conv, blocking_convolution, out_conv, yolo_boxes, yolo_out, yolo_nms
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
yolo_dense_2_layer_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])
yolo_dense_1_layer_anchor_masks = np.array([[0, 1, 2]])


def darknet_backbone(name=None, size=(None, None)):
    x = inputs = Input([size[0], size[0], 3])
    x = darknet_conv(x, 32, 3)
    x = blocking_convolution(x, 64, 1)
    x = blocking_convolution(x, 128, 2)  # skip connection
    x = x_36 = blocking_convolution(x, 256, 8)  # skip connection
    x = x_61 = blocking_convolution(x, 512, 8)
    x = blocking_convolution(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def dense_darknet(name=None, size=(None, None)):
    x = inputs = Input([size[0], size[0], 3])
    x = darknet_conv(x, 32, 3)
    x = blocking_convolution(x, 64, 1)
    x = x_36 = blocking_convolution(x, 128, 2)  # skip connection
    x = x_61 = blocking_convolution(x, 256, 8)  # skip connection
    x = blocking_convolution(x, 512, 8)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def yolo_3(
    size=None,
    channels=3,
    anchors=None,
    masks=yolo_anchor_masks,
    classes=7,
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

    outputs = Lambda(
        lambda x: yolo_nms(
            x,
            anchors,
            masks,
            classes,
            FLAGS.yolo_max_boxes,
            FLAGS.yolo_iou_threshold,
            FLAGS.yolo_score_threshold,
        ),
        name="yolo_nms",
    )((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name="yolov3")


def yolo_3_dual_scale(
    size=None,
    channels=3,
    anchors=None,
    masks=yolo_anchor_masks,
    classes=7,
    training=False,
    backbone=darknet_backbone,
):

    x = inputs = Input([size, size, channels], name="input")

    x_36, x_61, x = backbone(name="yolo_darknet")(x)

    x = out_conv(512, name="yolo_conv_0")(x)
    output_0 = yolo_out(512, len(masks[0]), classes, name="yolo_output_0")(x)

    x = out_conv(256, name="yolo_conv_1")((x, x_61))
    output_1 = yolo_out(256, len(masks[1]), classes, name="yolo_output_1")(x)

    if training:
        return Model(inputs, (output_0, output_1), name="yolov3")

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name="yolo_boxes_0")(
        output_0
    )
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name="yolo_boxes_1")(
        output_1
    )

    outputs = Lambda(
        lambda x: yolo_nms(
            x,
            anchors,
            masks,
            classes,
            FLAGS.yolo_max_boxes,
            FLAGS.yolo_iou_threshold,
            FLAGS.yolo_score_threshold,
        ),
        name="yolo_nms",
    )((boxes_0[:3], boxes_1[:3], boxes_1[:3]))

    return Model(inputs, outputs, name="yolov3_dual_scale")


def yolo_3_single_scale(
    size=None,
    channels=3,
    anchors=None,
    masks=yolo_anchor_masks,
    classes=7,
    training=False,
    backbone=darknet_backbone,
):

    x = inputs = Input([size, size, channels], name="input")

    x_36, x_61, x = backbone(name="yolo_darknet")(x)

    x = out_conv(512, name="yolo_conv_0")(x)
    output_0 = yolo_out(512, len(masks[0]), classes, name="yolo_output_0")(x)

    if training:
        return Model(inputs, outputs=[output_0], name="yolov3_dense_single")

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name="yolo_boxes_0")(
        output_0
    )

    outputs = Lambda(
        lambda x: yolo_nms(
            x,
            anchors,
            masks,
            classes,
            FLAGS.yolo_max_boxes,
            FLAGS.yolo_iou_threshold,
            FLAGS.yolo_score_threshold,
        ),
        name="yolo_nms",
    )((boxes_0[:3], boxes_0[:3], boxes_0[:3]))

    return Model(inputs, outputs, name="yolov3_dense_single_scale")


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

        # calculate all losses
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

        # sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss
