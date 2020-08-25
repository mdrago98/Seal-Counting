import tensorflow as tf


# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
IMAGE_FEATURE_MAP = {
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    "image/object/class/text": tf.io.VarLenFeature(tf.string),
}


@tf.function
def transform_targets_for_out(y_true, grid_size, anchor_idxs):
    """
    A function to transform the targets for training
    :param y_true: the ground truth labels
    :param grid_size: the grid size
    :param anchor_idxs: the anchor ids
    :return:
    """
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size, downsampling_factor=32):
    y_outs = []
    grid_size = size // downsampling_factor

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(
        box_wh[..., 1], anchors[..., 1]
    )
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_out(y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    """
    Transforms the input tensor down to the required size and centers integers to floats between 0 and 1
    :param x_train: the image tensor
    :param size: the size
    :return: the transformed tensor
    """
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


def parse_tfrecord(tfrecord, class_table, size, max_boxes: int = 100):
    """
    A function to parse an example from a tf record
    :param tfrecord: the record
    :param class_table: the class table
    :param size: the size of the image
    :param pad: true IFF the output tensor is to be padded
    :return:
    """
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_png(x["image/encoded"], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    class_text = tf.sparse.to_dense(x["image/object/class/text"], default_value="")
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack(
        [
            tf.sparse.to_dense(x["image/object/bbox/xmin"]),
            tf.sparse.to_dense(x["image/object/bbox/ymin"]),
            tf.sparse.to_dense(x["image/object/bbox/xmax"]),
            tf.sparse.to_dense(x["image/object/bbox/ymax"]),
            labels,
        ],
        axis=1,
    )

    if max_boxes != 0:
        paddings = [[0, max_boxes - tf.shape(y_train)[0]], [0, 0]]
        y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def load_tfrecord_dataset(files: list, class_file, size=416, max_boxes: int = 100):
    """
    Loads and parses the tf record dataset
    :param files: the list of files
    :param class_file: the class file path
    :param size: the size
    :param max_boxes: the maximum number of boxes contained in an image
    :return:
    """
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(
        tf.lookup.TextFileInitializer(
            class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"
        ),
        -1,
    )

    files = tf.data.Dataset.from_tensor_slices(files)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size, max_boxes))
