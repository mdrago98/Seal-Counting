from tensorflow.keras.layers import (
    ZeroPadding2D,
    BatchNormalization,
    LeakyReLU,
    Conv2D,
    Add,
    Input,
    UpSampling2D,
    Concatenate,
    Lambda,
    MaxPooling2D,
)
import sys


from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import os
import io
import configparser
from collections import defaultdict
from .utils.layer_utils import get_boxes, timer, default_logger, Mish

sys.path.append('..')


class BaseModel:
    def __init__(
            self,
            input_shape,
            model_configuration,
            classes=80,
            anchors=None,
            masks=None,
            max_boxes=100,
            iou_threshold=0.5,
            score_threshold=0.5,
    ):
        """
        Initialize yolo model.
        Args:
            input_shape: tuple(n, n, c)
            model_configuration: Path to DarkNet cfg file containing configuration.
            classes: Number of classes(defaults to 80 for Coco objects)
            anchors: numpy array of anchors (x, y) pairs
            masks: numpy array of masks.
            max_boxes: Maximum boxes in a single image.
            iou_threshold: Minimum overlap that counts as a valid detection.
            score_threshold: Minimum confidence that counts as a valid detection.
        """
        assert any(
            (
                '3' in model_configuration,
                '4' in model_configuration,
                'Invalid model configuration',
            )
        )
        self.version_anchors = {
            'v3': np.array(
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
            ),
            'v4': np.array(
                [
                    (12, 16),
                    (19, 36),
                    (40, 28),
                    (36, 75),
                    (76, 55),
                    (72, 146),
                    (142, 110),
                    (192, 243),
                    (459, 401),
                ]
            ),
        }
        self.version_masks = {
            'v3': np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]]),
            'v4': np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        }
        self.current_layer = 1
        self.input_shape = input_shape
        self.classes = classes
        self.anchors = anchors
        if anchors is None:
            if '3' in model_configuration:
                self.anchors = self.version_anchors['v3']
            if '4' in model_configuration:
                self.anchors = self.version_anchors['v4']
        if self.anchors[0][0] > 1:
            self.anchors = self.anchors / input_shape[0]
        self.masks = masks
        if masks is None:
            if '3' in model_configuration:
                self.masks = self.version_masks['v3']
            if '4' in model_configuration:
                self.masks = self.version_masks['v4']
        self.funcs = (
            ZeroPadding2D,
            BatchNormalization,
            LeakyReLU,
            Conv2D,
            Add,
            Input,
            UpSampling2D,
            Concatenate,
            Lambda,
            Mish,
            MaxPooling2D,
        )
        self.func_names = [
            'zero_padding',
            'batch_normalization',
            'leaky_relu',
            'conv2d',
            'add',
            'input',
            'up_sample',
            'concat',
            'lambda',
            'mish',
            'maxpool2d',
        ]
        self.layer_names = {
            func.__name__: f'layer_CURRENT_LAYER_{name}'
            for func, name in zip(self.funcs, self.func_names)
        }
        self.shortcuts = []
        self.previous_layer = None
        self.training_model = None
        self.inference_model = None
        self.output_indices = []
        self.output_layers = []
        self.max_boxes = max_boxes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.model_configuration = model_configuration
        self.model_layers = []

    def apply_func(self, func, x=None, *args, **kwargs):
        """
        Apply a function from self.funcs and increment layer count.
        Args:
            func: func from self.funcs.
            x: image tensor.
            *args: func args
            **kwargs: func kwargs

        Returns:
            result of func
        """
        name = self.layer_names[func.__name__].replace(
            'CURRENT_LAYER', f'{self.current_layer}'
        )
        result = func(name=name, *args, **kwargs)
        self.current_layer += 1
        if x is not None:
            return result(x)
        return result

    def read_dark_net_cfg(self):
        """
        Read model configuration from DarkNet cfg file.

        Returns:
            output_stream
        """
        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        with open(self.model_configuration) as cfg:
            for line in cfg:
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    adjusted_section = f'{section}_{section_counters[section]}'
                    section_counters[section] += 1
                    line = line.replace(section, adjusted_section)
                output_stream.write(line)
        output_stream.seek(0)
        return output_stream

    def get_nms(self, outputs):
        """
        Apply non-max suppression and get valid detections.
        Args:
            outputs: yolo model outputs.

        Returns:
            boxes, scores, classes, valid_detections
        """
        boxes, conf, type_ = [], [], []
        for output in outputs:
            boxes.append(
                tf.reshape(
                    output[0],
                    (tf.shape(output[0])[0], -1, tf.shape(output[0])[-1]),
                )
            )
            conf.append(
                tf.reshape(
                    output[1],
                    (tf.shape(output[1])[0], -1, tf.shape(output[1])[-1]),
                )
            )
            type_.append(
                tf.reshape(
                    output[2],
                    (tf.shape(output[2])[0], -1, tf.shape(output[2])[-1]),
                )
            )
        bbox = tf.concat(boxes, axis=1)
        confidence = tf.concat(conf, axis=1)
        class_probabilities = tf.concat(type_, axis=1)
        scores = confidence * class_probabilities
        (
            boxes,
            scores,
            classes,
            valid_detections,
        ) = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
            scores=tf.reshape(
                scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])
            ),
            max_output_size_per_class=self.max_boxes,
            max_total_size=self.max_boxes,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
        )
        return boxes, scores, classes, valid_detections

    def create_convolution(self, cfg_parser, section):
        """
        Create convolution layers.
        Args:
            cfg_parser: Model configuration cfg parser.
            section: cfg section

        Returns:
            None
        """
        filters = int(cfg_parser[section]['filters'])
        size = int(cfg_parser[section]['size'])
        stride = int(cfg_parser[section]['stride'])
        pad = int(cfg_parser[section]['pad'])
        activation = cfg_parser[section]['activation']
        batch_normalize = 'batch_normalize' in cfg_parser[section]
        padding = 'same' if pad == 1 and stride == 1 else 'valid'
        if filters == 255:
            filters = 3 * (self.classes + 5)
        if stride > 1:
            self.previous_layer = self.apply_func(
                ZeroPadding2D, self.previous_layer, ((1, 0), (1, 0))
            )
        convolution_layer = self.apply_func(
            Conv2D,
            self.previous_layer,
            filters=filters,
            kernel_size=size,
            strides=(stride, stride),
            use_bias=not batch_normalize,
            padding=padding,
            kernel_regularizer=l2(0.0005),
        )
        if batch_normalize:
            convolution_layer = self.apply_func(
                BatchNormalization, convolution_layer
            )
        self.previous_layer = convolution_layer
        if activation == 'linear':
            self.model_layers.append(self.previous_layer)
        if activation == 'leaky':
            act_layer = self.apply_func(
                LeakyReLU, self.previous_layer, alpha=0.1
            )
            self.previous_layer = act_layer
            self.model_layers.append(act_layer)
        if activation == 'mish':
            act_layer = self.apply_func(Mish, self.previous_layer)
            self.previous_layer = act_layer
            self.model_layers.append(act_layer)

    def create_route(self, cfg_parser, section):
        """
        Create concatenation layer.
        Args:
            cfg_parser: Model configuration cfg parser.
            section: cfg section

        Returns:
            None
        """
        ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
        layers = [self.model_layers[i] for i in ids]
        if len(layers) > 1:
            concatenate_layer = self.apply_func(Concatenate, layers)
            self.model_layers.append(concatenate_layer)
            self.previous_layer = concatenate_layer
        else:
            skip_layer = layers[0]
            self.model_layers.append(skip_layer)
            self.previous_layer = skip_layer

    def create_max_pool(self, cfg_parser, section):
        """
        Create max pooling layer.
        Args:
            cfg_parser: Model configuration cfg parser.
            section: cfg section

        Returns:
            None
        """
        size = int(cfg_parser[section]['size'])
        stride = int(cfg_parser[section]['stride'])
        layer = self.apply_func(
            MaxPooling2D,
            self.previous_layer,
            pool_size=(size, size),
            strides=(stride, stride),
            padding='same',
        )
        self.model_layers.append(layer)
        self.previous_layer = layer

    def create_shortcut(self, cfg_parser, section):
        """
        Create shortcut layer.
        Args:
            cfg_parser: Model configuration cfg parser.
            section: cfg section

        Returns:
            None
        """
        index = int(cfg_parser[section]['from'])
        activation = cfg_parser[section]['activation']
        assert activation == 'linear', 'Only linear activation supported.'
        layer = self.apply_func(
            Add, [self.model_layers[index], self.previous_layer]
        )
        self.model_layers.append(layer)
        self.previous_layer = layer

    def create_up_sample(self, cfg_parser, section):
        """
        Create up sample layer
        Args:
            cfg_parser: Model configuration cfg parser.
            section: cfg section

        Returns:
            None
        """
        stride = int(cfg_parser[section]['stride'])
        assert stride == 2, 'Only stride=2 supported.'
        layer = self.apply_func(UpSampling2D, self.previous_layer, stride)
        self.model_layers.append(layer)
        self.previous_layer = layer

    def create_output_layer(self):
        """
        Create output layer.

        Returns:
            None
        """
        self.output_indices.append(len(self.model_layers))
        x = self.model_layers[-1]
        x = self.apply_func(
            Lambda,
            x,
            lambda item: tf.reshape(
                item,
                (
                    -1,
                    tf.shape(item)[1],
                    tf.shape(item)[2],
                    3,
                    self.classes + 5,
                ),
            ),
        )
        self.model_layers.append(x)
        self.previous_layer = self.model_layers[-1]

    def create_section(self, section, cfg_parser):
        """
        Create a section from the model configuration file.
        Args:
            cfg_parser: Model configuration cfg parser.
            section: cfg section

        Returns:
            None
        """
        if section.startswith('convolutional'):
            self.create_convolution(cfg_parser, section)
        if section.startswith('route'):
            self.create_route(cfg_parser, section)
        if section.startswith('maxpool'):
            self.create_max_pool(cfg_parser, section)
        if section.startswith('shortcut'):
            self.create_shortcut(cfg_parser, section)
        if section.startswith('upsample'):
            self.create_up_sample(cfg_parser, section)
        if section.startswith('yolo'):
            self.create_output_layer()

    @timer(default_logger)
    def create_models(self):
        """
        Create training and inference yolo models.

        Returns:
            training, inference models
        """
        input_initial = self.apply_func(Input, shape=self.input_shape)
        cfg_out = self.read_dark_net_cfg()
        cfg_parser = configparser.ConfigParser()
        cfg_parser.read_file(cfg_out)
        self.output_indices = []
        self.previous_layer = input_initial
        for section in cfg_parser.sections():
            self.create_section(section, cfg_parser)
        if len(self.output_indices) == 0:
            self.output_indices.append(len(self.model_layers) - 1)
        self.output_layers.extend(
            [self.model_layers[i] for i in self.output_indices]
        )
        if '4' in self.model_configuration:
            self.output_layers.reverse()
        self.training_model = Model(
            inputs=input_initial, outputs=self.output_layers
        )
        output_0, output_1, output_2 = self.output_layers
        boxes_0 = self.apply_func(
            Lambda,
            output_0,
            lambda item: get_boxes(
                item, self.anchors[self.masks[0]], self.classes
            ),
        )
        boxes_1 = self.apply_func(
            Lambda,
            output_1,
            lambda item: get_boxes(
                item, self.anchors[self.masks[1]], self.classes
            ),
        )
        boxes_2 = self.apply_func(
            Lambda,
            output_2,
            lambda item: get_boxes(
                item, self.anchors[self.masks[2]], self.classes
            ),
        )
        outputs = self.apply_func(
            Lambda,
            (boxes_0[:3], boxes_1[:3], boxes_2[:3]),
            lambda item: self.get_nms(item),
        )
        self.inference_model = Model(
            input_initial, outputs, name='inference_model'
        )
        default_logger.info('Training and inference models created')
        return self.training_model, self.inference_model

    @timer(default_logger)
    def load_weights(self, weights_file):
        """
        Load DarkNet weights or checkpoint/pre-trained weights.
        Args:
            weights_file: .weights or .tf file path.

        Returns:
            None
        """
        assert weights_file.split('.')[-1] in [
            'tf',
            'weights',
        ], 'Invalid weights file'
        assert (
            self.classes == 80 if weights_file.endswith('.weights') else 1
        ), f'DarkNet model should contain 80 classes, {self.classes} is given.'
        if weights_file.endswith('.tf'):
            self.training_model.load_weights(weights_file)
            default_logger.info(f'Loaded weights: {weights_file} ... success')
            return
        with open(weights_file, 'rb') as weights_data:
            default_logger.info(f'Loading pre-trained weights ...')
            major, minor, revision, seen, _ = np.fromfile(
                weights_data, dtype=np.int32, count=5
            )
            self.model_layers = [
                layer
                for layer in self.training_model.layers
                if id(layer) not in [id(item) for item in self.output_layers]
            ]
            self.model_layers.sort(
                key=lambda layer: int(layer.name.split('_')[1])
            )
            self.model_layers.extend(self.output_layers)
            for i, layer in enumerate(self.model_layers):
                current_read = weights_data.tell()
                total_size = os.fstat(weights_data.fileno()).st_size
                if current_read == total_size:
                    break
                print(
                    f'\r{round(100 * (current_read / total_size))}%\
                    t{current_read}/{total_size}',
                    end='',
                )
                if 'conv2d' not in layer.name:
                    continue
                next_layer = self.model_layers[i + 1]
                b_norm_layer = (
                    next_layer
                    if 'batch_normalization' in next_layer.name
                    else None
                )
                filters = layer.filters
                kernel_size = layer.kernel_size[0]
                input_dimension = layer.get_input_shape_at(-1)[-1]
                convolution_bias = (
                    np.fromfile(weights_data, dtype=np.float32, count=filters)
                    if b_norm_layer is None
                    else None
                )
                bn_weights = (
                    np.fromfile(
                        weights_data, dtype=np.float32, count=4 * filters
                    ).reshape((4, filters))[[1, 0, 2, 3]]
                    if (b_norm_layer is not None)
                    else None
                )
                convolution_shape = (
                    filters,
                    input_dimension,
                    kernel_size,
                    kernel_size,
                )
                convolution_weights = (
                    np.fromfile(
                        weights_data,
                        dtype=np.float32,
                        count=np.product(convolution_shape),
                    )
                        .reshape(convolution_shape)
                        .transpose([2, 3, 1, 0])
                )
                if b_norm_layer is None:
                    try:
                        layer.set_weights(
                            [convolution_weights, convolution_bias]
                        )
                    except ValueError:
                        pass
                if b_norm_layer is not None:
                    layer.set_weights([convolution_weights])
                    b_norm_layer.set_weights(bn_weights)
            assert len(weights_data.read()) == 0, 'failed to read all data'
        default_logger.info(f'Loaded weights: {weights_file} ... success')
        print()
