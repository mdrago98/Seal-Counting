# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D, Add, DepthwiseConv2D, GlobalAveragePooling2D, Flatten, Softmax, GlobalMaxPooling2D, Flatten, Reshape
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TerminateOnNaN, LambdaCallback
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from PIL import Image
from functools import wraps, partial, reduce
from tensorflow import zeros
from tensorflow.keras import Input, Model
from os import path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorboard
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)
from tensorflow.python.keras.regularizers import l2

get_ipython().run_line_magic('load_ext', 'tensorboard')


# %%
_DarknetConv2D = partial(Conv2D, padding='same')

@wraps(Conv2D)
def darknet_conv_2d(*args, **kwargs):
    default_dict = {'kernel_regularizer': l2(5e-4)}
    default_dict.update(kwargs)
    return _DarknetConv2D(*args, **default_dict)


# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2D, padding='same')


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet weight regularizer for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs.update(kwargs)
    return _DarknetConv2D(*args, **darknet_conv_kwargs)


@wraps(DepthwiseConv2D)
def DarknetDepthwiseConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return DepthwiseConv2D(*args, **darknet_conv_kwargs)


def Darknet_Depthwise_Separable_Conv2D_BN_Leaky(filters, kernel_size=(3, 3), block_id_str=None, **kwargs):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetDepthwiseConv2D(kernel_size, name='conv_dw_' + block_id_str, **no_bias_kwargs),
        BatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_dw_%s_leaky_relu' % block_id_str),
        Conv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        BatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_pw_%s_leaky_relu' % block_id_str))


def Depthwise_Separable_Conv2D_BN_Leaky(filters, kernel_size=(3, 3), block_id_str=None):
    """Depthwise Separable Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    return compose(
        DepthwiseConv2D(kernel_size, padding='same', name='conv_dw_' + block_id_str),
        BatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_dw_%s_leaky_relu' % block_id_str),
        Conv2D(filters, (1,1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        BatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_pw_%s_leaky_relu' % block_id_str))


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def bottleneck_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
    return compose(
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def bottleneck_x2_block(outer_filters, bottleneck_filters):
    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
    return compose(
        bottleneck_block(outer_filters, bottleneck_filters),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.nn.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.
    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x


def darknet53_body(x):
    '''Darknet53 body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

# %%
def darknet53(input_shape, input_tensor, pooling: str = 'avg', include_top: boolean = true, weights: str = 'imagenet', classes: int = 1000, **kwargs: dict):
    """Defines a Darket53 cnn model.

    Args:
        input_shape ([type]): The input tensor shape.
        input_tensor ([type]): The input tensor.
        pooling (str, optional): A string representing the pooling type. Options: avg and max. Defaults to avg.
        include_top (boolean, optional): true IFF the model is to include a top section for classification. Defaults to true.
        weights (str, optional): the wieghts to use in the preinitialised model.
        classes (int, optional): [description]. Defaults to 1000.
    """
    if not (weights in ('imagenet', None) or path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('Classes should be 1000 when using imagenet')
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor
    
    x = darknet53_body(img_input)
    if include_top:
        model_name='darknet53'
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Reshape((1, 1, 1024))(x)
        x = DarknetConv2D(classes, (1, 1))(x)
        x = Flatten()(x)
        x = Softmax(name='Predictions/Softmax')(x)
    else:
        model_name='darknet53_headless'
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    inputs = img_input

    # Create model.
    model = Model(inputs, x, name=model_name)

    # TODO: Load weights.

    return model

# %%
def train(args: dict, annotation_file: str, model, input_shape, log_dir: str = 'logs', val_split: float = 0.2, optimiser=Adam(lr=learning_rate, decay=5e-4)):
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, cooldown=0, min_lr=1e-10)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1)
    terminate_on_nan = TerminateOnNaN()
    
    callbacks = [logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan]
    # TODO: create data handler
    dataset = get_dataset(annotation_file)
    val_split = val_split
    num_val = int(len(dataset)*val_split)
    num_train = len(dataset) - num_val
    # data generator

    # TODO create yolo3 definition
    get_train_model = get_yolo3_train_model
    data_generator = yolo3_data_generator_wrapper

    # TODO: create Eval callback
    # eval_callback = EvalCallBack(args.model_type, dataset[num_train:], args['anchors'], args['class_names'], args['model_image_size'], args['model_pruning'], log_dir, eval_epoch_interval=args['eval_epoch_interval'], save_eval_checkpoint=args['save_eval_checkpoint'])
    # callbacks += [eval_callback]

    # TODO create model pruning callback and pruning end step

    model = get_train_model(args.model_type, args['anchors'], args['num_classes'], weights_path=args.weights_path, freeze_level=args['freeze_level'], optimizer=optimiser, label_smoothing=args['label_smoothing'], model_pruning=args['model_pruning'], pruning_end_step=pruning_end_step)
    model.summary()

    pass
# %%
