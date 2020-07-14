# %%
from helpers import get_feature_map, read_example, read_tfr, transform_images, transform_targets
from yolo.trainer import Trainer
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


#%%
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


#%%
input_shape = (416, 416, 3)

#%%
trainer = Trainer(
    input_shape=input_shape,
    model_configuration="/data2/seals/tfrecords/yolo3.cfg",
    image_width=416,  # The original image width
    image_height=416,  # The original image height
    train_tf_record="/data2/seals/tfrecords/back_up/train.tfrecord",
    valid_tf_record="/data2/seals/tfrecords/back_up/test.tfrecord",
    classes_file="/data2/seals/tfrecords/416/classes.txt",
    output_path="/home/md273/model_zoo/416",
)

# #%%
# feature_map = get_feature_map()
#
# #%%
# with open("/data2/seals/tfrecords/416/classes.txt", "r") as classes_file:
#     classes = [item.strip() for item in classes_file.readlines()]
#
# dataset = read_tfr(
#     "/data2/seals/tfrecords/416/train.tfrecord",
#     "/data2/seals/tfrecords/416/classes.txt",
#     get_feature_map(),
#     9,
# )
# dataset = dataset.shuffle(512)
# dataset = dataset.batch(8)
# dataset = dataset.map(lambda x, y: (transform_images(x, input_shape[0]), y))
# dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
#
# [y for x, y in dataset.take(10)]

trainer.train(
    epochs=100,
    batch_size=8,
    learning_rate=1e-3,
    dataset_name="dataset_name",
    merge_evaluation=False,
    min_overlaps=0.5,
    # new_anchors_conf=anchors_conf,  # check step 6
    #  weights='/path/to/weights'  # If you're using DarkNet weights or resuming training
)
