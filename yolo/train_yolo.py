# %%
from helpers import get_feature_map, read_example, read_tfr, transform_images, transform_targets
from yolo.trainer import Trainer
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import tensorflow_datasets as tfds


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

# ds_train = tfds.load("coco/2014", split="train", shuffle_files=True)

#%%
trainer = Trainer(
    input_shape=input_shape,
    model_configuration="/data2/seals/tfrecords/yolo4.cfg",
    image_width=416,  # The original image width
    image_height=416,  # The original image height
    train_tf_record="/data2/seals/tfrecords/416_10/train.tfrecord",
    valid_tf_record="/data2/seals/tfrecords/416_10/test.tfrecord",
    classes_file="/data2/seals/tfrecords/416/classes.txt",
    output_path="/home/md273/model_zoo/416_yolo4_10",
    all_records_path="/data2/seals/tfrecords/416_10/416_test_all_records.csv",
    ground_truth={
        "train": "/data2/seals/tfrecords/416_10/416_train_all_records.csv",
        "valid": "/data2/seals/tfrecords/416_10/416_train_all_records.csv",
    },
)

# #%%
feature_map = get_feature_map()

#%%
with open("/data2/seals/tfrecords/416/classes.txt", "r") as classes_file:
    classes = [item.strip() for item in classes_file.readlines()]

# raw_dataset = tf.data.TFRecordDataset("/data2/seals/tfrecords/416/train.tfrecord")
# test = []
# for example in raw_dataset.shuffle(512).take(20):
#     test += [read_example(example, feature_map=feature_map, class_table=None, max_boxes=9)]

## TODO write new
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


# [y for x, y in dataset.take(10)]

## TODO: pickle settings
trainer.train(
    epochs=300,
    batch_size=8,
    learning_rate=1e-3,
    dataset_name="seal_416_yolo4",
    merge_evaluation=False,
    min_overlaps=0.5,
    # new_anchors_conf=anchors_conf,  # check step 6
    #  weights='/path/to/weights'  # If you're using DarkNet weights or resuming training
)

#%%
trainer.evaluate(
    "/home/md273/model_zoo/416/models/seal_416_model.tf", True, 1, 512, 0.5, True, True, True,
)
