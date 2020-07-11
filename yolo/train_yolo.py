# %%
from yolo.trainer import Trainer

# %%
trainer = Trainer(
    input_shape=(416, 416, 3),
    model_configuration="/data2/seals/tfrecords/yolo3.cfg",
    image_width=416,  # The original image width
    image_height=416,  # The original image height
    train_tf_record="/data2/seals/tfrecords/416/train.tfrecord",
    valid_tf_record="/data2/seals/tfrecords/416/test.tfrecord",
    classes_file="/data2/seals/tfrecords/416/classes.txt",
)
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
