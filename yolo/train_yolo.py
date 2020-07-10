# %%
import models

# %%
trainer = models.Trainer(
        input_shape=(416, 416, 3),
        model_configuration='/home/drago/PycharmProjects/CS5099-SealCounting/config/yolo3.cfg',
        image_width=416,  # The original image width
        image_height=416,  # The original image height
        train_tf_record='/data2/seals/tfrecords/416/train.tfrecord',
        valid_tf_record='/data2/seals/tfrecords/416/test.tfrecord'
)
