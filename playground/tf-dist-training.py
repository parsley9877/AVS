import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
from time import sleep
from moviepy.editor import VideoFileClip, AudioFileClip
sys.path.append(os.path.join(os.getcwd(), 'utils', 'sys'))
sys.path.append(os.path.join(os.getcwd(), 'utils'))
sys.path.append(os.path.join(os.getcwd(), 'models'))
sys.path.append(os.path.join(os.getcwd(), 'dataset_utils'))
sys.path.append(os.path.join(os.getcwd(), 'configs'))
from os_tools import check_fetched, log_fetched_dataset, PATH_TO_PERSISTENT_STORAGE,\
    check_shifted, if_exists_delete, filter_fetched_data, number_of_examples
from loader import VideoSetShifter, Video2TFRecord, TFRecord2Video, temp
from abstract_classes import R3D, PlainModel
from audio.preprocessing import MFCC
from video.preprocessing import BasicPreprocessing
from global_namespace import PATH_TO_SAMPLE_CHECKPOINT

# def worker_on_batch(batch):
#     out = weight * batch[0]
#     m.update_state(out, batch[1], sample_weight=batch[2])
#
# strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1'])
# with strategy.scope():
#     weight = tf.constant(0.5)
#     m = tf.keras.metrics.Accuracy()
# dataset = tf.data.Dataset.from_tensor_slices(([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0],
#                                               [0.4,1.0,1.5,2.0,2.5,3.5,3.5,4.0,4.5,5.0,5.5,6.0],
#                                               [0.7,1.0,1.0,1.0,1.0,0.76,1.0,1.0,1.0,1.0,1.0,1.0])).shuffle(13)
# dist_dataset = strategy.experimental_distribute_dataset(dataset)
# for batch in dist_dataset:
#     strategy.run(worker_on_batch, args=(batch,))
#
# print(m.result().numpy())
# for i in range(10):
#     for x,y,z in dataset:
#         print(x)
#     print('\n')

strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1'])
with strategy.scope():
    optimizer = tf.keras.optimizers.SGD()
    model = keras.Sequential([
        keras.layers.Dense(1)
    ])
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

manager = tf.train.CheckpointManager(
    checkpoint, PATH_TO_SAMPLE_CHECKPOINT, 3, keep_checkpoint_every_n_hours=None,
    checkpoint_name='ckpt', step_counter=None, checkpoint_interval=None,
    init_fn=None
)
print(model.weights)
# ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform((100,), -1, 1)).batch(2)
# dds = strategy.experimental_distribute_dataset(ds)

# for elem in dds:
#     print(model(elem))

# for i in range(10):
#     optimizer.lr.assign(float(i))
#     sleep(2)
#     manager.save()
#     print(i)
optimizer.lr.assign(2)
status = checkpoint.restore(manager.latest_checkpoint)
status.expect_partial()
# print(tf.train.list_variables(tf.train.latest_checkpoint(PATH_TO_SAMPLE_CHECKPOINT)))
# checkpoint.restore(tf.train.latest_checkpoint(PATH_TO_SAMPLE_CHECKPOINT))

#
print(optimizer.lr)
print(model.weights)


