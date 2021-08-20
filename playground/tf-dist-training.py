import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
from moviepy.editor import VideoFileClip, AudioFileClip
sys.path.append(os.path.join(os.getcwd(), 'utils', 'sys'))
sys.path.append(os.path.join(os.getcwd(), 'utils'))
sys.path.append(os.path.join(os.getcwd(), 'models'))
sys.path.append(os.path.join(os.getcwd(), 'dataset_utils'))
from os_tools import check_fetched, log_fetched_dataset, PATH_TO_PERSISTENT_STORAGE,\
    check_shifted, if_exists_delete, filter_fetched_data, number_of_examples
from loader import VideoSetShifter, Video2TFRecord, TFRecord2Video, temp
from abstract_classes import R3D, PlainModel
from audio.preprocessing import MFCC
from video.preprocessing import BasicPreprocessing

# def worker_on_batch(batch):
#     out = weight * batch[0]
#     m.update_state(out, batch[1], sample_weight=batch[2])
#
# strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1'])
# with strategy.scope():
#     weight = tf.constant(0.5)
#     m = tf.keras.metrics.Accuracy()
dataset = tf.data.Dataset.from_tensor_slices(([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0],
                                              [0.4,1.0,1.5,2.0,2.5,3.5,3.5,4.0,4.5,5.0,5.5,6.0],
                                              [0.7,1.0,1.0,1.0,1.0,0.76,1.0,1.0,1.0,1.0,1.0,1.0])).shuffle(13)
# dist_dataset = strategy.experimental_distribute_dataset(dataset)
# for batch in dist_dataset:
#     strategy.run(worker_on_batch, args=(batch,))
#
# print(m.result().numpy())
for i in range(10):
    for x,y,z in dataset:
        print(x)
    print('\n')


