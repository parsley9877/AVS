import json
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
import sys
sys.path.append(os.path.join(os.getcwd(), 'dataset_utils'))
sys.path.append(os.path.join(os.getcwd(), 'utils'))
sys.path.append(os.path.join(os.getcwd(), 'models'))
from abstract_classes import SpatioTemporalSoftAttentionLayer, BasicBlock_R3D, R3D, LRN, PlainModel, ParallelNet
from loader import TFRecord2Video
from visualization import disp_video
from audio.preprocessing import MFCC
from video.preprocessing import BasicPreprocessing
from tensorflow.keras.layers import Conv3D, Conv2D, Dense
from tensorflow.keras.layers import ReLU, Softmax
from tensorflow import keras
def temp(ds):
    for record in ds:
        audio = (record['audio'])
        id = record['id']
        w = record['w']
        h = record['h']
        fps = record['fps']
        num_frames = record['num_frames']
        audio_shape = record['audio_shape']
        frames = record['vid']
        label = record['label']
        sr = record['sr']
        mis = record['misalignment']
        # frames = tf.map_fn(fn=decode_frame, elems=frames, fn_output_signature=tf.uint8)
        break
    print('id = ', id)
    print('w = ', w)
    print('h = ', h)
    print('fps = ', fps)
    print('num_frames = ', num_frames)
    print('audio_shape = ', audio_shape)
    print('label = ', label)
    print('sr = ', sr)
    print('frames = ', type(frames), frames.shape, frames.dtype)
    print('audio = ', audio)
    print('misalignment = ', mis)
def crop_frames(x):
    cropper = CenterCrop(100, 100)
    x['vid'] = cropper(x['vid'])
    return x
def sample_frames(x):
    preprocessor = BasicPreprocessing('type2', 3, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    x['vid'] = preprocessor(x['vid'][:150])
    return x
def sample_audio(x):
    specs = {
        "samplerate": 16000,
        "winlen": 0.025,
        "winstep": 0.01,
        "numcep": 13,
        "nfilt": 26,
        "nfft": 512,
        "lowfreq": 0,
        "highfreq": 8000,
        "preemph": 0.97,
        "ceplifter": 22,
        "appendEnergy": True,
        "winfunc": np.hamming
    }
    mfcc = MFCC(specs, 2)
    x['audio'] = tf.numpy_function(mfcc.process, [x['audio'][:300000]], tf.float32)
    return x
if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3"])
    dataset = tf.data.Dataset.from_tensor_slices(20 * [1, 2, 3, 4, 5]).batch(20)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    with strategy.scope():
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
        optimizer = tf.keras.optimizers.SGD()






    # path_to_R3D_config = './models/video_models/R3D/R3D_SoftSTAttention/R3D18_SoftSTAttention.json'
    # path_to_audio_model = './models/audio_models/CNNS/CNNS_config.json'
    # path_to_audio_model = './models/audio_models/VGGM/VGGM_config.json'
    # with open(path_to_R3D_config) as json_obj:
    #     video_model_spec = json.load(json_obj)
    # with open(path_to_audio_model) as json_obj:
    #     audio_model_spec = json.load(json_obj)
    # video_net = R3D(video_model_spec)
    # audio_net = PlainModel(audio_model_spec)
    #
    # input_audio_batch = tf.random.uniform((1, 45, 200, 3), -1, 1)
    # input_video_batch = tf.random.uniform((1, 150, 100, 100, 3), -1, 1)
    #
    # input_video = keras.Input(shape=(150, 100, 100, 3))
    # input_audio = keras.Input(shape=(45, 200, 3))
    #
    # ao = audio_net(input_audio)
    # vo = video_net(input_video)
    #
    # model = keras.Model(
    #     inputs=[input_video, input_audio],
    #     outputs=[vo, ao],
    # )
    # model.summary()

    # with tf.GradientTape() as tape:
    #     out = model([input_video_batch, input_audio_batch])
    #     loss = tf.reduce_sum(out[0] - out[1])
    # def mul(listi):
    #     tem = 1
    #     for item in list(listi):
    #         tem *= item
    #     return tem
    # grads = tape.gradient(loss, model.trainable_weights)






















