import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv3D, ReLU, Conv1D, Permute,\
    GlobalAvgPool3D, Dense, MaxPooling2D, ZeroPadding2D, Dropout, Flatten, Add
from memory_profiler import profile
import os
import json
import copy
from typing import Tuple, Iterable
import sys

sys.path.append(os.path.join(os.getcwd(), 'configs'))
sys.path.append(os.path.join(os.getcwd(), 'utils'))
from global_namespace import PATH_TO_PROJECT
from video.preprocessing import BasicPreprocessing


def list_to_tuple(dictionary):
    spec = copy.deepcopy(dictionary)
    for key in spec:
        if isinstance(spec[key], list):
            for i in range(len(spec[key])):
                if isinstance(spec[key][i], list):
                    spec[key][i] = tuple(spec[key][i])
            spec[key] = tuple(spec[key])
    return spec


class Conv1x1(keras.layers.Layer):
    def __init__(self, dim, num_filters, strides, padding, activation, use_bias,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(Conv1x1, self).__init__()
        assert dim == 1 or dim == 2 or dim == 3, 'Bad dim for Conv1x1 layer'
        if dim == 1:
            component = Conv1D
        elif dim == 2:
            component = Conv2D
        elif dim == 3:
            component = Conv3D
        self.core_layer = component(
            num_filters,
            tuple(np.ones(dim, dtype=int)),
            padding=padding, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer,
            strides=strides, bias_initializer=bias_initializer
        )

    def call(self, input, **kwargs):
        return self.core_layer(input)


class SpatioTemporalSoftAttentionLayer(keras.layers.Layer):
    def __init__(self):
        super(SpatioTemporalSoftAttentionLayer, self).__init__()
        self.weight_generator = Conv1x1(
            dim=3, num_filters=1, strides=(1, 1, 1), padding='same', activation='softmax', use_bias=False,
            kernel_initializer='glorot_uniform', bias_initializer='zeros'
        )

    def call(self, inputs, **kwargs):
        weights = self.weight_generator(inputs)
        return tf.math.multiply(inputs, weights)


class SpatialSoftAttentionLayer(keras.layers.Layer):
    def __init__(self):
        super(SpatialSoftAttentionLayer, self).__init__()
        self.weight_generator = Conv1x1(
            dim=2, num_filters=1, strides=(1, 1), padding='same', activation='softmax', use_bias=False,
            kernel_initializer='glorot_uniform', bias_initializer='zeros'
        )

    def call(self, inputs, **kwargs):
        weights = self.weight_generator(inputs)
        return tf.math.multiply(inputs, weights)


class BasicBlock_R3D(keras.Model):
    def __init__(self, group_num, dim, striding_version, type_block=None):
        super(BasicBlock_R3D, self).__init__()
        principal_freeway_layer = Conv1x1
        self.layer = keras.Sequential([])
        freeway_stride = tuple(np.ones(dim, dtype=int))
        if dim == 2:
            principal_layer = Conv2D
        elif dim == 3:
            if type_block == '3D':
                principal_layer = Conv3D
            elif type_block == '2+1D':
                principal_layer = {'spatial': Conv2D, 'temporal': Conv1D}
        path_to_config = os.path.join(PATH_TO_PROJECT, 'models', 'video_models', 'R3D', 'R3D_block_config',
                                      'R3D_group{}_config.json'.format(group_num))
        with open(path_to_config) as json_obj:
            block_config = json.load(json_obj)
        if type_block != '2+1D':
            for layer_key in block_config:
                block_config[layer_key]['kernel_size'] = tuple(block_config[layer_key]['kernel_size'][-dim:])
                block_config[layer_key]['strides'] = tuple(block_config[layer_key]['strides'][-dim:])\
                    if striding_version == 'main' else tuple(dim * [1])
                freeway_stride = tuple(np.multiply(block_config[layer_key]['strides'], freeway_stride))\
                    if striding_version == 'main' else tuple(dim * [1])
                freeway_filters = block_config[layer_key]['filters']
                self.layer.add(principal_layer(**block_config[layer_key]))
        elif type_block == '2+1D':
            spatial_layers_config =copy.deepcopy(block_config)
            temporal_layers_config = copy.deepcopy(block_config)
            for layer_key in block_config:
                freeway_stride = tuple(np.multiply(block_config[layer_key]['strides'], freeway_stride)) if striding_version == 'main' else tuple(dim * [1])
                freeway_filters = block_config[layer_key]['filters']
                spatial_layers_config[layer_key]['kernel_size'] = tuple(
                    spatial_layers_config[layer_key]['kernel_size'][1:])
                temporal_layers_config[layer_key]['kernel_size'] = temporal_layers_config[layer_key]['kernel_size'][0]
                spatial_layers_config[layer_key]['strides'] = tuple(
                    spatial_layers_config[layer_key]['strides'][1:]) if striding_version == 'main' else (1, 1)
                temporal_layers_config[layer_key]['strides'] = temporal_layers_config[layer_key]['strides'][0] if striding_version == 'main' else 1
                self.layer.add(principal_layer['spatial'](**spatial_layers_config[layer_key]))
                self.layer.add(Permute((2, 3, 1, 4)))
                self.layer.add(principal_layer['temporal'](**temporal_layers_config[layer_key]))
                self.layer.add(Permute((3, 1, 2, 4)))
        self.freeway = keras.Sequential([
            principal_freeway_layer(
                dim=dim, num_filters=freeway_filters, strides=freeway_stride, padding='same', activation=None, use_bias=False,
                kernel_initializer='glorot_uniform', bias_initializer='zeros'
            )
        ])
        self.adding_layer = Add()

    def call(self, inputs, training=None, mask=None):
        return self.adding_layer([self.freeway(inputs), self.layer(inputs)])


class R3D(keras.Model):
    def __init__(self, spec):
        super(R3D, self).__init__()
        self.spec = spec
        self.attention_counter = 0
        self.indicator = ''
        for layer_or_block in spec:
            if layer_or_block.startswith('group'):
                for i in range(spec[layer_or_block]['num_blocks']):
                    setattr(self, layer_or_block + '_' + 'block_' + str(i+1), keras.Sequential([BasicBlock_R3D(int(layer_or_block[-1]),
                                                         spec[layer_or_block]['dim'][i],
                                                         'main' + self.indicator, spec[layer_or_block]['type'][i])], name=layer_or_block+'_block'+str(i+1)))
                    self.indicator = 'stride_applied!'
                self.indicator = ''
            else:
                setattr(self, layer_or_block, keras.Sequential([
                    eval(spec[layer_or_block]['type'])(**spec[layer_or_block]['config'])
                ], name=layer_or_block))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer_or_block in self.spec:
            if layer_or_block.startswith('group'):
                for i in range(self.spec[layer_or_block]['num_blocks']):
                    x = getattr(self, layer_or_block + '_' + 'block_' + str(i+1))(x)
            else:
                x = getattr(self, layer_or_block)(x)
        return x


class LRN(keras.layers.Layer):
    def __init__(self, depth_radius=5, bias=1, alpha=1, beta=0.5):
        super(LRN, self).__init__()
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs, **kwargs):
        return tf.nn.local_response_normalization(inputs, self.depth_radius, self.bias, self.alpha, self.beta)


class PlainModel(keras.Model):
    def __init__(self, spec):
        super(PlainModel, self).__init__()
        self.spec = spec
        for layer in spec:
            setattr(self, layer, keras.Sequential([
                eval(spec[layer]['type'])(**list_to_tuple(spec[layer]['config']))
            ], name=layer))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.spec:
            x = getattr(self, layer)(x)
        return x

@profile
def model_maker():
    with open('./models/video_models/simple_models/simple_v1.json') as json_obj:
        video_model_config = json.load(json_obj)
    with open("./models/audio_models/VGGM/VGGM_config.json") as json_obj:
        audio_model_config = json.load(json_obj)
    video_model = R3D(video_model_config)
    audio_model = PlainModel(audio_model_config)
    audio_model.build(input_shape=(None, 24, 64, 3))
    video_model.build(input_shape=(None, 80, 128, 128, 3))
    audio_model.summary()
    video_model.summary()

    video_input_layer = keras.Input(shape=(80, 128, 128, 3))
    audio_input_layer = keras.Input(shape=(24, 64, 3))
    video_preprocessing = BasicPreprocessing('type2', 3, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))(video_input_layer)
    video_net = R3D(video_model_config)(video_preprocessing)
    audio_net = PlainModel(audio_model_config)(audio_input_layer)
    model = keras.Model(
        inputs=[video_input_layer, audio_input_layer],
        outputs=[video_net, audio_net],
    )
    vid_batch = tf.random.uniform((32,80,128,128,3), -1, 1)
    audio_batch = tf.random.uniform((32, 24, 64, 3), -1, 1)

    v_o, a_o = model([vid_batch, audio_batch])

if __name__ == '__main__':
    model_maker()







