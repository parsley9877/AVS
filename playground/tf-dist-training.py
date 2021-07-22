from tensorflow import keras
import tensorflow as tf
import torch

print(tf.__version__)
# tf.debugging.set_log_device_placement(enabled=True)
all_cpu_devices = tf.config.get_visible_devices('CPU')
all_gpu_devices = tf.config.get_visible_devices('GPU')
print(all_gpu_devices)
print(all_cpu_devices)
tf.config.set_visible_devices([all_gpu_devices[0], all_gpu_devices[1], all_gpu_devices[2], all_cpu_devices[0]])
visible_gpu_devices = tf.config.get_visible_devices('GPU')
print(visible_gpu_devices)

a = tf.constant([1,2,3], dtype=tf.float32)

print(a.device)

# class Model(keras.Model):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.a = 10.0
#         self.b = 10.0
# strategy = tf.distribute.MirroredStrategy()