import json
import numpy as np
import tensorflow as tf
from absl import app, flags
import matplotlib.pyplot as plt
import matplotlib
import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(proj_dir)
matplotlib.use('Agg')

from utils.video.preprocessing import BasicPreprocessing

def main(argv):
    del argv
    config_path = os.path.join(proj_dir, 'test_experiments', 'BasicPreprocessing_test', 'BasicPreprocessing_config.json')
    output_path = os.path.join(proj_dir, 'test_experiments', 'BasicPreprocessing_test', 'output')
    with open(config_path) as json_obj:
        experiment_config = json.load(json_obj)
    experiment_config['adaption_data'] = eval(experiment_config['adaption_data'])
    processor = BasicPreprocessing(**experiment_config)
    processed_data = processor.process(experiment_config['adaption_data'])
    reconstructed_data = processor.inverse_process(processed_data)
    print('adaption data = ', experiment_config['adaption_data'])
    print('processed data = ', processed_data)
    print('reconstructed data = ', reconstructed_data)
    print(processed_data)
    print(reconstructed_data)
    print('reconstructed data == original data: ', (tf.experimental.numpy.allclose(experiment_config['adaption_data'], reconstructed_data)))
    print('adaption data shape: ', experiment_config['adaption_data'].shape)
    print('processed data shape: ', processed_data.shape)
    print('reconstructed data shape: ', reconstructed_data.shape)
    print('channel 0 mean - adaption data: ', tf.math.reduce_mean(experiment_config['adaption_data'][:, :, :, :, 0]))
    print('channel 0 mean - processed data: ', tf.math.reduce_mean(processed_data[:,:,:,:,0]))
    print('channel 0 mean - adaption data: ', tf.math.reduce_variance(experiment_config['adaption_data'][:,:,:,0]))
    print('channel 0 mean - processed data: ', tf.math.reduce_variance(processed_data[:,:,:,:,0]))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    fig1, ax1 = plt.subplots()
    ax1.imshow(experiment_config['adaption_data'].numpy()[0][0]/255.0)
    ax1.set_title('adaption_data')
    fig1.savefig(os.path.join(output_path, 'adaption_data.png'))
    fig2, ax2 = plt.subplots()
    ax2.imshow(reconstructed_data.numpy()[0][0]/255.0)
    ax2.set_title('reconstructed_data')
    fig2.savefig(os.path.join(output_path, 'reconstructed_data.png'))


if __name__ == '__main__':
    app.run(main)

