from absl import app, flags
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
proj_dir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(proj_dir)

import json
from scipy.io import wavfile
import tensorflow as tf
import numpy
from utils.audio.preprocessing import MFCC

def main(argv):
    del argv
    audio_path = os.path.join(proj_dir, 'datasets', 'sample_audio', 'audio.wav')
    config_path = os.path.join(proj_dir, 'test_experiments', 'MFCC_test', 'MFCC_config.json')
    output_path = os.path.join(proj_dir, 'test_experiments', 'MFCC_test', 'output')
    rate, data = wavfile.read(audio_path)
    with open(config_path) as json_obj:
        mfcc_config = json.load(json_obj)
    mfcc_config['winfunc'] = eval(mfcc_config['winfunc'])
    mfcc_obj = MFCC(mfcc_config, delta=2)
    mfcc_out = mfcc_obj.process(tf.constant(data, dtype=tf.float32))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    file_obj = open(os.path.join(output_path, 'output.txt'), 'w')
    file_obj.write('output shape: ' + str(mfcc_out.shape) + '\n' + 'MFCC')
    file_obj.close()


if __name__ == '__main__':
    app.run(main)
