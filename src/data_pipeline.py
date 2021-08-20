from absl import flags
from absl import app
from absl import logging
import json
import sys
import os
import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(), 'utils', 'sys'))
sys.path.append(os.path.join(os.getcwd(), 'dataset_utils'))
from os_tools import check_fetched, log_fetched_dataset, PATH_TO_PERSISTENT_STORAGE,\
    check_shifted, if_exists_delete, filter_fetched_data, number_of_examples
from loader import VideoSetShifter, Video2TFRecord, TFRecord2Video, temp



flags.DEFINE_string('exp', None, 'path to a json containing setup parameters and configuration of a single experiment')
flags.mark_flag_as_required('exp')

def main(argv):

    del argv

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    ##################################################
    """
    loading experiment config:
    """
    with open(flags.FLAGS.exp) as handler:
        exp_config = json.load(handler)
    fetched_ds_path = os.path.join(PATH_TO_PERSISTENT_STORAGE ,exp_config['dataset']['name'])
    ##################################################
    """
    checking if dataset is fetched
    """
    fetched = check_fetched(exp_config['dataset']['name'])
    print(fetched)
    #################################################
    """
    after fetching, log
    """
    # path_to_fetch_log = os.path.join(fetched_ds_path, 'after_fetch_log')
    # if_exists_delete(path_to_fetch_log)
    # data_info = log_fetched_dataset(fetched_ds_path)
    # with open(os.path.join(path_to_fetch_log, 'after_fetch_log.json'), 'w') as file_obj:
    #     json.dump(data_info, file_obj, sort_keys=True, indent=4)
    # with open(os.path.join(path_to_fetch_log, 'after_fetch_log.json'), 'r') as json_obj:
    #     fetching_info =  json.load(json_obj)
    # print(sum(fetching_info['all_durations'])/len((fetching_info['all_durations'])))
    # print(sum(fetching_info['all_fpss']) / len((fetching_info['all_fpss'])))
    # print(sum([x for (x, y) in fetching_info['all_sizes']]) / len((fetching_info['all_sizes'])))
    # print(sum([y for (x, y) in fetching_info['all_sizes']]) / len((fetching_info['all_sizes'])))
    ##################################################
    """
    after fetching, filter
    """
    # filter_fetched_data(exp_config, fetched_ds_path)
    ##################################################
    """
    after fetching filtering, log
    """
    # print(number_of_examples(os.path.join(fetched_ds_path, 'train')))
    # print(number_of_examples(os.path.join(fetched_ds_path, 'test')))
    # print(number_of_examples(os.path.join(fetched_ds_path, 'eval')))
    ##################################################
    """
    shifting, cropping, fps setting
    """
    # shifter = VideoSetShifter(fetched_ds_path)
    # log = shifter.shift_set(16, exp_config['dataset']['shifting_style'], exp_config['dataset']['video_size'][0],
    #                         exp_config['dataset']['video_size'][1],
    #                         exp_config['dataset']['fps'], exp_config['dataset']['duration'])
    # print(log)
    ##################################################
    """
     after shifting, cropping, fps logging
    """
    # path_to_fetch_log = os.path.join(PATH_TO_PERSISTENT_STORAGE, 'avs_log')
    # print(number_of_examples(os.path.join(fetched_ds_path, 'train')))
    # print(number_of_examples(os.path.join(fetched_ds_path, 'test')))
    # print(number_of_examples(os.path.join(fetched_ds_path, 'eval')))
    #
    # data_info = log_fetched_dataset(fetched_ds_path)
    # with open(os.path.join(path_to_fetch_log, 'after_load_log.json'), 'w') as file_obj:
    #     json.dump(data_info, file_obj, sort_keys=True, indent=4)
    ##################################################
    """
     after shifting, cropping, fps logging
    """
    base_tfrecoed_path = os.path.join(fetched_ds_path, exp_config['dataset']['name'] + '_tfrecords')
    # records_path_train = os.path.join(base_tfrecoed_path, 'train')
    records_path_eval = os.path.join(base_tfrecoed_path, 'eval')
    # records_path_test = os.path.join(base_tfrecoed_path, 'test')
    # if_exists_delete(base_tfrecoed_path)
    # if_exists_delete(records_path_train)
    if_exists_delete(records_path_eval)
    # if_exists_delete(records_path_test)
    converter = Video2TFRecord()
    # failed_train = converter.convert(os.path.join(fetched_ds_path, 'train'),
    #                                  records_path_train, exp_config['dataset']['records_per_file'], 8)
    failed_eval = converter.convert(os.path.join(fetched_ds_path, 'eval'),
                                     records_path_eval, exp_config['dataset']['records_per_file'], 8)
    # failed_test = converter.convert(os.path.join(fetched_ds_path, 'test'),
    #                                  records_path_test, exp_config['dataset']['records_per_file'], 8)
    # print(failed_train)
    # print(failed_test)
    print(failed_eval)
    ##################################################
    """
    logging tfrecords
    """
    # subdirs = ['train', 'eval']
    # dec = TFRecord2Video()
    # vid_shape = []
    # audio_shape = []
    # for sub in subdirs:
    #     for file in [path for path in os.listdir(os.path.join(base_tfrecoed_path, sub)) if path.endswith('tfrecords')]:
    #         temp(dec.get_records(os.path.join(base_tfrecoed_path, sub, file)), vid_shape, audio_shape)
    # # audio_shape_idx = []
    # # for idx, elem in enumerate(vid_shape):
    # #     if elem[0] != 100:
    # #         vid_shape_idx.append(idx)
    # # for idx, elem in enumerate(audio_shape):
    # #     if elem[0] != 222264:
    # #         audio_shape_idx.append(idx)
    #
    # # print(vid_shape_idx)
    # # print(audio_shape_idx)
    # print(audio_shape)
    # print(vid_shape)




if __name__ == '__main__':
    app.run(main)