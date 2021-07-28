import os.path
import subprocess
from typing import Iterable
import shutil
import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
from pathos.multiprocessing import ProcessingPool as Pool
import time
from scipy.io import wavfile

from utils.sys.os_tools import corrupted_mp4_finder, shifted_filter, non_mp4_finder,\
    bad_mp4_finder, bad_duration_finder, get_list_of_mp4, non_mp4_checker

class Video2TFRecord(object):
    """
    Description: Convert all mp4 videos in a path to a tfrecord file
    """
    def __init__(self):
        pass

    def convert(self, in_path, out_path, record_per_file):
        """
        Description: This is the function used by user. It gets the root path of some mp4 files
        and generate some tfrecord files in desired path based on them.

        :param in_path:
        :param out_path:
        :param record_per_file:
        :return:
        """
        list_of_mp4 = self._get_list_of_files(in_path)
        corrupted_mp4s = corrupted_mp4_finder(in_path)
        good_mp4s = [elem for elem in list_of_mp4 if elem[0] not in corrupted_mp4s]
        first_arg = []
        second_arg = []

        reminder = len(good_mp4s) % record_per_file

        k = 0
        for i in range(0, len(good_mp4s) - reminder, record_per_file):
            first_arg.append(good_mp4s[i:i + record_per_file])
            second_arg += [os.path.join(out_path, str(k) + '.tfrecords')]
            k += 1
        if reminder != 0:
            first_arg.append(good_mp4s[len(good_mp4s) - reminder:])
            second_arg += [os.path.join(out_path, str(k) + '.tfrecords')]

        for arg1, arg2 in zip(first_arg, second_arg):
            self.generate_tfrecord(arg1, arg2)


    def vid_encoder(self, vid):
        list_of_encoded = []
        for i in range(vid.shape[0]):
            list_of_encoded.append(tf.io.encode_jpeg(vid[i]).numpy())
        return list_of_encoded
    def generate_tfrecord(self, list_of_pathes_and_labels, out_path):
        with tf.io.TFRecordWriter(out_path) as writer:
            with tqdm(total=len(list_of_pathes_and_labels), colour='#00ff00', desc='tfrecord progress') as pbar:
                for file, label in list_of_pathes_and_labels:
                    video_numpy, fps, num_frames, height, width = self.get_frames(file)
                    encoded_video = self.vid_encoder(video_numpy)
                    audio_numpy, sr = self._get_audio_waveform(file)
                    audio_numpy_shape = audio_numpy.shape[0]
                    audio_numpy = audio_numpy.reshape([audio_numpy.shape[0] * audio_numpy.shape[1]])
                    tf_example = self.create_example(encoded_video, fps, num_frames, height,
                                                     width, audio_numpy, audio_numpy_shape, sr,
                                                     os.path.basename(file), label)
                    writer.write(tf_example.SerializeToString())
                    pbar.update(1)
    def _get_audio_waveform(self, path_to_mp4):
        clip = VideoFileClip(path_to_mp4)
        audio = clip.audio
        audio.write_audiofile(os.path.join(os.path.dirname(path_to_mp4), 'temp.wav'))
        sample_rate, data = wavfile.read(os.path.join(os.path.dirname(path_to_mp4), 'temp.wav'))
        os.remove(os.path.join(os.path.dirname(path_to_mp4), 'temp.wav'))
        clip.close()
        return data, sample_rate

    def get_frames(self, path_to_mp4):
        cap = cv2.VideoCapture(path_to_mp4)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # assert (height == 0 or width == 0 or fps == 0 or num_frames == 0), path + ': Corrupted File!!'
        # print(fps, num_frames, height, width)
        channels = 3
        video_numpy = np.zeros([num_frames, height, width, channels], dtype=np.uint16)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            video_numpy[i, :, :, :] = frame
            i += 1
        cap.release()
        cv2.destroyAllWindows()
        return video_numpy, fps, num_frames, height, width

    def bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    def audio_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))

    def float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def byte_feature_list(self, value):
        """Returns a list of float_list from a float / double."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    # Create the features dictionary.
    def create_example(self, vid, fps, num_frames, h, w, audio, audio_shape, sr, id, label):
        feature = {
            "vid": self.byte_feature_list(vid),
            "w": self.int64_feature(w),
            "h": self.int64_feature(h),
            "fps": self.int64_feature(fps),
            "num_frames": self.int64_feature(num_frames),
            "sr": self.int64_feature(sr),
            "audio_shape": self.int64_feature(audio_shape),
            "id": self.bytes_feature(id),
            "label": self.bytes_feature(label),
            "audio": self.audio_feature(audio),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    def _get_list_of_files(self, path):
        listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(path):
            if False in [f.endswith('txt') for f in filenames]:
                listOfFiles += [(os.path.join(dirpath, file), os.path.basename(dirpath)) for file in filenames if file.endswith('mp4')]
        return listOfFiles

class TFRecord2Video(object):
    """
    Description: Convert a tfrecord to a tf dataset and load it
    """

    def __init__(self):
        pass

    def get_records(self, path_to_record):
        """
        Description: this is the function used by user. It gets path to tfr file, and generate a tf.dataset
        based on it.
        :param path_to_record:
        :return:
        """
        # Create the dataset object from tfrecord file(s)
        raw_dataset = tf.data.TFRecordDataset(path_to_record)
        dataset = raw_dataset.map(self.parse_tfrecord_fn)
        return dataset

    def parse_tfrecord_fn(self, example):
        feature_description = {
            "vid": tf.io.VarLenFeature(tf.string),
            "w": tf.io.FixedLenFeature([], tf.int64),
            "h": tf.io.FixedLenFeature([], tf.int64),
            "fps": tf.io.FixedLenFeature([], tf.int64),
            "num_frames": tf.io.FixedLenFeature([], tf.int64),
            "sr": tf.io.FixedLenFeature([], tf.int64),
            "audio_shape": tf.io.FixedLenFeature([], tf.int64),
            "id": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["vid"] = tf.sparse.to_dense(example["vid"])
        example["audio"] = tf.io.decode_raw(
            example["audio"], np.int16, little_endian=True, fixed_length=None, name=None
        )
        example['audio'] = tf.reshape(example['audio'], [example['audio_shape'], 2])
        return example


class VideoShifter(object):
    """
    Description: Baseclass for shifting videos in a directory video
    """
    def __init__(self):
        pass
    def shift(self, path_to_mp4, out_path, shift_number):
        """
        Description: Shift a single mp4 file, and store it as a new mp4 in out_path
        :param path_to_mp4:
        :param out_path:
        :param shift_number:
        :return:
        """
        # assert out_path.endswith('.mp4')
        try:
            original_clip = VideoFileClip(path_to_mp4)
        except:
            return path_to_mp4
        original_audio = original_clip.audio
        fps = original_clip.fps
        duration = original_clip.duration
        assert np.abs(shift_number) < duration, path_to_mp4
        assert fps != 0
        new_duration = duration - np.abs(shift_number)
        t_video_start = 0 if shift_number < 0 else np.abs(shift_number)
        t_video_end = new_duration if shift_number < 0 else duration
        t_audio_start = np.abs(shift_number) if shift_number < 0 else 0
        t_audio_end = duration if shift_number < 0 else new_duration
        new_video_clip = original_clip.subclip(t_video_start, t_video_end)
        new_audio_clip = original_audio.subclip(t_audio_start, t_audio_end)
        shifted_clip = new_video_clip.set_audio(new_audio_clip)
        shifted_clip.write_videofile(out_path, fps, audio=True)
        original_clip.close()
        return None
class VideoSetShifter(VideoShifter):
    """
    Description: Class for shifting a dataset
    """
    def __init__(self, path_to_videos):
        super(VideoSetShifter, self).__init__()
        self.path = path_to_videos
        self.train_path = os.path.join(self.path, 'train')
        self.test_path = os.path.join(self.path, 'test')
        self.eval_path = os.path.join(self.path, 'eval')
    def shift_set(self, num_processes, style):
        shifted_filter(self.train_path)
        shifted_filter(self.test_path)
        shifted_filter(self.eval_path)

        print(non_mp4_checker(self.train_path), non_mp4_finder(self.train_path))
        print(non_mp4_checker(self.test_path), non_mp4_finder(self.test_path))
        print(non_mp4_checker(self.eval_path), non_mp4_finder(self.eval_path))
        #
        print(bad_duration_finder(self.train_path, 3.01, 100))
        print(bad_duration_finder(self.eval_path, 3.01, 100))
        print(bad_duration_finder(self.test_path, 3.01, 100))
        #
        print(bad_mp4_finder(self.train_path))
        print(bad_mp4_finder(self.eval_path))
        print(bad_mp4_finder(self.test_path))

        print(corrupted_mp4_finder(self.train_path))
        print(corrupted_mp4_finder(self.eval_path))
        print(corrupted_mp4_finder(self.test_path))

        time.sleep(5)
        listOfFiles_train = get_list_of_mp4(self.train_path)
        listOfFiles_eval = get_list_of_mp4(self.eval_path)
        listOfFiles_test = get_list_of_mp4(self.test_path)
        # shifter = VideoShifter()
        if style == 1:
            regions = [(-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (3, 2)]
        elif style == 2:
            regions = [(-2, -1), (-1, 0), (0, 1), (1, 2)]
        file_args = [val for val in listOfFiles_train for _ in (0, len(regions) - 1)] \
                    + [val for val in listOfFiles_eval for _ in (0, len(regions) - 1)] \
                    + [val for val in listOfFiles_test for _ in (0, len(regions) - 1)]
        shifted_path_args = []
        random_args = []
        for idx, file in enumerate(file_args):
            region = regions[idx % len(regions)]
            dir_path = os.path.dirname(file)
            shifted_path_args += [os.path.join(dir_path, str(region) + '(shifted)' + os.path.basename(file))]
            random_args += [np.random.uniform(region[0], region[1])]
        pool = Pool(processes=num_processes)
        output_train = pool.map(self.shift, file_args, shifted_path_args, random_args)
        pool.close()
        pool.join()
        return output_train


# def temp(ds, i):
#     for record in ds:
#         audio = (record['audio'])
#         id = record['id']
#         w = record['w']
#         h = record['h']
#         fps = record['fps']
#         num_frames = record['num_frames']
#         audio_shape = record['audio_shape']
#         frames = record['vid']
#         label = record['label']
#         sr = record['sr']
#         break
#     wavfile.write(os.path.join(base_path, str(i) + '.wav'), 44100, audio.numpy())
#     print('id = ', id)
#     print('w = ', w)
#     print('h = ', h)
#     print('fps = ', fps)
#     print('num_frames = ', num_frames)
#     print('audio_shape = ', audio_shape)
#     print('label = ', label)
#     print('sr = ', sr)
#     print('frames = ', decode_video(frames))
#     print('audio = ', audio)
#



# if __name__ == '__main__':
#     base_path = './datasets/datasets'
#     kinetic_base_path = './datasets/datasets/Kinetic700'
#     records_path = os.path.join(base_path, 'sample_tfrecords')
#
#     if os.path.exists(records_path):
#         shutil.rmtree(records_path)
#         os.mkdir(records_path)
#
#     converter = Video2TFRecordKinetic700()
#     converter.convert(os.path.join(kinetic_base_path, 'train'), os.path.join(base_path, 'sample_tfrecords'), 20)
#     print([path for path in os.listdir(records_path) if path.endswith('tfrecords')])
#     dec = TFRecord2VideoKinetic700()
#     ds = []
#     for file in [path for path in os.listdir(records_path) if path.endswith('tfrecords')]:
#         ds.append(dec.get_records(os.path.join(records_path, file)))
#
#     for idx, set in enumerate(ds):
#         temp(set, idx)






