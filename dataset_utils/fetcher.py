import subprocess
from typing import Iterable
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Lock
from timeit import default_timer as timer
from pathos.multiprocessing import ProcessingPool as Pool
import shutil
import json
import sys

sys.path.append(os.path.join(os.getcwd(), 'configs'))
from global_namespace import PATH_TO_PERSISTENT_STORAGE


class DatasetFetcher(object):
    """
    Description: Base class for all dataset fetchers
    """

    def __init__(self, name: str, split: float, output_path: str,
                 class_labels: Iterable, train_csv_path: str = None,
                 test_csv_path: str = None, eval_csv_path: str = None, _rng: int = None):
        """
        Description: class constructor

        :param name: name of the dataset (currently supported: AudioSet, Kinetic700)
        :param split: if eval subset is not available, the split ratio for getting eval data from train data
        :param output_path: path to where the dataset should be saved
        :param class_labels: class labels we want from the dataset (Iterable)
        :param train_csv_path: path to csv config of train data
        :param test_csv_path: path to csv config of test data
        :param eval_csv_path: path to csv config of eval data
        :param _rng: seed for managing randomness (numpy)
        """

        self.name = name
        self.output_path = os.path.join(output_path, self.name)
        self.split = split
        self.labels = class_labels
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.eval_csv_path = eval_csv_path
        self.seed = _rng

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.mkdir(self.output_path)

        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            pass

        self.train_df = self._get_df(self.train_csv_path)
        self.train_df = self.train_df.reindex(np.random.permutation(self.train_df.index)).reset_index().drop(
            'index', axis=1)

        if self.eval_csv_path is None:
            num_eval = int(split * len(self.train_df.index))
            self.eval_df = self.train_df.loc[0:num_eval]
            self.train_df = self.train_df.loc[num_eval:len(self.train_df.index)].reset_index().drop('index', axis=1)
        else:
            self.eval_df = self._get_df(self.eval_csv_path)
            self.eval_df = self.eval_df.reindex(np.random.permutation(self.eval_df.index)).reset_index().drop(
                'index', axis=1)

        if self.test_csv_path is not None:
            self.test_df = self._get_df(self.test_csv_path)
            self.test_df = self.test_df.reindex(np.random.permutation(self.test_df.index)).reset_index().drop('index', axis=1)
        else:
            self.test_df = None

        self.paths = [os.path.join(self.output_path, branch) for branch in ['train', 'eval', 'test']]
        for path in self.paths:
            os.mkdir(path)

    def _get_df(self, path_to_csv: str) -> pd.DataFrame:
        """
        Description: function for translating the csv file to the format, class understands
        :param path_to_csv:

        :return: pandas.DataFrame
        """
        df = pd.read_csv(path_to_csv, sep='\n', header=1)
        df = df.iloc[:, 0].str.split(',', expand=True)
        df = df.iloc[:, 0:3].reset_index().drop('index', axis=1)
        return df

    def _dl_video(self, id: str, start: str, end: str, data_path: str, category: str, **kwargs) -> None:
        """
        Description: downloads video with a given id, trims it according to what it should be, and handle all sorts of error
        while downloading. It also logs every errors, and important events by a logger function.

        :param id:
        :param start:
        :param end:
        :param data_path:
        :param category:
        :param kwargs:
        :return:
        """
        start = str(int(float(start)))
        end = str(int(float(end)))
        url = 'https://www.youtube.com/embed/' + id + '?start=' + start + '&end=' + end
        print(url)

        data_file_path = os.path.join(data_path, id)

        subprocess.run(['youtube-dl', url, '-o', data_file_path])

        all_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
                     and not (f.startswith('.'))]
        coruption_flag =True in [file.startswith(id) for file in all_files]

        if not os.path.exists(data_file_path + '.mp4') and coruption_flag:
            my_stuff = [file for file in all_files if file.startswith(id)][0]
            path_to_bad_file = os.path.join(data_path, my_stuff)
            path_to_good_file = os.path.join(data_path, id + '.mp4')
            subprocess.run(['ffmpeg', '-i', path_to_bad_file, path_to_good_file])
            os.remove(path_to_bad_file)
        'ffmpeg -i input.mp4 -ss 01:19:27 -to 02:18:51 -c:v copy -c:a copy output.mp4'
        'ffmpeg -ss 00:10:45 -i input.avi -c:v libx264 -crf 18 -to 00:11:45 -c:a copy output.mp4'
        subprocess.run(
            ['ffmpeg', '-i', data_file_path + '.mp4', '-ss', start, '-to', end, '-c:v', 'libx264', '-crf', '18', '-c:a',
             'copy', data_file_path + '(t).mp4'])
        os.remove(data_file_path + '.mp4')

    def fetch_dataset(self, train_num: int = None, test_num: int = None, eval_num: int = None, num_processes: int = 1) -> None:
        """
        Description: the function user should use for fetching dataset

        :param train_num: number of training examples (if None, all of the available training data is fetched)
        :param test_num: number of testing examples (if None, all of the available testing data is fetched)
        :param eval_num: number of eval examples (if None, all of the available eval data is fetched)
        :param num_processes: number of independent processes we want to use for downloading
        :return: None
        """
        flag_train,\
        flag_eval,\
        flag_test\
            = train_num if train_num is not None else len(self.train_df.index),\
              eval_num if eval_num is not None else len(self.eval_df.index),\
              test_num if test_num is not None else len(self.test_df.index)

        video_ids = self.train_df[0].tolist()[0:flag_train] + self.eval_df[0].tolist()[0:flag_eval]
        starts = self.train_df[1].tolist()[0:flag_train] + self.eval_df[1].tolist()[0:flag_eval]
        ends = self.train_df[2].tolist()[0:flag_train] + self.eval_df[2].tolist()[0:flag_eval]
        paths = [self.paths[0]]*flag_train + [self.paths[1]]*flag_eval
        cats = ['train']*flag_train + ['eval']*flag_eval

        if self.test_df is not None:
            video_ids += self.test_df[0].tolist()[0:flag_test]
            starts += self.test_df[1].tolist()[0:flag_test]
            ends += self.test_df[2].tolist()[0:flag_test]
            paths += [self.paths[2]]*flag_test
            cats += ['test']*flag_test

        pool = Pool(processes=num_processes)
        output_train = pool.map(self._dl_video, video_ids, starts, ends, paths, cats)
        pool.close()
        pool.join()

class AudioSetFetcher(DatasetFetcher):
    """
    Description: child class for downloading AudioSet
    """
    def __init__(self, **kwargs):
        super(AudioSetFetcher, self).__init__(name='AudioSet', **kwargs)

    def _get_df(self, path_to_csv: str) -> pd.DataFrame:
        def _map(s):
            q = s.replace('"', '')
            return q.replace(' ', '')
        df = pd.read_csv(path_to_csv, sep='\n', header=2).applymap(_map)
        df = df.iloc[:, 0].str.split(',', expand=True)
        df = df[df.isin(self.labels).any(axis=1)]
        df = df.iloc[:, 0:3].reset_index().drop('index', axis=1)
        return df

class Kinetic700Fetcher(DatasetFetcher):
    """
    Description: child class for downloading Kinetic700
    """

    def __init__(self, **kwargs):
        super(Kinetic700Fetcher, self).__init__(name='Kinetic700', **kwargs)

        for label in self.labels:
            os.mkdir(os.path.join(self.paths[0], label))
            os.mkdir(os.path.join(self.paths[1], label))
        if self.test_csv_path is not None:
            self.test_df = super()._get_df(self.test_csv_path)
            self.test_df = self.test_df.reindex(np.random.permutation(self.test_df.index)).reset_index().drop(
                'index', axis=1)

    def _get_df(self, path_to_csv: str) -> pd.DataFrame:
        df = pd.read_csv(path_to_csv, sep='\n', header=1)
        df = df.iloc[:, 0].str.split(',', expand=True)
        df = df[df.isin(self.labels).any(axis=1)]
        df = df.iloc[:, 0:4].reset_index().drop('index', axis=1)
        cols = df.columns.tolist()
        cols.append(cols[0])
        cols = cols[1:len(cols)]
        df = df[cols]
        return df

    def _dl_video(self, id: str, start: str, end: str, data_path: str, category: str, class_label: str, **kwargs) -> None:
        start = str(int(float(start)))
        end = str(int(float(end)))
        url = 'https://www.youtube.com/embed/' + id + '?start=' + start + '&end=' + end
        print(url)

        if class_label is not None:
            data_path_class = os.path.join(data_path, class_label)
        else:
            data_path_class = data_path

        data_file_path = os.path.join(data_path_class, id)
        subprocess.run(['youtube-dl', url, '-o', data_file_path])

        all_files = [f for f in os.listdir(data_path_class) if os.path.isfile(os.path.join(data_path_class, f))
                     and not (f.startswith('.'))]
        coruption_flag = True in [file.startswith(id) for file in all_files]

        if not os.path.exists(data_file_path + '.mp4') and coruption_flag:
            my_stuff = [file for file in all_files if file.startswith(id)][0]
            path_to_bad_file = os.path.join(data_path_class, my_stuff)
            path_to_good_file = os.path.join(data_path_class, id + '.mp4')
            subprocess.run(['ffmpeg', '-i', path_to_bad_file, path_to_good_file])
            os.remove(path_to_bad_file)
        'ffmpeg -i input.mp4 -ss 01:19:27 -to 02:18:51 -c:v copy -c:a copy output.mp4'
        'ffmpeg -ss 00:10:45 -i input.avi -c:v libx264 -crf 18 -to 00:11:45 -c:a copy output.mp4'
        if not os.path.exists(data_file_path + '(t).mp4') and os.path.exists(data_file_path + '.mp4'):
            subprocess.run(
                ['ffmpeg', '-i', data_file_path + '.mp4', '-ss', start, '-to', end, '-c:v', 'libx264', '-crf', '18',
                 '-c:a',
                 'copy', data_file_path + '(t).mp4'])
            os.remove(data_file_path + '.mp4')

    def fetch_dataset(self, train_num: int = None, test_num: int = None, eval_num: int = None, num_processes:int = 1) -> None:
        flag_train,\
        flag_eval,\
        flag_test\
            = train_num if train_num is not None else len(self.train_df.index),\
              eval_num if eval_num is not None else len(self.eval_df.index),\
              test_num if test_num is not None else len(self.test_df.index)

        video_ids = self.train_df[1].tolist()[0:flag_train] + self.eval_df[1].tolist()[0:flag_eval]
        starts = self.train_df[2].tolist()[0:flag_train] + self.eval_df[2].tolist()[0:flag_eval]
        ends = self.train_df[3].tolist()[0:flag_train] + self.eval_df[3].tolist()[0:flag_eval]
        paths = [self.paths[0]]*flag_train + [self.paths[1]]*flag_eval
        cats = ['train']*flag_train + ['eval']*flag_eval
        labels = self.train_df[0].tolist()[0:flag_train] + self.eval_df[0].tolist()[0:flag_eval]
        self.train_df.head(flag_train).to_csv(os.path.join(self.paths[0], 'train_df.csv'), index_label=False)
        self.eval_df.head(flag_eval).to_csv(os.path.join(self.paths[1], 'eval_df.csv'), index_label=False)

        if self.test_df is not None:
            self.test_df.head(flag_test).to_csv(os.path.join(self.paths[2], 'test_df.csv'), index_label=False)
            video_ids += self.test_df[0].tolist()[0:flag_test]
            starts += self.test_df[1].tolist()[0:flag_test]
            ends += self.test_df[2].tolist()[0:flag_test]
            paths += [self.paths[2]]*flag_test
            cats += ['test']*flag_test
            labels += [None] * flag_test

        pool = Pool(processes=num_processes)
        output_train = pool.map(self._dl_video, video_ids, starts, ends, paths, cats, labels)
        pool.close()
        pool.join()

class CheckerKinetic700(object):
    def __init__(self, dl_path, train_info_path, test_info_path, eval_info_path):
        self.dl_path = dl_path
        self.info_paths = [train_info_path, eval_info_path, test_info_path]

    def check_file_in_path(self, path, logger):
        if not os.path.exists(path):
            logger.write(path + '\n')

    def check_df_in_path(self, df, path, logger, branch):
        for index, row in df.iterrows():
            file_path = os.path.join(path, row[3], row[0] + '(t).mp4') if branch != 'test' else os.path.join(path, row[0] + '(t).mp4')
            self.check_file_in_path(file_path, logger)

    def check(self):
        branchs = ['train', 'eval', 'test']
        logger = open(os.path.join(self.dl_path, 'info.txt'), 'w')
        dfs = [pd.read_csv(path) for path in self.info_paths]
        paths = [os.path.join(self.dl_path, branch) for branch in branchs]

        [self.check_df_in_path(df, _path, logger, branch) for df, _path, branch in zip(dfs, paths, branchs)]

        logger.close()




if __name__ == '__main__':
#     train_path_audioset = './datasets/datasets/AudioSetCSV/balanced_train_segments.csv'
#     eval_path_audioset = None
#     test_path_audioset = './datasets/datasets/AudioSetCSV/eval_segments.csv'
#     classes_audioset = ['/m/09x0r', "/m/05zppz", "/m/02zsn", "/m/0ytgt", "/m/01h8n0", "/m/02qldy", "/m/0261r1", "/m/0brhx"]
#     dl_path = './datasets/datasets/'
    # fetcher_config_audioset = {
    #     'split': 0.1,
    #     'output_path': dl_path,
    #     'class_labels': classes_audioset,
    #     'train_csv_path': train_path_audioset,
    #     'test_csv_path': test_path_audioset,
    #     'eval_csv_path': eval_path_audioset,
    #     '_rng': 1
    # }
#     #
    train_path_kinetic = './datasets/datasets/kinetics700_2020/train.csv'
    eval_path_kinetic = './datasets/datasets/kinetics700_2020/validate.csv'
    test_path_kinetic = './datasets/datasets/kinetics700_2020/test.csv'
    classes_kinetic = ['slapping', 'clapping', 'side kick', 'dribbling basketball',
                       'breaking boards', 'tapping pen', 'hitting baseball', 'ripping paper', 'playing drums']
    # dl_path = './datasets/datasets/'
    fetcher_config_kinetic = {
        'split': 0.1,
        'output_path': PATH_TO_PERSISTENT_STORAGE,
        'class_labels': classes_kinetic,
        'train_csv_path': train_path_kinetic,
        'test_csv_path': test_path_kinetic,
        'eval_csv_path': eval_path_kinetic,
        '_rng': 1
    }
    fetcher = Kinetic700Fetcher(**fetcher_config_kinetic)
    start = timer()
    fetcher.fetch_dataset(None, 1, None, num_processes=8)
    end = timer()
    print('time(s) = ', end-start)
    # train_info_path = os.path.join(PATH_TO_PERSISTENT_STORAGE, 'Kinetic700_source', 'train', 'train_df.csv')
    # eval_info_path = os.path.join(PATH_TO_PERSISTENT_STORAGE, 'Kinetic700_source', 'eval', 'eval_df.csv')
    # test_info_path = os.path.join(dl_path, 'Kinetic700_source', 'test', 'test_df.csv')
    # target = os.path.join(dl_path, 'Kinetic700_source')

    # chekcerkinetic = CheckerKinetic700(target, train_info_path, test_info_path, eval_info_path)
    # chekcerkinetic.check()