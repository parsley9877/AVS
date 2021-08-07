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
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.mkdir(self.output_path)
        self.split = split
        self.labels = class_labels
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.eval_csv_path = eval_csv_path
        self.seed = _rng
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
        self.train_path = os.path.join(self.output_path, 'train')
        self.eval_path = os.path.join(self.output_path, 'eval')
        self.test_path = os.path.join(self.output_path, 'test')
        os.mkdir(self.train_path)
        os.mkdir(self.test_path)
        os.mkdir(self.eval_path)
        self.logger = open(os.path.join(self.output_path, 'log.txt'), 'w')

        self.total_train_fails = 0
        self.total_eval_fails = 0
        self.total_test_fails = 0

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

    def _convert_format(self, data_path: str) -> None:
        """
        Description: convert the format of all videos in data_path, to mp4

        :param data_path:
        :return: None
        """
        bad_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
                     and not (f.startswith('.') or f.endswith('mp4'))]
        for file in bad_files:
            path_to_bad_file = os.path.join(data_path, file)
            path_to_good_file = os.path.join(data_path, file[0:file.find('.')] + '.mp4')
            subprocess.run(['ffmpeg', '-i', path_to_bad_file, path_to_good_file])
            os.remove(os.path.join(data_path, file))

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
        if not coruption_flag:
            self.logger.write(url)
            self.logger.write(',  Failed in ' + category + '\n')
            setattr(self, 'total_' + category + '_fails', getattr(self, 'total_' + category + '_fails') + 1)
        else:
            if not os.path.exists(data_file_path + '.mp4'):
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
        if num_processes == 1:
            with tqdm(total=flag_train, colour='#00ff00', desc='train') as pbar:
                for idx, row in self.train_df.iterrows():
                    if idx > flag_train:
                        break
                    self._dl_video(row[0], row[1], row[2], self.train_path, category='train')
                    pbar.update(1)
            with tqdm(total=flag_eval, colour='#00ff00', desc='eval') as pbar:
                for idx, row in self.eval_df.iterrows():
                    if idx > flag_eval:
                        break
                    self._dl_video(row[0], row[1], row[2], self.eval_path, category='eval')
                    pbar.update(1)
            if self.test_df is not None:
                with tqdm(total=flag_test, colour='#00ff00', desc='test') as pbar:
                    for idx, row in self.test_df.iterrows():
                        if idx > flag_test:
                            break
                        self._dl_video(row[0], row[1], row[2], self.test_path, category='test')
                        pbar.update(1)
        else:
            video_ids = self.train_df[0].tolist()[0:flag_train] + self.eval_df[0].tolist()[0:flag_eval]
            starts = self.train_df[1].tolist()[0:flag_train] + self.eval_df[1].tolist()[0:flag_eval]
            ends = self.train_df[2].tolist()[0:flag_train] + self.eval_df[2].tolist()[0:flag_eval]
            paths = [self.train_path]*flag_train + [self.eval_path]*flag_eval
            cats = ['train']*flag_train + ['eval']*flag_eval
            if self.test_df is not None:
                video_ids += self.test_df[0].tolist()[0:flag_test]
                starts += self.test_df[1].tolist()[0:flag_test]
                ends += self.test_df[2].tolist()[0:flag_test]
                paths += [self.test_path]*flag_test
                cats += ['test']*flag_test
            pool = Pool(processes=num_processes)
            output_train = pool.map(self._dl_video, video_ids, starts, ends, paths, cats)
            pool.close()
            pool.join()
        self.logger.write('\n\n')
        self.logger.write('total failures in train: ' + str(self.total_train_fails) +'\n')
        self.logger.write('total failures in eval: ' + str(self.total_eval_fails) + '\n')
        self.logger.write('total failures in test: ' + str(self.total_test_fails) + '\n')


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
            os.mkdir(os.path.join(self.train_path, label))
            os.mkdir(os.path.join(self.eval_path, label))
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

        subprocess.run(['youtube-dl', url, '-o', data_file_path])
        all_files = [f for f in os.listdir(data_path_class) if os.path.isfile(os.path.join(data_path_class, f))
                     and not (f.startswith('.'))]
        coruption_flag = True in [file.startswith(id) for file in all_files]
        if not coruption_flag:
            self.logger.write(url)
            self.logger.write(',  Failed in ' + category + ' ,' + str(class_label) + '\n')
            setattr(self, 'total_' + category + '_fails', getattr(self, 'total_' + category + '_fails') + 1)
        else:
            if not os.path.exists(data_file_path + '.mp4'):
                my_stuff = [file for file in all_files if file.startswith(id)][0]
                path_to_bad_file = os.path.join(data_path_class, my_stuff)
                path_to_good_file = os.path.join(data_path_class, id + '.mp4')
                subprocess.run(['ffmpeg', '-i', path_to_bad_file, path_to_good_file])
                os.remove(path_to_bad_file)
            'ffmpeg -i input.mp4 -ss 01:19:27 -to 02:18:51 -c:v copy -c:a copy output.mp4'
            'ffmpeg -ss 00:10:45 -i input.avi -c:v libx264 -crf 18 -to 00:11:45 -c:a copy output.mp4'
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
        if num_processes == 1:
            with tqdm(total=flag_train, colour='#00ff00', desc='train') as pbar:
                for idx, row in self.train_df.iterrows():
                    if idx > flag_train:
                        break
                    self._dl_video(row[1], row[2], row[3], self.train_path, category='train', class_label=row[0])
                    pbar.update(1)
            with tqdm(total=flag_eval, colour='#00ff00', desc='eval') as pbar:
                for idx, row in self.eval_df.iterrows():
                    if idx > flag_eval:
                        break
                    self._dl_video(row[1], row[2], row[3], self.eval_path, category='eval', class_label=row[0])
                    pbar.update(1)
            if self.test_df is not None:
                with tqdm(total=flag_test, colour='#00ff00', desc='test') as pbar:
                    for idx, row in self.test_df.iterrows():
                        if idx > flag_test:
                            break
                        self._dl_video(row[0], row[1], row[2], self.test_path, category='test', class_label=None)
                        pbar.update(1)
        else:
            video_ids = self.train_df[1].tolist()[0:flag_train] + self.eval_df[1].tolist()[0:flag_eval]
            starts = self.train_df[2].tolist()[0:flag_train] + self.eval_df[2].tolist()[0:flag_eval]
            ends = self.train_df[3].tolist()[0:flag_train] + self.eval_df[3].tolist()[0:flag_eval]
            paths = [self.train_path]*flag_train + [self.eval_path]*flag_eval
            cats = ['train']*flag_train + ['eval']*flag_eval
            labels = self.train_df[0].tolist()[0:flag_train] + self.eval_df[0].tolist()[0:flag_eval]
            self.train_df.head(flag_train).to_csv(os.path.join(self.train_path, 'train_df.csv'), index_label=False)
            self.eval_df.head(flag_eval).to_csv(os.path.join(self.eval_path, 'eval_df.csv'), index_label=False)
            if self.test_df is not None:
                self.test_df.head(flag_test).to_csv(os.path.join(self.test_path, 'test_df.csv'), index_label=False)
                video_ids += self.test_df[0].tolist()[0:flag_test]
                starts += self.test_df[1].tolist()[0:flag_test]
                ends += self.test_df[2].tolist()[0:flag_test]
                paths += [self.test_path]*flag_test
                cats += ['test']*flag_test
                labels += [None] * flag_test
            pool = Pool(processes=num_processes)
            output_train = pool.map(self._dl_video, video_ids, starts, ends, paths, cats, labels)
            pool.close()
            pool.join()
        self.logger.write('\n\n')
        self.logger.write('total failures in train: ' + str(self.total_train_fails) +'\n')
        self.logger.write('total failures in eval: ' + str(self.total_eval_fails) + '\n')
        self.logger.write('total failures in test: ' + str(self.total_test_fails) + '\n')

class CheckerKinetic700(object):
    def __init__(self, dl_path, train_info_path, test_info_path, eval_info_path):
        self.dl_path = dl_path
        self.train_info_path = train_info_path
        self.test_info_path = test_info_path
        self.eval_info_path = eval_info_path
    def check(self):
        logger = open(os.path.join(self.dl_path, 'info.txt'), 'w')
        df_train = pd.read_csv(self.train_info_path)
        df_test = pd.read_csv(self.test_info_path)
        df_eval = pd.read_csv(self.eval_info_path)
        train_path = os.path.join(self.dl_path, 'train')
        eval_path = os.path.join(self.dl_path, 'eval')
        test_path = os.path.join(self.dl_path, 'test')
        for index, row in df_train.iterrows():
            if not os.path.exists(os.path.join(train_path, row[3], row[0] + '(t).mp4')):
                logger.write('train: ' + row[0] + ', ' + row[3] + '\n')
        for index, row in df_eval.iterrows():
            if not os.path.exists(os.path.join(eval_path, row[3], row[0] + '(t).mp4')):
                logger.write('eval: ' + row[0] + ', ' + row[3] + '\n')
        for index, row in df_test.iterrows():
            if not os.path.exists(os.path.join(test_path, row[0] + '(t).mp4')):
                logger.write('test: ' + row[0] + '\n')
        logger.close()




if __name__ == '__main__':
#     train_path_audioset = './datasets/datasets/AudioSetCSV/balanced_train_segments.csv'
#     eval_path_audioset = None
#     test_path_audioset = './datasets/datasets/AudioSetCSV/eval_segments.csv'
#     classes_audioset = ['/m/09x0r', "/m/05zppz", "/m/02zsn", "/m/0ytgt", "/m/01h8n0", "/m/02qldy", "/m/0261r1", "/m/0brhx"]
#     dl_path = './datasets/datasets/'
#     # fetcher_config_audioset = {
#     #     'split': 0.1,
#     #     'output_path': dl_path,
#     #     'class_labels': classes_audioset,
#     #     'train_csv_path': train_path_audioset,
#     #     'test_csv_path': test_path_audioset,
#     #     'eval_csv_path': eval_path_audioset,
#     #     '_rng': 1
#     # }
#     #
    train_path_kinetic = './datasets/datasets/kinetics700_2020/train.csv'
    eval_path_kinetic = './datasets/datasets/kinetics700_2020/validate.csv'
    test_path_kinetic = './datasets/datasets/kinetics700_2020/test.csv'
    classes_kinetic = ['slapping', 'playing drums', 'clapping', 'playing tennis', 'tapping pen']
    # #154 + 209 + 100 + 111 +
    dl_path = './datasets/datasets/'
    fetcher_config_kinetic = {
        'split': 0.1,
        'output_path': dl_path,
        'class_labels': classes_kinetic,
        'train_csv_path': train_path_kinetic,
        'test_csv_path': test_path_kinetic,
        'eval_csv_path': eval_path_kinetic,
        '_rng': 1
    }
    # fetcher = Kinetic700Fetcher(**fetcher_config_kinetic)
    # start = timer()
    # fetcher.fetch_dataset(50, 50, 50, num_processes=16)
    # end = timer()
    # print('time(s) = ', end-start)
    train_info_path = os.path.join(dl_path, 'Kinetic700', 'train', 'train_df.csv')
    eval_info_path = os.path.join(dl_path, 'Kinetic700', 'eval', 'eval_df.csv')
    test_info_path = os.path.join(dl_path, 'Kinetic700', 'test', 'test_df.csv')
    target = os.path.join(dl_path, 'Kinetic700')

    chekcerkinetic = CheckerKinetic700(target, train_info_path, test_info_path, eval_info_path)
    chekcerkinetic.check()