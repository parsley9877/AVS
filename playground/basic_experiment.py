import os
import subprocess
import urllib
import pandas as pd
import numpy as np
import os

def dl_video(id, path, start, end):
    start = str(int(float(start)))
    end = str(int(float(end)))
    url = 'https://www.youtube.com/embed/' + id + '?start=' + start + '&end=' + end
    print(url)
    data_path = os.path.join(path, 'audioset')
    data_file_path = os.path.join(data_path, id)
    subprocess.run(['youtube-dl', url, '-o', data_file_path])
    convert_format(path)
    'ffmpeg -i input.mp4 -ss 01:19:27 -to 02:18:51 -c:v copy -c:a copy output.mp4'
    'ffmpeg -ss 00:10:45 -i input.avi -c:v libx264 -crf 18 -to 00:11:45 -c:a copy output.mp4'
    subprocess.run(['ffmpeg','-i', data_file_path + '.mp4', '-ss', start, '-to', end , '-c:v', 'libx264', '-crf', '18', '-c:a', 'copy', data_file_path + '(t).mp4'])
    if os.path.exists(data_file_path + '.mp4'):
        os.remove(data_file_path + '.mp4')
    else:
        print(url + '----- Video Not Found!!')
def convert_format(path):
    data_path = os.path.join(path, 'audioset')
    bad_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))
             and not (f.startswith('.')
             or f.endswith('mp4'))]
    for file in bad_files:
        path_to_bad_file = os.path.join(data_path, file)
        path_to_good_file = os.path.join(data_path, file[0:file.find('.')] + '.mp4')
        print(path_to_good_file)
        subprocess.run(['ffmpeg', '-i', path_to_bad_file, path_to_good_file])
        os.remove(os.path.join(data_path, file))




def get_df(path, classes: list):

    def map(s):
        q = s.replace('"', '')
        return q.replace(' ', '')

    df = pd.read_csv(path, sep='\n', header=2)
    df = df.applymap(map)
    df = df.iloc[:, 0].str.split(',', expand=True)
    df = df[df.isin(classes).any(axis=1)]
    df = df.iloc[:, 0:3].reset_index().drop('index', axis=1)

    return df


train_path = './datasets/datasets/AudioSetCSV/balanced_train_segments.csv'
test_path = './datasets/datasets/AudioSetCSV/eval_segments.csv'
classes = ['/m/09x0r', "/m/05zppz", "/m/02zsn", "/m/0ytgt", "/m/01h8n0", "/m/02qldy", "/m/0261r1", "/m/0brhx"]

train_df = get_df(train_path, ['/m/09x0r', "/m/05zppz", "/m/02zsn", "/m/0ytgt", "/m/01h8n0", "/m/02qldy", "/m/0261r1", "/m/0brhx"])
test_df = get_df(test_path, ['/m/09x0r', "/m/05zppz", "/m/02zsn", "/m/0ytgt", "/m/01h8n0", "/m/02qldy", "/m/0261r1", "/m/0brhx"])

dl_path = './datasets/datasets/'

# convert_format(dl_path)

for idx, row in train_df.iterrows():
    if idx>10:
        break
    dl_video(row[0], dl_path, row[1], row[2])

