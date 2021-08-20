import os
from moviepy.editor import VideoFileClip
import cv2
import shutil
import sys
import json

sys.path.append(os.path.join(os.getcwd(), 'configs'))
from global_namespace import PATH_TO_PROJECT, PATH_TO_PERSISTENT_STORAGE,\
    PATH_TO_LOCAL_DRIVE, PATH_TO_KINETICS_DATASET


def shifted_croped_filter(path):
    listOfFiles = get_list_of_mp4(path)
    for file in listOfFiles:
        if os.path.basename(file).startswith('('):
            os.remove(file)
def non_mp4_checker(path):
    listOfFiles = get_list_of_all_files(path)
    if False in [file.endswith('.mp4') for file in listOfFiles]:
        return True
    return False
def non_mp4_finder(path):
    out = []
    listOfFiles = get_list_of_all_files(path)
    for elem in listOfFiles:
        if not elem.endswith('.mp4'):
            out.append(elem)
    return out
def bad_duration_finder(path, low, high):
    out = []
    listOfFiles = get_list_of_mp4(path)
    for elem in listOfFiles:
        try:
            clip = VideoFileClip(elem)
            if clip.duration < low or clip.duration > high:
                out.append(elem)
            clip.close()
        except:
            pass
    return out
def bad_duration_filter(path, low, high):
    listOfFiles = get_list_of_mp4(path)
    for elem in listOfFiles:
        try:
            clip = VideoFileClip(elem)
            if clip.duration < low or clip.duration > high:
                os.remove(elem)
            clip.close()
        except:
            pass
def list_of_all_durations(path):
    listOfFiles = get_list_of_mp4(path)
    o = []
    for elem in listOfFiles:
        try:
            clip = VideoFileClip(elem)
            o.append(clip.duration)
            clip.close()
        except:
            pass
    return o
def list_of_all_fps(path):
    listOfFiles = get_list_of_mp4(path)
    o = []
    for elem in listOfFiles:
        try:
            clip = VideoFileClip(elem)
            o.append(clip.fps)
            clip.close()
        except:
            pass
    return o
def list_of_all_sizes(path):
    listOfFiles = get_list_of_mp4(path)
    o = []
    for elem in listOfFiles:
        try:
            clip = VideoFileClip(elem)
            o.append(clip.size)
            clip.close()
        except:
            pass
    return o
def list_of_all_nf(path):
    listOfFiles = get_list_of_mp4(path)
    o = []
    for elem in listOfFiles:
        try:
            clip = VideoFileClip(elem)
            cap = cv2.VideoCapture(elem)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            o.append(num_frames)
            cap.release()
            clip.close()
        except:
            pass
    return o
def bad_mp4_finder(path):
    out = []
    listOfFiles = get_list_of_mp4(path)
    for elem in listOfFiles:
        try:
            clip = VideoFileClip(elem)
            if clip.duration == 0 or clip.fps == 0:
                out.append(elem)
            clip.close()
        except:
            pass
    return out
def bad_mp4_filter(path):
    listOfFiles = get_list_of_mp4(path)
    for elem in listOfFiles:
        try:
            clip = VideoFileClip(elem)
            if clip.duration == 0 or clip.fps == 0:
                os.remove(elem)
            clip.close()
        except:
            pass
def get_list_of_mp4(path):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames if file.endswith('.mp4')]
    return listOfFiles
def get_list_of_all_files(path):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles
def corrupted_mp4_finder(path):
    out = []
    listOfFiles = get_list_of_mp4(path)
    for file in listOfFiles:
        try:
            clip = VideoFileClip(file)
            clip.close()
        except:
            out.append(file)
    return out

def if_exists_delete(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

def check_fetched(dataset_name):
    if os.path.exists(os.path.join(PATH_TO_PERSISTENT_STORAGE, dataset_name)):
        return True
    else:
        return False

def check_shifted(dataset_name):
    if os.path.exists(os.path.join(PATH_TO_PERSISTENT_STORAGE, dataset_name + '_exp')):
        return True
    else:
        return False

def log_fetched_dataset(dataset_name, lower_bound_duration=None, upper_bound_duration=None):
    path_to_dataset = os.path.join(PATH_TO_PERSISTENT_STORAGE, dataset_name)
    path_to_failed_files = os.path.join(path_to_dataset, 'info.txt')
    info = {}
    all_mp4s = get_list_of_mp4(path_to_dataset)
    non_mp4_files = non_mp4_finder(path_to_dataset)
    if lower_bound_duration is not None:
        bad_duration_files = bad_duration_finder(path_to_dataset, lower_bound_duration, upper_bound_duration)
    else:
        bad_duration_files = [None]
    all_durations = list_of_all_durations(path_to_dataset)
    all_fpss = list_of_all_fps(path_to_dataset)
    all_sizes = list_of_all_sizes(path_to_dataset)
    all_nfs = list_of_all_nf(path_to_dataset)
    corrupted_mp4s = corrupted_mp4_finder(path_to_dataset)
    with open(path_to_failed_files) as handler:
        failed_files = handler.readlines()
    info['all_durations'] = all_durations
    info['bad_duration_files'] = bad_duration_files
    info['non_mp4_files'] = non_mp4_files
    info['all_fpss'] = all_fpss
    info['all_sizes'] = all_sizes
    info['all_nfs'] = all_nfs
    info['corrupted_mp4s'] = corrupted_mp4s
    info['failed_files'] = failed_files
    info['all_mp4s'] = all_mp4s
    assert len(info['all_durations']) == len(info['all_fpss'])
    assert len(info['all_mp4s']) == len(info['all_fpss']) + len(info['corrupted_mp4s'])
    return info

def filter_fetched_data(exp_config, path):
    listOfFiles = get_list_of_mp4(path)
    cnts = [0, 0, 0]
    for elem in listOfFiles:
        if not elem.endswith('(t).mp4'):
            print('Not trimmed: ', elem)
            if 'train' in elem:
                cnts[0] += 1
            elif 'eval' in elem:
                cnts[1] += 1
            elif 'test' in elem:
                cnts[2] += 1
            print(cnts)
            os.remove(elem)
        try:
            clip = VideoFileClip(elem)
            if clip.duration < exp_config['dataset']['min_duration'] or clip.fps < exp_config['dataset']['min_fps']:
                print('Bad duration or fps (Deleted):',  elem)
                if 'train' in elem:
                    cnts[0] += 1
                elif 'eval' in elem:
                    cnts[1] += 1
                elif 'test' in elem:
                    cnts[2] += 1
                print(cnts)
                os.remove(elem)
            clip.close()
        except:
            print('Corrupted(Deleted):', elem)
            if 'train' in elem:
                cnts[0] += 1
            elif 'eval' in elem:
                cnts[1] += 1
            elif 'test' in elem:
                cnts[2] += 1
            print(cnts)
            os.remove(elem)

def number_of_examples(path):
    listOfFiles = get_list_of_mp4(path)
    return len(listOfFiles)