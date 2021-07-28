import os
from moviepy.editor import VideoFileClip


def shifted_filter(path):
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
        if False in [(f.endswith('.txt') or f.endswith('.tfrecords') or f.endswith('.csv')) for f in filenames]:
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