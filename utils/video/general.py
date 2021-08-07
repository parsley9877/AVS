import cv2
import numpy as np

def get_frames(path_to_mp4):
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
    # cv2.destroyAllWindows()
    return video_numpy, fps, num_frames, height, width

def save_video_from_frames(frames, fps, path):
    # frames = frames.astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (frames.shape[1], frames.shape[2]))
    for i in range(frames.shape[0]):
        out.write(frames[i])
    out.release()