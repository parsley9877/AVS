import cv2

def disp_video(frames, fps):
    for i in range(frames.shape[0]):
        cv2.imshow('video sample', frames[i])
        cv2.waitKey(int((1 / fps) * 1000))