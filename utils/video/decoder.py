import tensorflow as tf

def decode_video(encoded_video_tensor):
    vid = []
    for frame in encoded_video_tensor:
        vid.append(tf.io.decode_jpeg(frame, channels=3))
    return tf.stack(vid)