
import json
import sys
import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow import keras
import tracemalloc
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), 'utils', 'sys'))
sys.path.append(os.path.join(os.getcwd(), 'configs'))
sys.path.append(os.path.join(os.getcwd(), 'utils'))
sys.path.append(os.path.join(os.getcwd(), 'models'))
sys.path.append(os.path.join(os.getcwd(), 'dataset_utils'))
from os_tools import check_fetched, log_fetched_dataset, PATH_TO_PERSISTENT_STORAGE,\
    check_shifted, if_exists_delete, filter_fetched_data, number_of_examples
from global_namespace import PATH_TO_STATE_CHECKPOINT, PATH_TO_OTHER_CHECKPOINT, PATH_TO_CHECKPOINT
from loader import VideoSetShifter, Video2TFRecord, TFRecord2Video, temp
from abstract_classes import R3D, PlainModel
from audio.preprocessing import MFCC
from video.preprocessing import BasicPreprocessing


def mfcc_audio(x):
    specs = {
        "samplerate": 44100,
        "winlen": 0.023,
        "winstep": 0.05*4/3.1,
        "numcep": 24,
        "nfilt": 24,
        "nfft": 1024,
        "lowfreq": 0,
        "highfreq": 8000,
        "preemph": 0.97,
        "ceplifter": 22,
        "appendEnergy": True,
        "winfunc": np.hamming
    }
    mfcc = MFCC(specs, 2)
    x['audio'] = tf.numpy_function(mfcc.process, [x['audio']], tf.float32)
    return x


def audio_dim_matching(x):
    x['audio'] = x['audio'][0:178605]
    return x


def video_dim_matching(x):
    x['vid'] = x['vid'][0:80]
    return x


def loss_fn(v_o, a_o, mis, weights_, sum_of_batch_weights_):
    loss_ = tf.divide(
        tf.reduce_sum(
            tf.multiply(
                tf.square(tf.reduce_sum(tf.square(v_o - a_o), axis=1) - tf.abs(mis)),
                weights_
            )
        ),
        sum_of_batch_weights_
    )
    return loss_


def convert_label_from_mis(misalignment):
    return tf.where(misalignment == 0, tf.zeros_like(misalignment), tf.ones_like(misalignment))


def convert_label_from_out(v_o, a_o, threshold):
    outputs = tf.reduce_sum(tf.square(v_o - a_o), axis=1)
    return tf.where(outputs < threshold, tf.zeros_like(outputs), tf.ones_like(outputs))


def train_inference(batch, sum_of_batch_weights_, threshold, mode):
    v_o, a_o = model([batch['vid'], batch['audio']])
    if mode == 'train':
        train_acc_metric.update_state(
            convert_label_from_out(v_o, a_o, threshold),
            convert_label_from_mis(batch['misalignment']),
            sample_weight=batch['example_weight']
        )
    else:
        eval_acc_metric.update_state(
            convert_label_from_out(v_o, a_o, threshold),
            convert_label_from_mis(batch['misalignment']),
            sample_weight=batch['example_weight']
        )
    loss_ = loss_fn(v_o, a_o, batch['misalignment'], batch['example_weight'], sum_of_batch_weights_)
    return loss_


def train_step(batch, sum_of_batch_weights_):
    with tf.GradientTape() as tape:
        v_o, a_o = model([batch['vid'], batch['audio']], training=True)
        loss_ = loss_fn(v_o, a_o, batch['misalignment'], batch['example_weight'], sum_of_batch_weights_)
    gradients = tape.gradient(loss_, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_


def get_sum_over_replicas_and_batches(shared_object):
    per_replica_values_ = mirrored_strategy.experimental_local_results(shared_object)
    sum_of_batch_weights_ = sum([tf.reduce_sum(per_replica_values_[i], axis=0) for i in range(len(per_replica_values_))])
    return sum_of_batch_weights_


@tf.function
def distributed_train_step(batch):
    batch_wise_sum = get_sum_over_replicas_and_batches(record['example_weight'])
    per_replica_losses = mirrored_strategy.run(train_step, args=(batch, batch_wise_sum))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=None)


@tf.function
def distributed_train_inference(batch, threshold, mode):
    batch_wise_sum = get_sum_over_replicas_and_batches(record['example_weight'])
    per_replica_losses = mirrored_strategy.run(train_inference, args=(batch, batch_wise_sum, threshold, mode))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=None)


def cosine_annealing(epoch, lr_min, lr_max, t):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos((epoch/t) * np.pi))


def step_decay(epoch, epochs_drop, drop, lr0):
    return lr0 * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
##################################################


"""
loading experiment config:
"""
config_path = './configs/exp1.json'
with open(config_path) as handler:
    exp_config = json.load(handler)
fetched_ds_path = os.path.join(PATH_TO_PERSISTENT_STORAGE, exp_config['dataset']['name'])
##################################################


"""
setting experiment setup
"""
global_batch_size = exp_config['setup']['global_batch_size']
replicas = exp_config['setup']['replicas']
epochs = exp_config['setup']['epochs']
use_weight_checkpoint = exp_config['setup']['use_weight_checkpoint']
use_state_checkpoint = exp_config['setup']['use_state_checkpoint']
use_metric_checkpoint = exp_config['setup']['use_metric_checkpoint']
video_model_base_class, video_model_config = exp_config['model']['video']
audio_model_base_class, audio_model_config = exp_config['model']['audio']
optimizer_type = exp_config['optimizer']['type']
optimizer_config = exp_config['optimizer']['config']
lr_policy = exp_config['optimizer']['lr_policy']['name']
lr_policy_configs = exp_config['optimizer']['lr_policy']['configs']
detection_threshold = exp_config['inference_config']['threshold']

with open(video_model_config) as json_obj:
    video_model_config = json.load(json_obj)
with open(audio_model_config) as json_obj:
    audio_model_config = json.load(json_obj)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices([gpus[index] for index in replicas], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

##################################################


"""
training
"""
mirrored_strategy = tf.distribute.MirroredStrategy(['GPU:' + str(i) for i in replicas])
decoder = TFRecord2Video()

with mirrored_strategy.scope():
    video_input_layer = keras.Input(shape=(80, 128, 128, 3))
    audio_input_layer = keras.Input(shape=(24, 64, 3))
    # video_preprocessing = BasicPreprocessing('type2', 3, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))(video_input_layer)
    video_preprocessing = keras.Sequential([
        Rescaling(scale=1./127.5, offset=-1),
    ])(video_input_layer)
    video_net = eval(video_model_base_class)(video_model_config)(video_preprocessing)
    audio_net = eval(audio_model_base_class)(audio_model_config)(audio_input_layer)
    model = keras.Model(
        inputs=[video_input_layer, audio_input_layer],
        outputs=[video_net, audio_net],
    )
    train_acc_metric = tf.keras.metrics.Accuracy()
    eval_acc_metric = tf.keras.metrics.Accuracy()
    optimizer = eval(optimizer_type)(**optimizer_config)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

manager = tf.train.CheckpointManager(
    checkpoint, PATH_TO_STATE_CHECKPOINT, 3, keep_checkpoint_every_n_hours=None,
    checkpoint_name='ckpt', step_counter=None, checkpoint_interval=None,
    init_fn=None
)

##################################################

"""
checkpoint stuff
"""
model.summary()
optimizer.lr.assign(12.0)
if os.path.exists(PATH_TO_CHECKPOINT):
    print('checkpoint found!')
    if use_weight_checkpoint:
        print('using weights checkpoint...')
        with open(os.path.join(PATH_TO_OTHER_CHECKPOINT, 'weights.json'), 'r') as file_obj:
            weights = json.load(file_obj)
            sum_of_weights_train = weights['train']
            sum_of_weights_eval = weights['eval']
    else:
        print('not using weights checkpoint...')
        sum_of_weights_train = 0
        sum_of_weights_eval = 0
    if use_metric_checkpoint:
        print('using metric checkpoint...')
        with open(os.path.join(PATH_TO_OTHER_CHECKPOINT, 'metrics.json'), 'r') as file_obj:
            metrics = json.load(file_obj)
            train_losses = metrics['train']['loss']
            train_accs = metrics['train']['acc']
            eval_losses = metrics['eval']['loss']
            eval_accs = metrics['eval']['acc']
            saved_epochs = metrics['train']['saved_epochs']
    else:
        train_losses = []
        train_accs = []
        eval_losses = []
        eval_accs = []
        saved_epochs = 0
    if use_state_checkpoint:
        print('using state checkpoint...')
        checkpoint.restore(manager.latest_checkpoint)
        model.summary()
else:
    assert not(use_weight_checkpoint or use_metric_checkpoint or use_state_checkpoint)
    print('checkpoint directory not found')
    train_losses = []
    train_accs = []
    eval_losses = []
    eval_accs = []
    saved_epochs = 0
    sum_of_weights_train = 0
    sum_of_weights_eval = 0
    os.makedirs(PATH_TO_STATE_CHECKPOINT)
    os.mkdir(PATH_TO_OTHER_CHECKPOINT)
##################################################

train_record_path = os.path.join(PATH_TO_PERSISTENT_STORAGE, 'Kinetic700_source',
                                 'Kinetic700_source_tfrecords', 'train')
eval_record_path = os.path.join(PATH_TO_PERSISTENT_STORAGE, 'Kinetic700_source',
                                'Kinetic700_source_tfrecords', 'eval')
list_of_train_records = [os.path.join(train_record_path, record) for record in
                         os.listdir(train_record_path) if record.endswith('.tfrecords')]
list_of_eval_records = [os.path.join(eval_record_path, record) for record in
                        os.listdir(eval_record_path) if record.endswith('.tfrecords')]
training_set_cardinality = len(list_of_train_records) * exp_config['dataset']['records_per_file']
eval_set_cardinality = len(list_of_eval_records) * exp_config['dataset']['records_per_file']
train_dataset = decoder.get_records(list_of_train_records)\
    .map(video_dim_matching).map(audio_dim_matching)\
    .map(mfcc_audio).batch(global_batch_size, drop_remainder=True)
dist_train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
eval_dataset = decoder.get_records(list_of_eval_records)\
    .map(video_dim_matching).map(audio_dim_matching)\
    .map(mfcc_audio).batch(global_batch_size, drop_remainder=True)
dist_eval_dataset = mirrored_strategy.experimental_distribute_dataset(eval_dataset)

num_batches_train = training_set_cardinality // global_batch_size
num_batches_eval = eval_set_cardinality // global_batch_size

checkpoint_prefix = os.path.join(PATH_TO_STATE_CHECKPOINT, 'checkpoint')

# WEIGHT ESTIMATION & CHECKPOINT
if not use_weight_checkpoint:
    with tqdm(total=num_batches_train, colour='#00ff00', desc='training weight progress') as pbar_weights:
        for idx, record in enumerate(dist_train_dataset):
            batch_sum_of_weights = 0
            per_replica_values = mirrored_strategy.experimental_local_results(
                record['example_weight'])
            for share in per_replica_values:
                batch_sum_of_weights += tf.reduce_sum(share, axis=0).numpy()

            sum_of_weights_train += batch_sum_of_weights
            pbar_weights.update(1)

    with tqdm(total=num_batches_eval, colour='#00ff00', desc='eval weight progress') as pbar_weights_eval:
        for idx, record in enumerate(dist_eval_dataset):
            batch_sum_of_weights = 0
            per_replica_values = mirrored_strategy.experimental_local_results(
                record['example_weight'])
            for share in per_replica_values:
                batch_sum_of_weights += tf.reduce_sum(share, axis=0).numpy()

            sum_of_weights_eval += batch_sum_of_weights
            pbar_weights_eval.update(1)

    weights_to_be_saved = {
        'train': sum_of_weights_train,
        'eval': sum_of_weights_eval
    }

    with open(os.path.join(PATH_TO_OTHER_CHECKPOINT, 'weights.json'), 'w') as file_obj:
        json.dump(weights_to_be_saved, file_obj, sort_keys=True, indent=4)

print('sum of all eval weights: {}'.format(sum_of_weights_eval))
print('sum of all train weights: {}'.format(sum_of_weights_train))

# TRAINING & INFERENCE & CHECKPOINT
for i in range(saved_epochs, epochs):
    current_lr = eval(lr_policy)(i, **lr_policy_configs)
    optimizer.lr.assign(current_lr)
    train_loss_epoch = 0
    eval_loss_epoch = 0
    train_acc_metric.reset_states()
    eval_acc_metric.reset_states()

    # TRAINING INFERENCE
    with tqdm(total=num_batches_train, colour='#00ff00', desc='train inference {}'.format(current_lr)) as pbar_loss:
        for idx, record in enumerate(dist_train_dataset):
            batch_loss = distributed_train_inference(record, detection_threshold, mode='train')
            per_replica_values = mirrored_strategy.experimental_local_results(
                record['example_weight'])
            sum_of_batch_weights = sum(
                [tf.reduce_sum(per_replica_values[i], axis=0) for i in range(len(per_replica_values))])
            train_loss_epoch += (batch_loss * sum_of_batch_weights).numpy()
            pbar_loss.update(1)
    train_losses.append(train_loss_epoch/sum_of_weights_train)
    train_accs.append(train_acc_metric.result().numpy())
    print('epoch {}: training loss: {}'.format(i, train_loss_epoch/sum_of_weights_train))
    print('epoch {}: training accuracy: {}'.format(i, train_acc_metric.result().numpy()))

    # EVAL INFERENCE
    with tqdm(total=num_batches_eval, colour='#00ff00', desc='eval inference {}'.format(current_lr)) as pbar_loss:
        for idx, record in enumerate(dist_eval_dataset):
            batch_loss = distributed_train_inference(record, detection_threshold, mode='eval')
            per_replica_values = mirrored_strategy.experimental_local_results(
                record['example_weight'])
            sum_of_batch_weights = sum(
                [tf.reduce_sum(per_replica_values[i], axis=0) for i in range(len(per_replica_values))])
            eval_loss_epoch += (batch_loss * sum_of_batch_weights).numpy()
            pbar_loss.update(1)
    eval_losses.append(eval_loss_epoch / sum_of_weights_eval)
    eval_accs.append(eval_acc_metric.result().numpy())
    print('epoch {}: eval loss: {}'.format(i, eval_loss_epoch / sum_of_weights_eval))
    print('epoch {}: eval accuracy: {}'.format(i, eval_acc_metric.result().numpy()))

    # CHECKPOINT
    manager.save()
    metrics = {
        "train": {
            "acc": [np.float64(x) for x in train_accs],
            "loss": [np.float64(x) for x in train_losses],
            "saved_epochs": i
        },
        "eval": {
            "acc": [np.float64(x) for x in eval_accs],
            "loss": [np.float64(x) for x in eval_losses]
        }
    }
    with open(os.path.join(PATH_TO_OTHER_CHECKPOINT, 'metrics.json'), 'w') as file_obj:
        json.dump(metrics, file_obj, sort_keys=True, indent=4)

    # TRAINING
    with tqdm(total=num_batches_train, colour='#00ff00', desc='batch training progress') as pbar_train:
        for idx, record in enumerate(dist_train_dataset):
            loss = distributed_train_step(record)
            pbar_train.update(1)
