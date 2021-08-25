import os


PATH_TO_PROJECT = os.getcwd()
PATH_TO_PERSISTENT_STORAGE = os.path.join(PATH_TO_PROJECT, 'datasets', 'datasets')
PATH_TO_LOCAL_DRIVE = os.path.join(PATH_TO_PERSISTENT_STORAGE, 'local')
PATH_TO_KINETICS_DATASET = os.path.join(PATH_TO_PERSISTENT_STORAGE, 'kinetics700_2020')
PATH_TO_CHECKPOINT = os.path.join(PATH_TO_PERSISTENT_STORAGE, 'avs_checkpoint')
PATH_TO_STATE_CHECKPOINT = os.path.join(PATH_TO_CHECKPOINT, 'state_checkpoint')
PATH_TO_OTHER_CHECKPOINT = os.path.join(PATH_TO_CHECKPOINT, 'other_checkpoint')
PATH_TO_SAMPLE_CHECKPOINT = os.path.join(PATH_TO_PERSISTENT_STORAGE, 'sample_checkpoint')


