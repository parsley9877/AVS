from absl import flags
from absl import app
from absl import logging

from test_experiments.MFCC_test.main import main

FLAGS = flags.FLAGS


def exec(argv):
    del argv
    main()

if __name__ == '__main__':
    app.run(exec)