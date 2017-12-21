import tensorflow as tf

from config.argument_loader import ArgumentLoader
from model.dnc_trainer import DNCTrainer
# Loading Arguments
al = ArgumentLoader()
FLAGS = al.get_configuration()

def main(args):
  tf.logging.set_verbosity(3)
  dt = DNCTrainer(FLAGS)
  dt.train_model()

if __name__ == "__main__":
  tf.app.run()
