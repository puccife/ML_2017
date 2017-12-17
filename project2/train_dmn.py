import tensorflow as tf

from utils.argument_loader import ArgumentLoader
#from model.dnc_trainer import DNCTrainer
from model.dmn_trainer import DMNTrainer
# Arguments
al = ArgumentLoader()
FLAGS = al.get_configuration()

def main(args):
  tf.logging.set_verbosity(3)
  dm = DMNTrainer(FLAGS)
  dm.train_DMN()

if __name__ == "__main__":
  tf.app.run()
