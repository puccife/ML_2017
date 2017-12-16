import tensorflow as tf

from utils.argument_loader_cnn import ArgumentLoader_cnn
from model.cnn_trainer import CNNTrainer

# Arguments
al_cnn = ArgumentLoader_cnn()
FLAGS = al_cnn.get_configuration()

def main(args):
  tf.logging.set_verbosity(3)
  cnn = CNNTrainer(FLAGS)

if __name__ == "__main__":
  tf.app.run()
