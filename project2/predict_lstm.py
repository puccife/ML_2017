import tensorflow as tf

from config.argument_loader_lstm import ArgumentLoaderLstm
from model.lstm_trainer import LSTMTrainer

# Arguments
al = ArgumentLoaderLstm()
FLAGS = al.get_configuration()

def main(args):
  tf.logging.set_verbosity(3)
  lstm = LSTMTrainer(FLAGS)
  lstm.test_model()

if __name__ == "__main__":
  tf.app.run()
