import tensorflow as tf

from utils.manipulator import DatasetManipulator
from utils.pretrained_glove import GloveTrainer
from utils.argument_loader import ArgumentLoader

# Arguments
al = ArgumentLoader()
FLAGS = al.get_configuration()

def main(args):
  tf.logging.set_verbosity(3)
  # Glove word embeddings model
  gt = GloveTrainer(vector_size=FLAGS.word_dimension, glove_dir=FLAGS.glove_dir)
  word_embeddings = gt.generate_word_embeddings()
  # Dataset manipulator
  dm = DatasetManipulator(FLAGS.dataset_pos,FLAGS.dataset_neg)
  tweets = dm.generate_dataset(total_samples=FLAGS.total_samples)
  tweets_glove = gt.manipulate_dataset(tweets.copy(), word_embeddings)
  train, test = dm.split_and_shuffle(tweets_glove, ratio=FLAGS.ratio, seed=FLAGS.seed)
  print(train[0][0][0])

if __name__ == "__main__":
  tf.app.run()
