import tensorflow as tf

from utils.manipulator import DatasetManipulator
from utils.pretrained_glove import GloveTrainer
# paths to twitter dataset
NEG_URL = 'twitter-datasets/train_neg.txt'
POS_URL = 'twitter-datasets/train_pos.txt'

# hyperparameters
SEED = 333
RATIO = 0.8
TOTAL_SAMPLES = 1000
VECTOR_SIZE = 25

def main(unused_argv):
  tf.logging.set_verbosity(3)
  # Glove word embeddings model
  gt = GloveTrainer(vector_size=VECTOR_SIZE)
  word_embeddings = gt.generate_word_embeddings()
  # Dataset manipulator
  dm = DatasetManipulator(POS_URL,NEG_URL)
  tweets = dm.generate_dataset(total_samples=TOTAL_SAMPLES)
  tweets_glove = gt.manipulate_dataset(tweets.copy(), word_embeddings)
  train, test = dm.split_and_shuffle(tweets_glove, ratio=RATIO, seed=SEED)

if __name__ == "__main__":
  tf.app.run()
