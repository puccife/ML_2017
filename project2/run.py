from utils.manipulator import DatasetManipulator
from utils.pretrained_glove import GloveTrainer

gt = GloveTrainer(vector_size=50)
word_embeddings = gt.generate_word_embeddings()

seed = 333
neg_url = 'twitter-datasets/train_neg.txt'
pos_url = 'twitter-datasets/train_pos.txt'
dm = DatasetManipulator(pos_url,neg_url)
tweets = dm.generate_dataset(total_samples=2)
tweets_glove = gt.manipulate_dataset(tweets.copy(), word_embeddings)
train, test = dm.split_and_shuffle(tweets_glove, ratio=0.9, seed=seed)
