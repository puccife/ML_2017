from manipulator import DatasetManipulator
import numpy as np

seed = 33
neg_url = 'twitter-datasets/train_neg.txt'
pos_url = 'twitter-datasets/train_pos.txt'
dm = DatasetManipulator(pos_url,neg_url)
dm.generate_dataset(total_samples=10000)
tweets = dm.get_tweets()
train, test = dm.split_and_shuffle(tweets, ratio=0.9, seed=seed)
