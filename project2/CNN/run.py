from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model
from keras.layers.merge import Concatenate
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
import re
import itertools
from collections import Counter
from utils.preprocessing import clean_tweets
import os

from CNN.cnn import load_data_and_labels,pad_sentences,generate_word_embeddings,manipulate_dataset,create_vector,split_and_shuffle

sentences, labels = load_data_and_labels()
sentences_padded = pad_sentences(sentences)
vocabulary, vocabulary_inv_list = build_vocab(sentences_padded)
x = sentences_padded
y = np.array(labels)
#vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
y = y.argmax(axis=1)

x_train,x_test,y_train,y_test = split_and_shuffle(x,y,0.9,33)

print("x_train shape:", len(x_train))
print("x_test shape:", len(x_test))
print("y_train shape:", len(y_train))
print("y_test shape:", len(y_test))
#print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

embeddings_words = generate_word_embeddings()

x_train_embeddings,missing_words_train = manipulate_dataset(x_train,embeddings_words)
x_test_embeddings,missing_words_test = manipulate_dataset(x_test,embeddings_words)
