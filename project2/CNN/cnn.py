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

def load_data_and_labels():

    # Load data from files
    positive_examples = list(open("./twitter-datasets/train_pos.txt").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./twitter-datasets/train_neg.txt").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_tweets(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def pad_sentences(sentences, padding_word="padding_word"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = 20
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if num_padding < 0:
            new_sentence = sentence[:sequence_length]
        else:
            new_sentence = sentence + [padding_word] * num_padding

        padded_sentences.append(new_sentence)
    return padded_sentences

def generate_word_embeddings():
    embeddings_index = {}
    print('Indexing word vectors.')
    f = open(os.path.join('./glove.twitter.27B/', 'glove.twitter.27B.'+str(200)+'d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def manipulate_dataset(dataset,word_embeddings):
    missing_voc={}
    output_array = np.ndarray((len(dataset),20,200))
    for i,sentence in enumerate(dataset):
        matrix_embedding = []
        for word in sentence:
            try:
                matrix_embedding.append(word_embeddings[word])
            except:
                vector = create_vector(word,word_embeddings,200,silent=True)
                matrix_embedding.append(vector)
                try:
                    missing_voc[word] = missing_voc[word] + 1
                except KeyError:
                    missing_voc[word] = 1
        output_array[i]=(matrix_embedding)
    return output_array

def create_vector(word, word_embeddings, word_vector_size, silent=True):
    # if the word is missing from Glove or Google Vectors, create some fake vector and store in glove!
    vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
    word_embeddings[word] = vector
    if not silent:
        print("utils.py::create_vector => %s is missing" % word)
    return vector

def split_and_shuffle(x,y, ratio, seed):
    split_index = int(len(x)*ratio)
    train_x, test_x = x[:split_index], x[split_index:]
    train_y, test_y = y[:split_index], y[split_index:]
    np.random.seed(seed)
    np.random.shuffle(train_x)
    np.random.seed(seed)
    np.random.shuffle(test_x)
    np.random.seed(seed)
    np.random.shuffle(train_y)
    np.random.seed(seed)
    np.random.shuffle(test_y)
    return train_x,test_x,train_y,test_y
