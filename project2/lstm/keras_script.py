########################################
# import packages
########################################
import os
import re
import codecs
import numpy as np
import pandas as pd

from string import punctuation
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Merge
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

########################################
# set directories and parameters
########################################
DATA_DIR = 'data\\'
GLOVE_DIR = 'glove\\'
EMBEDDING_FILE = GLOVE_DIR + 'glove.twitter.27B.200d.txt'
TRAIN_DATA_FILE_POS = DATA_DIR + 'train_pos_full.txt'
TRAIN_DATA_FILE_NEG = DATA_DIR + 'train_neg_full.txt'
TEST_DATA_FILE = DATA_DIR + 'test_data.txt'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

#######################################
# index word vectors
#######################################
print('Indexing word vectors')

embeddings_index = {}
f = open(EMBEDDING_FILE)
count = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %d word vectors of glove.' % len(embeddings_index))

########################################
# process texts in datasets
########################################
print('Processing text dataset')


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"<user>", "", text)
    text = re.sub(r"<url>", "", text)
    text = re.sub(r"plz", "please", text)
    text = re.sub(r"dat", "that", text)
    text = re.sub(r"bc", "because", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return text


tweets_pos = [text_to_wordlist(line.rstrip('\n')) for line in open(TRAIN_DATA_FILE_POS)]
tweets_neg = [text_to_wordlist(line.rstrip('\n')) for line in open(TRAIN_DATA_FILE_NEG)]
tweets_train = tweets_pos + tweets_neg
labels = []
for i in range(0, len(tweets_train)):
    if i > len(tweets_pos) / 2:
        labels.append(0)
        continue
    labels.append(1)

print('Number of train tweets: %d' % len(tweets_train))
print('Number of labels: %d' % len(labels))

# tweets_test = []
# tweets_test_ids = []
# for line in open(TEST_DATA_FILE):
#     temp = line.split(',')
#     tweets_test_ids.append(temp[0])
#     temp = temp.pop(0)
#     temp = text_to_wordlist(" ".join(temp))
#     tweets_test.append(temp)
# print('Number of test tweets: %d' % len(tweets_test))

print('Initializing Tokenizer')
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(tweets_train)
# tokenizer.fit_on_texts(tweets_test)

sequences = tokenizer.texts_to_sequences(tweets_train)
# test_sequences = tokenizer.texts_to_sequences(tweets_test)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.array(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# test_ids = np.array(tweets_test_ids)

#######################################
# prepare embeddings
#######################################
print('Preparing embedding matrix')

num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

# ########################################
# #validation data
# ########################################
print('Preparing validation data')

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print (y_train.shape)
print (y_val.shape)

#######################################
# define the model structure
#######################################

print('Creating and fitting model')

model = Sequential()
model.add(Embedding(num_words,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=512, nb_epoch=50,
          verbose=1, shuffle=True, callbacks=[early_stopping])
model.save('models' + '\\' + 'lstm_128_dropout_0-5_biggest_datasetT&G.h5', overwrite=True)
model.save_weights('models' + '\\' + 'lstm_128_dropout_0-5_biggest_datasetT&G_weights.h5')


###########################################
# test the model and make submission file
###########################################

def test_model(model_name, test_data, test_ids):
    print('Making the submission file')

    model = load_model('models' + '\\' + model_name + '.h5')
    model.load_weights('models' + '\\' + model_name + '_weights.h5')
    preds = model.predict(test_data, batch_size=1024, verbose=1)

    temp = preds.ravel()
    temp_binary = []
    for i in temp:
        if i < 0.5:
            temp_binary.append(-1)
        else:
            temp_binary.append(1)
    temp_binary = np.array(temp_binary)

    submission = pd.DataFrame({'Id': test_ids, 'Prediction': temp_binary})
    submission.to_csv('submissions\\' + model_name + '.csv', sep=',', index=False)

# test_model('lstm_128_dropout_0-5_biggest_datasetT&G')
