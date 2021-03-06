import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import tensorflow as tf
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences


########################################
# set directories and parameters
########################################
DATA_DIR = 'data\\'
GLOVE_DIR = 'glove\\'
EMBEDDING_DIM = 200
MODEL_NAME = 'lstm_256_x2_dropout_0-25_biggest_datasetT&G_200d_adam.tflearn'
MODEL_PATH = os.path.join('models', MODEL_NAME)
EMBEDDING_FILE = GLOVE_DIR + 'glove.twitter.27B.' + str(EMBEDDING_DIM) + 'd.txt'
TRAIN_DATA_FILE_POS = DATA_DIR + 'train_pos_full.txt'
TRAIN_DATA_FILE_NEG = DATA_DIR + 'train_neg_full.txt'
TEST_DATA_FILE = DATA_DIR + 'test_data.txt'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
VALIDATION_SPLIT = 0.2


#######################################
# index word vectors
#######################################
# This part of the code is for getting the word vectors from the GloVe TXT file with pre-trained weight vectors
print('Indexing word vectors')

embeddings_index = {}
f = open(EMBEDDING_FILE, 'r', encoding='utf-8')
count = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs #
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
    text = re.sub(r"jk", "joke", text)
    text = re.sub(r"ya", "your", text)
    text = re.sub(r"thang", "thing", text)
    text = re.sub(r"dunno", "do not know", text)
    text = re.sub(r"doin", "doing", text)
    text = re.sub(r"lil", "little", text)
    text = re.sub(r"tmr", "tomorrow", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r">", "", text)
    text = re.sub(r"> >", " ", text)
    text = re.sub(r"[^A-Za-z0-9^,!./'+-=]", " ", text)
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
    text = re.sub(r"/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return text

# All of the train tweets are read from the data files and are being put in a list. Before that, they are cleaned by the text_to_wordlist function
tweets_pos = [text_to_wordlist(line.rstrip('\n')) for line in open(TRAIN_DATA_FILE_POS, 'r', encoding='utf-8')]
tweets_neg = [text_to_wordlist(line.rstrip('\n')) for line in open(TRAIN_DATA_FILE_NEG, 'r', encoding='utf-8')]
tweets_train = tweets_pos + tweets_neg
labels = np.ones(len(tweets_train), dtype=np.int8)
for i in range(int(len(labels)/2), len(labels)):  # this is for generating labels for the positive and negative tweets
    labels[i] = 0

print('Number of train tweets: %d' % len(tweets_train))
print('Number of labels: %d' % len(labels))

# Reading and "cleaning" of the test dataset
tweets_test = []
tweets_test_ids = []
for line in open(TEST_DATA_FILE, 'r', encoding='utf-8'):
    temp = line.split(',')  # we split the string, since the first element will be the id, and the rest is the whole tweet
    tweets_test_ids.append(temp[0])
    temp.pop(0)
    temp = text_to_wordlist(" ".join(temp))
    tweets_test.append(temp)

print('Number of test tweets: %d' % len(tweets_test))


######################################
# prepare tokenizer
######################################
print('Initializing Tokenizer')

# The tokenizer is fitted on both of the datasets and the maximum num words is being set manually
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(tweets_train + tweets_test)
# tokenizer.fit_on_texts(tweets_test)

sequences = tokenizer.texts_to_sequences(tweets_train)
test_sequences = tokenizer.texts_to_sequences(tweets_test)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # we are padding the sequences to the maximum length of 30
labels = to_categorical(np.array(labels), nb_classes=2)  # the labels are converted to binary matrix for the neural net, since we are using categorical_crossentropy
print('Shape of train_data tensor:', train_data.shape)
print('Shape of label tensor:', labels.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(tweets_test_ids)
print('Shape of test_data tensor:', test_data.shape)


######################################
# prepare embeddings
######################################
print('Preparing embedding matrix')
# We use the generated dictionary from the GloVe text file in order to get all of the word vectors
# from our word dictionary - word_index. This dictionary is generated by the Tokenizer from all of the possible train tweets.

num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= num_words:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


######################################
# validation data
######################################
print('Preparing validation data')
# In this part of the code we generate the validation data from the train set

indices = np.arange(train_data.shape[0])  # we get the number of the max possible indices
np.random.shuffle(indices)              # and we shuffle them
data = train_data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])  # our validation set is 20% from the train tweets

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


#######################################
# define the model structure
#######################################

print('Starting the training process!')
tf.reset_default_graph()
net = tflearn.input_data([None, MAX_SEQUENCE_LENGTH])
net = tflearn.embedding(net, input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, trainable=False, name='embeddingLayer')
net = tflearn.lstm(net, 256, return_seq=True)
net = tflearn.dropout(net, 0.25)
net = tflearn.lstm(net, 256)
net = tflearn.dropout(net, 0.25)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=3, best_val_accuracy=0.864,  # We are using tensorboard_verbose=3 for the best possible
                    best_checkpoint_path='checkpoints\\model6\\'+MODEL_NAME)                # visualisation and save checkpoints from the model if the
embeddingsLayer = tflearn.get_layer_variables_by_name('embeddingLayer')[0]                  # validation accuracy is bigger than 0.864.
model.set_weights(embeddingsLayer, embedding_matrix)  # Custom weight matrix generated from the GloVe is set as weights for the Embedding layer
model.fit(x_train, y_train, validation_set=(x_val, y_val), n_epoch=5,
          show_metric=True, batch_size=256, shuffle=True)
model.save(MODEL_PATH)
print('Training done!')



#######################################
# testing the model
#######################################

print('Testing the model!')
# model.load(model_file=MODEL_PATH)
preds = model.predict(test_data)
preds_array = []
for i in range(0, len(preds)):
    index = np.argmax(preds[i, :])  # We have a predict matrix with a dimension of 10000x2. The column with index 0 is the probability for the negative sentiment
    if index == 0:                  # and the column with index 1 is the probability for the positive sentiment.
        preds_array.append(-1)      # If the value in column one is bigger, then the prediction for this tweet is negative (-1).
    else:                           # The opposite is, of course, that this tweet has positive sentiment.
        preds_array.append(1)
preds_array = np.array(preds_array)

# Generating submission file
submission = pd.DataFrame({'Id': test_ids, 'Prediction': preds_array})
submission.to_csv('submissions\\' + MODEL_NAME + '.csv', sep=',', index=False)
