import numpy as np
from preprocessing.preprocessing import clean_tweets
import os
import csv
from config.argument_loader_cnn import ArgumentLoader_cnn

# Used Flags
al_cnn = ArgumentLoader_cnn()
FLAGS = al_cnn.get_configuration()

def load_data_test():
    """
    Loads the data and and preprocesses them.
    """
    # Load data from files
    x_text = list(open("./twitter-datasets/test_data.txt").readlines())
    x_text = [s.strip() for s in x_text]
    # Split by words
    x_text = [clean_tweets(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    return x_text

def create_vector(word, word_embeddings, word_vector_size, silent=True):
    """
    Creating a new word in case he word is missing from Glove!
    :param word: word to process
    :param word2vec: word embedding dictionary
    :param word_vector_size: size of the word vector to create
    :param silent: debug option verbose
    :return: the new vector of the created word.
    """
    vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
    word_embeddings[word] = vector
    if not silent:
        print("utils.py::create_vector => %s is missing" % word)
    return vector

def manipulate_dataset(dataset,word_embeddings):
    """
    Returning the dataset as matrix of vectors. Where each word is mapped to a vector
    :param dataset: the dataset
    :param word_embeddings: the word embedding dictionary
    :return: The corresponding manipulated dataset.
    """
    # Dictionary for the missing words and their count
    missing_voc={}
    # Size of the output array
    output_array = np.ndarray((len(dataset),FLAGS.max_length,FLAGS.word_dimension))
    # For each word in the sentence find the corresponding word embedding
    for i,sentence in enumerate(dataset):
        matrix_embedding = []
        for word in sentence:
            try:
                matrix_embedding.append(word_embeddings[word])
            except:
                try:
                    missing_voc[word] = missing_voc[word] + 1
                except KeyError:
                    missing_voc[word] = 1
        for j in range(FLAGS.max_length - len(matrix_embedding)):
            matrix_embedding.append(np.zeros(FLAGS.word_dimension))
        output_array[i]=(matrix_embedding)
    return output_array

def generate_word_embeddings():
    """
    This functions is created to load the pretrained glove embedding
    :return: The word embedding as a dictionary
    """
    embeddings_index = {}
    print('Indexing word vectors.')
    # Opening the file directory
    f = open(os.path.join('./glove.twitter.27B/', 'glove.twitter.27B.'+str(FLAGS.word_dimension)+'d.txt'))
    # Getting the coefficients of each word
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def pad_sentences(sentences, padding_word="padding_word"):
    """
    Pads all sentences to the same length.
    :param sentences: The tweet dataset
    :param padding_word: "padding_word" to fill the senteces with a shorter length.
    Returns padded sentences.
    """
    # Setted length of a tweet
    sequence_length = FLAGS.max_length
    # Output array of padded sentences
    padded_sentences = []
    # Pad the sentence if it's short or return a cutted version of the tweet if too long
    for i in range(len(sentences)):
        sentence = sentences[i]
        # If num_padding is negative then the sentence is over the required length
        num_padding = sequence_length - len(sentence)
        if num_padding < 0:
            new_sentence = sentence[:sequence_length]
        # If not then pad it with the padding_word
        else:
            new_sentence = sentence + [padding_word] * num_padding

        padded_sentences.append(new_sentence)
    return padded_sentences


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
