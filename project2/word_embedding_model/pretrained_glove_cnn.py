import os
import numpy as np

class GloveTrainer_cnn:

    embeddings_index={}
    vector_size = None
    GLOVE_DIR = None
    missing_voc = None
    tweet_length = None

    def __init__(self, tweet_length , vector_size, glove_dir):
        """
        Initialize the glove trainer
        :param vector_size: Specify the size of the word_vector
        :param glove_dir: Specify the location of glove dir
        :param tweet_length: Maximum number of words in a tweet
        """
        self.vector_size = vector_size
        self.GLOVE_DIR = glove_dir
        self.tweet_length = tweet_length

    def generate_word_embeddings(self):
        """
        This functions is created to load the pretrained glove embedding
        :return: The word embedding as a dictionary
        """
        print('Indexing word vectors.')
        # Opening the file directory
        f = open(os.path.join(self.GLOVE_DIR, 'glove.twitter.27B.'+str(self.vector_size)+'d.txt'), encoding='utf-8')
        # Getting the coefficients of each word
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        return self.embeddings_index


    def manipulate_dataset(self,dataset,word_embeddings):
        """
        Returning the dataset as matrix of vectors. Where each word is mapped to a vector
        :param dataset: the dataset
        :param word_embeddings: the word embedding dictionary
        :return: The corresponding mani pulated dataset.
        """
        # Dictionary for the missing words and their count
        self.missing_voc={}
        # Size of the output array
        self.output_array = np.ndarray((len(dataset),self.tweet_length,self.vector_size))
        # For each word in the sentence find the corresponding word embedding
        for i,sentence in enumerate(dataset):
            matrix_embedding = []
            for word in sentence:
                try:
                    matrix_embedding.append(word_embeddings[word])
                except:
                    #print('missing word not added')
                    # if the word is not found in the embeddings create a random vector with the word embedding
                    #vector = self.create_vector(word,word_embeddings,self.vector_size,silent=True)
                    #matrix_embedding.append(vector)
                    try:
                        self.missing_voc[word] = self.missing_voc[word] + 1
                    except KeyError:
                        self.missing_voc[word] = 1
            for j in range(self.tweet_length - len(matrix_embedding)):
                matrix_embedding.append(np.zeros(self.vector_size))
            self.output_array[i]=(matrix_embedding)
        return self.output_array

    def create_vector(self,word, word_embeddings, word_vector_size, silent=True):
        """
        Creating a new word in case he word is missing from Glove!
        :param word: word to process
        :param word2vec: word embedding dictionary
        :param word_vector_size: size of the word vector to create
        :param silent: debug option verbose
        :return: the new vector of the created word.
        """
        vector = np.random.uniform(0.0, 0.0, (word_vector_size,))
        word_embeddings[word] = vector
        if not silent:
            print("word_embedding_model.py::create_vector => %s is missing" % word)
        return vector

    def get_missing_voc(self):
        """
        Function used to return a dictionary of missing words with their frequence
        :return: Dictionary of missing words.
        """
        return self.missing_voc
