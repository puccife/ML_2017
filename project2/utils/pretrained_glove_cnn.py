import os
import numpy as np

class GloveTrainer_cnn:

    embeddings_index={}
    vector_size = None
    GLOVE_DIR = None
    missing_voc = None
    tweet_length = None

    def __init__(self, tweet_length , vector_size, glove_dir):
        self.vector_size = vector_size
        self.GLOVE_DIR = glove_dir
        self.tweet_length = tweet_length

    def generate_word_embeddings(self):
        print('Indexing word vectors.')
        f = open(os.path.join(self.GLOVE_DIR, 'glove.twitter.27B.'+str(self.vector_size)+'d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        return self.embeddings_index


    def manipulate_dataset(self,dataset,word_embeddings):
        self.missing_voc={}
        self.output_array = np.ndarray((len(dataset),self.tweet_length,self.vector_size))
        for i,sentence in enumerate(dataset):
            matrix_embedding = []
            for word in sentence:
                try:
                    matrix_embedding.append(word_embeddings[word])
                except:
                    vector = self.create_vector(word,word_embeddings,self.vector_size,silent=True)
                    matrix_embedding.append(vector)
                    try:
                        self.missing_voc[word] = self.missing_voc[word] + 1
                    except KeyError:
                        self.missing_voc[word] = 1
            self.output_array[i]=(matrix_embedding)
        return self.output_array

    def create_vector(self,word, word_embeddings, word_vector_size, silent=True):
        # if the word is missing from Glove or Google Vectors, create some fake vector and store in glove!
        vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
        word_embeddings[word] = vector
        if not silent:
            print("utils.py::create_vector => %s is missing" % word)
        return vector

    def get_missing_voc(self):
        return self.missing_voc
