import os
import numpy as np

GLOVE_DIR = 'glove.twitter.27B/'
glove_25 = 'glove.twitter.27B.25d.txt'
glove_50 = 'glove.twitter.27B.50d.txt'
glove_100 = 'glove.twitter.27B.100d.txt'
glove_200 = 'glove.twitter.27B.200d.txt'

class GloveTrainer:

    embeddings_index = {}
    vector_size = None

    def __init__(self, vector_size=25):
        self.vector_size = vector_size

    def generate_word_embeddings(self):
        print('Indexing word vectors.')
        f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.'+str(self.vector_size)+'d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        return self.embeddings_index


    def manipulate_dataset(self, dataset, word_embeddings):
        for i in range(len(dataset)):
            matrix_embedding = []
            for word in dataset[i][0].split():
                try:
                    matrix_embedding.append(word_embeddings[word])
                except:
                    word
                    # print("word: " + word)
            dataset[i] = ((matrix_embedding, dataset[i][1]))
        return dataset
