import numpy as np

class GloveTrainer:

    embeddings_index = {}
    vector_size = None
    GLOVE_DIR = None
    missing_voc = None

    def __init__(self, vector_size, glove_dir):
        """
        Initialize the glove trainer
        :param vector_size: Specify the size of the word_vector
        :param glove_dir: Specify the location of glove dir
        """
        self.vector_size = vector_size
        self.GLOVE_DIR = glove_dir

    def generate_word_embeddings(self):
        """
        This functions is created to load the pretrained glove embedding
        :return: The word embedding as a dictionary
        """
        print('Indexing word vectors.')
        f = open('./glove.twitter.27B/glove.twitter.27B.'+str(self.vector_size)+'d.txt', encoding="utf-8", errors='ignore')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        return self.embeddings_index


    def manipulate_dataset(self, dataset, word_embeddings):
        """
        Returning the dataset as matrix of vectors. Where each word is mapped to a vector
        :param dataset: the dataset
        :param word_embeddings: the word embedding dictionary
        :return: The corresponding manipulated dataset.
        """
        self.missing_voc = {}
        for i in range(len(dataset)):
            matrix_embedding = []
            for word in dataset[i][0].split():
                try:
                    matrix_embedding.append(word_embeddings[word])
                except:
                    try:
                        self.missing_voc[word] = self.missing_voc[word] + 1
                    except KeyError:
                        self.missing_voc[word] = 1
            dataset[i] = ((matrix_embedding, dataset[i][1]))
        return dataset

    def get_missing_voc(self):
        """
        Function used to return a dictionary of missing words with their frequence
        :return: Dictionary of missing words.
        """
        return self.missing_voc


    def process_word(self, word_embeddings, word, vocab, ivocab, word_size, to_return='wemb', silent=True):
        """
        Function used to process a word into an embedding vector.
        :param word_embeddings: 
        :param word: word to process
        :param vocab: Dictionary containing word vectors
        :param ivocab: Dictionary containing word vectors
        :param word_size: Size of each word vector
        :param to_return: type of vector to return
        :param silent: Verbose debug option
        :return: the new embedded word.
        """
        if not word in word_embeddings:
            self.create_vector(word, word_embeddings, word_size, silent)
        if not word in vocab:
            next_index = len(vocab)
            vocab[word] = next_index
            ivocab[next_index] = word
        if to_return == "wemb":
            return word2vec[word]
        elif to_return == "index":
            return vocab[word]
        return word_embeddings[word]

    def create_vector(self, word, word2vec, word_vector_size, silent=True):

        """
        Creating a new word in case he word is missing from Glove!
        :param word: word to process
        :param word2vec: word embedding dictionary
        :param word_vector_size: size of the word vector to create
        :param silent: debug option verbose
        :return: the new vector of the created word.
        """
        vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
        word2vec[word] = vector
        if not silent:
            print("word_embedding_model.py::create_vector missing word")
        return vector

    def process_input(self, data_raw, floatX, word2vec, vocab, ivocab, embed_size, split_sentences=False):
        """
        Process input of tweets using Babi format.
        :param data_raw: Raw data to process
        :param floatX: size of float to be used
        :param word2vec: pretrained word embedding dictionary
        :param vocab: dictionary to store the words in terms of their sequential appearance 
        :param ivocab: dictionary to store the words in terms of their sequential appearance 
        :param embed_size: size of the word vector
        :param split_sentences: flag used to split sentences in the same tweet - NOT USED
        :return: the cleaned input
        """
        questions = []
        inputs = []
        answers = []
        input_masks = []
        relevant_labels = []
        for x in data_raw:
            if split_sentences:
                inp = x["C"].lower().split(' . ')
                inp = [w for w in inp if len(w) > 0]
                inp = [i.split() for i in inp]
            else:
                inp = x["C"].lower().split(' ')
                inp = [w for w in inp if len(w) > 0]

            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]

            if split_sentences:
                inp_vector = [[self.process_word(word_embeddings=word2vec,
                                            word=w,
                                            vocab=vocab,
                                            ivocab=ivocab,
                                            word_size=embed_size,
                                            to_return="index") for w in s] for s in inp]
            else:
                inp_vector = [self.process_word(word=w,
                                        word_embeddings=word2vec,
                                        vocab=vocab,
                                        ivocab=ivocab,
                                        word_size=embed_size,
                                        to_return="index") for w in inp]

            q_vector = [self.process_word(word=w,
                                    word_embeddings=word2vec,
                                    vocab=vocab,
                                    ivocab=ivocab,
                                    word_size=embed_size,
                                    to_return="index") for w in q]

            if split_sentences:
                inputs.append(inp_vector)
            else:
                inputs.append(np.vstack(inp_vector).astype(floatX))
            questions.append(np.vstack(q_vector).astype(floatX))
            answers.append(self.process_word(word=x["A"],
                                        word_embeddings=word2vec,
                                        vocab=vocab,
                                        ivocab=ivocab,
                                        word_size=embed_size,
                                        to_return="index"))
            # NOTE: here we assume the answer is one word!

            if not split_sentences:
                input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32))

            relevant_labels.append(x["S"])

        return inputs, questions, answers, input_masks, relevant_labels

    def create_embedding(self, word2vec, ivocab, embed_size):
        """
        Function used to create the embedding dictionary
        :param word2vec: pretrained word2vec embedding
        :param ivocab: dictionary used to create the embedding
        :param embed_size: size of the embedding
        :return: the embedding dictionary
        """
        embedding = np.zeros((len(ivocab), embed_size))
        for i in range(len(ivocab)):
            word = ivocab[i]
            word2vec_values = []
            for value in word2vec[word]:
                word2vec_values.append(value)
            embedding[i] = word2vec_values
        return embedding

    def get_sentence_lens(self, inputs):
        """
        Function used to get lens of a sentence
        :param inputs: tweet
        :return: parameters on the sentence lens
        """
        lens = np.zeros((len(inputs)), dtype=int)
        sen_lens = []
        max_sen_lens = []
        for i, t in enumerate(inputs):
            sentence_lens = np.zeros((len(t)), dtype=int)
            for j, s in enumerate(t):
                sentence_lens[j] = len(s)
            lens[i] = len(t)
            sen_lens.append(sentence_lens)
            max_sen_lens.append(np.max(sentence_lens))
        return lens, sen_lens, max(max_sen_lens)

    def get_lens(self, inputs, split_sentences=False):
        """
        Function used to get the lens of a sentence
        :param inputs: the sentence
        :param split_sentences: Flag used to split sentences -- not used
        :return: the lens of the tweet
        """
        lens = np.zeros((len(inputs)), dtype=int)
        for i, t in enumerate(inputs):
            lens[i] = t.shape[0]
        return lens    

    def pad_inputs(self, inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
        """
        Function used to pad the input in case the input is shorter than the predefined lenght of each sentence
        :param inputs: sentences
        :param lens: defined lens of tweet
        :param max_len: max length defined
        :param mode: defined mode to pad - using mask or splitting (mask used)
        :param sen_lens: sentences
        :param max_sen_len: lens of sentences
        :return: the padded input.
        """
        if mode == "mask":
            padded = [np.pad(inp, (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
            return np.vstack(padded)

        elif mode == "split_sentences":
            padded = np.zeros((len(inputs), max_len, max_sen_len))
            for i, inp in enumerate(inputs):
                padded_sentences = [np.pad(s, (0, max_sen_len - sen_lens[i][j]), 'constant', constant_values=0) for j, s in
                                    enumerate(inp)]
                # trim array according to max allowed inputs
                if len(padded_sentences) > max_len:
                    padded_sentences = padded_sentences[(len(padded_sentences) - max_len):]
                    lens[i] = max_len
                padded_sentences = np.vstack(padded_sentences)
                padded_sentences = np.pad(padded_sentences, ((0, max_len - lens[i]), (0, 0)), 'constant', constant_values=0)
                padded[i] = padded_sentences
            return padded

        padded = [np.pad(np.squeeze(inp, axis=1), (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in
                enumerate(inputs)]
        return np.vstack(padded)