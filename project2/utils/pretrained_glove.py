import os
import numpy as np

class GloveTrainer:

    embeddings_index = {}
    vector_size = None
    GLOVE_DIR = None
    missing_voc = None

    def __init__(self, vector_size, glove_dir):
        self.vector_size = vector_size
        self.GLOVE_DIR = glove_dir

    def generate_word_embeddings(self):
        print('Indexing word vectors.')
        f = open('./glove.twitter.27B.'+str(self.vector_size)+'d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        return self.embeddings_index


    def manipulate_dataset(self, dataset, word_embeddings):
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
                    # print("word: " + word)
            dataset[i] = ((matrix_embedding, dataset[i][1]))
        return dataset

    def get_missing_voc(self):
        return self.missing_voc


    def process_word(self, word_embeddings, word, vocab, ivocab, word_size, to_return='wemb', silent=False):
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
        # if the word is missing from Glove or Google Vectors, create some fake vector and store in glove!
        vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
        word2vec[word] = vector
        if not silent:
            print("utils.py::create_vector => %s is missing" % word)
        return vector

    def process_input(self, data_raw, floatX, word2vec, vocab, ivocab, embed_size, split_sentences=False):
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
                if input_mask_mode == 'word':
                    input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32))
                elif input_mask_mode == 'sentence':
                    input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32))
                else:
                    raise Exception("invalid input_mask_mode")

            relevant_labels.append(x["S"])

        return inputs, questions, answers, input_masks, relevant_labels

    def create_embedding(self, word2vec, ivocab, embed_size):
        embedding = np.zeros((len(ivocab), embed_size))
        for i in range(len(ivocab)):
            word = ivocab[i]
            word2vec_values = []
            for value in word2vec[word]:
                word2vec_values.append(value)
            embedding[i] = word2vec_values
        return embedding

    def get_sentence_lens(self, inputs):
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
        lens = np.zeros((len(inputs)), dtype=int)
        for i, t in enumerate(inputs):
            lens[i] = t.shape[0]
        return lens    

    def pad_inputs(self, inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
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