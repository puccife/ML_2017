import numpy as np
import re
from utils.preprocessing import clean_tweets

class DatasetManipulator_cnn:

    negative_url = None

    positive_url = None

    tweet_length = None

    padded_sentences = []
    
    def __init__(self, tweet_length,positive_url, negative_url):
        self.positive_url = positive_url
        self.negative_url = negative_url
        self.tweet_length = tweet_length

    def set_positive_url(self, positive_url):
        self.positive_url = positive_url


    def set_negative_url(self, negative_url):
        self.negative_url = negative_url


    def get_tweets(self):
        return self.tweets

    def load_data_and_labels(self):

        # Load data from files
        positive_examples = list(open(self.positive_url, encoding="utf-8", errors='ignore').readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(self.negative_url, encoding="utf-8", errors='ignore').readlines())
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

    def pad_sentences(self,sentences, padding_word="padding_word"):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        sequence_length = self.tweet_length
        self.padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = sequence_length - len(sentence)
            if num_padding < 0:
                new_sentence = sentence[:sequence_length]
            else:
                new_sentence = sentence + [padding_word] * num_padding

            self.padded_sentences.append(new_sentence)
        return self.padded_sentences


    def split_and_shuffle(self,x,y, ratio, seed):
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
