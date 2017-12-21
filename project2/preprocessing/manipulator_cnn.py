import numpy as np
from preprocessing.preprocessing import clean_tweets

class DatasetManipulator_cnn:

    negative_url = None

    positive_url = None

    tweet_length = None

    padded_sentences = []

    def __init__(self, tweet_length,positive_url, negative_url):
        """
        Initialize the dataset manipulator
        :param tweet_length: Maximum number of words in a tweet
        :param positive_url: url of positive tweets
        :param negative_url: url of negative tweets
        """
        self.positive_url = positive_url
        self.negative_url = negative_url
        self.tweet_length = tweet_length

    def set_positive_url(self, positive_url):
        """
        Setter for positive tweet url
        :param positive_url: url of positive tweets
        """
        self.positive_url = positive_url


    def set_negative_url(self, negative_url):
        """
        Setter for negative tweet url
        :param negative_url: url of negative tweets
        """
        self.negative_url = negative_url

    def set_tweet_length(self, tweet_length):
        """
        Setter for negative tweet url
        :param negative_url: url of negative tweets
        """
        self.tweet_length = tweet_length

    def get_tweets(self):
        """
        Getter for tweets
        :return: the stored tweets
        """
        return self.tweets

    def load_data_and_labels(self):
        """
        Loads the data and creates the corresponding label for the positive/negative tweets
        after preprocessing them.
        """
        # Load data from files
        positive_examples = list(open(self.positive_url, encoding="utf-8", errors='ignore').readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(self.negative_url, encoding="utf-8", errors='ignore').readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # Clean the tweets and split by words
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
        Pads all sentences to the same length.
        :param sentences: The tweet dataset
        :param padding_word: "padding_word" to fill the senteces with a shorter length.
        Returns padded sentences.
        """
        # Setted length of a tweet
        sequence_length = self.tweet_length
        # Output array of padded sentences
        self.padded_sentences = []
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

            self.padded_sentences.append(new_sentence)
        return self.padded_sentences


    def split_and_shuffle(self,x,y, ratio, seed):
        """
        Split and shuffle tweets according to specified ratio and seed
        :param x: tweets to shuffle
        :param y: label of the tweets
        :param ratio: ratio of the splitting
        :param seed: seed to use
        :return: returns the splitted training and testing dataset
        """
        # Ratio length of the total
        split_index = int(len(x)*ratio)
        # Splitting according to the split_index
        train_x, test_x = x[:split_index], x[split_index:]
        train_y, test_y = y[:split_index], y[split_index:]
        # Generating random numbers from a fixed seed and shuffling the data.
        np.random.seed(seed)
        np.random.shuffle(train_x)
        np.random.seed(seed)
        np.random.shuffle(test_x)
        np.random.seed(seed)
        np.random.shuffle(train_y)
        np.random.seed(seed)
        np.random.shuffle(test_y)
        return train_x,test_x,train_y,test_y
