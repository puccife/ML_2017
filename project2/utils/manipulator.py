import numpy as np

class DatasetManipulator:

    negative_url = None

    positive_url = None

    tweets = []


    def __init__(self, positive_url, negative_url):
        self.positive_url = positive_url
        self.negative_url = negative_url


    def set_positive_url(self, positive_url):
        self.positive_url = positive_url


    def set_negative_url(self, negative_url):
        self.negative_url = negative_url


    def get_tweets(self):
        return self.tweets


    def generate_dataset(self, total_samples):
        loadedInstances = 0
        if self.positive_url == None or self.negative_url == None:
            raise Exception('Dataset url not set')
        negative_tweets = open(self.negative_url, "r")
        positive_tweets = open(self.positive_url, "r")
        self.tweets = []
        while loadedInstances < total_samples/2:
            raw_negative = negative_tweets.readline()
            raw_positive = positive_tweets.readline()
            self.tweets.append((raw_negative,0))
            self.tweets.append((raw_positive,1))
            loadedInstances = loadedInstances+1
        negative_tweets.close()
        positive_tweets.close()
        return self.get_tweets()


    def split_and_shuffle(self, tweets, ratio, seed):
        split_index = int(len(tweets)*ratio)
        train, test = tweets[:split_index], tweets[split_index:]
        np.random.seed(seed)
        np.random.shuffle(train)
        np.random.seed(seed)
        np.random.shuffle(test)
        return train, test
