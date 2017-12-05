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

    def get_generator(self, training_set, FLAGS):
        word_size = FLAGS.word_dimension
        batch_size = FLAGS.batch_size
        max_lenght = FLAGS.max_lenght
        num_classes = FLAGS.num_classes

        batched_input = []
        batched_label = []
        batched_mask = []

        i = 0
        num_batch = 0
        batch_completed = False
        while i < (len(training_set) + 1):
            if not batch_completed:
                if len(batched_input) < batch_size:
                    data = training_set[i][0]
                    data = data[:max_lenght]
                    if len(data) > 0:
                        padded = np.lib.pad(data, ((max_lenght - len(data), 0), (0, 0)), 'constant', constant_values=(0))
                        delta = 1.0 / (len(data))
                    else:
                        padded = np.zeros(shape=[max_lenght, word_size])
                        delta = 1.0
                    batched_input.append(padded)
                    if True:
                        label = np.zeros(shape=[max_lenght,num_classes])
                        for j in range(0,max_lenght):
                            np.put(label[j],training_set[i][1],1.0)
                        batched_label.append(label)
                        mask = np.zeros(max_lenght)
                        weight = np.zeros(len(data))
                        for number in range(0, len(data)):
                            np.put(weight, number, delta * (number + 1))
                        np.put(mask, np.arange(max_lenght - len(data), max_lenght), weight)
                        batched_mask.append(mask)
                        i += 1
                else:
                    reshaped = np.reshape(batched_input, newshape=[batch_size, max_lenght, word_size])
                    labels = np.reshape(batched_label, newshape=[batch_size, max_lenght, num_classes])
                    masks = np.reshape(batched_mask, newshape=[batch_size, max_lenght])
                    toAppend = (reshaped, labels, masks)
                    num_batch += 1
                    if (len(training_set) // batch_size) == num_batch:
                        batch_completed = True
                    batched_input = []
                    batched_label = []
                    batched_mask = []
                    yield toAppend
            else:
                break
