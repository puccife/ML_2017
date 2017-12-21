import numpy as np
from preprocessing.preprocessing import clean_tweets

class DatasetManipulator:

    negative_url = None

    positive_url = None

    testing_url = None

    testing_tweets = []

    tweets = []


    def __init__(self, positive_url, negative_url, testing_url):
        """
        Initialize the dataset manipulator
        :param positive_url: url of positive tweets
        :param negative_url: url of negative tweets
        :param testing_url: url of test tweets
        """
        self.positive_url = positive_url
        self.negative_url = negative_url
        self.testing_url = testing_url

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


    def get_tweets(self):
        """
        Getter for tweets
        :return: the stored tweets
        """
        return self.tweets


    def generate_dataset(self, total_samples):
        """
        Generating the dataset given the url defined in __init__ function
        :param total_samples: number of samples to load
        :return: the generated dataset with half positive sentences and half negative sentences.
        """
        loadedInstances = 0
        if self.positive_url == None or self.negative_url == None:
            raise Exception('Dataset url not set')
        negative_tweets = open(self.negative_url, "r", encoding="utf-8", errors='ignore')
        positive_tweets = open(self.positive_url, "r", encoding="utf-8", errors='ignore')
        self.tweets = []
        while loadedInstances < total_samples/2:
            raw_negative = negative_tweets.readline()
            raw_positive = positive_tweets.readline()
            clean_negative = clean_tweets(raw_negative)
            clean_positive = clean_tweets(raw_positive)
            self.tweets.append((clean_negative,0))
            self.tweets.append((clean_positive,1))
            loadedInstances = loadedInstances+1
        negative_tweets.close()
        positive_tweets.close()
        return self.get_tweets()

    def generate_testing_dataset(self, size=10000):
        """
        Generating the testing dataset
        :return: the generated testing dataset
        """
        loadedInstances = 0
        if self.testing_url == None:
            raise Exception('Testing dataset url not set')
        testing_tweets = open(self.testing_url, "r", encoding="utf-8", errors='ignore')
        self.testing_tweets = []
        while loadedInstances < size:
            raw_test = testing_tweets.readline()
            raw_test = (raw_test.split(",", 1)[1])
            clean_test = clean_tweets(raw_test)
            self.testing_tweets.append((clean_test,0))
            loadedInstances = loadedInstances+1
        testing_tweets.close()
        return self.testing_tweets

    def format_like_babi(self, reviews_to_format):
        """
        Format sentences in Babi Facebook format
        :param reviews_to_format: tweets
        :return: the formatted sentences
        """
        reviews_text_to_output = ""
        s_index = 1
        question = "What is the sentiment?"
        for review in (reviews_to_format):
            reviews_text_to_output += str(s_index) + ' ' + review[0] + '\n'
            # Add question to classify the current review like proposed in
            #  Ask Me Anything: Dynamic Memory Networks for Natural Language Processing paper.
            # http://proceedings.mlr.press/v48/kumar16.pdf
            answer = str(review[1])
            reviews_text_to_output += str(s_index + 1) + ' ' + question + '\t' + answer + '\t' + str(s_index) + '\n'
        return reviews_text_to_output    

    def save_reviews_splitted(self, reviews_train, reviews_test):
        """
        Save formatted tweets
        :param reviews_train: training tweets
        :param reviews_test: testing tweets
        """
        with (open("./data/tweet_train.txt", "w", encoding="utf-8", errors='ignore')) as rev_file_train:
            rev_file_train.write(reviews_train)
        # Test
        with (open("./data/tweet_test.txt", "w", encoding="utf-8", errors='ignore')) as rev_file_train:
            rev_file_train.write(reviews_test)

    def init_babi(self, fname):
        """
        Initializing dictionary in babi format before training
        :param fname: name of the file to open (training or testing usually)
        :return: the Babi formatted dictionary
        """
        print("==> Loading test from %s" % fname)
        tasks = []
        task = None
        for i, line in enumerate(open(fname, encoding="utf-8", errors='ignore')):
            id = int(line[0:line.find(' ')])
            if id == 1:
                task = {"C": "", "Q": "", "A": "", "S": ""}
                counter = 0
                id_map = {}
            line = line.strip()
            line = line.replace('.', ' . ')
            line = line[line.find(' ') + 1:]
            # if not a question
            if line.find('?') == -1 or id == 1:
                task["C"] += line
                id_map[id] = counter
                counter += 1
            else:
                idx = line.find('?')
                tmp = line[idx + 1:].split('\t')
                task["Q"] = line[:idx]
                task["A"] = tmp[1].strip()
                task["S"] = []
                for num in tmp[2].split():
                    task["S"].append(id_map[int(num.strip())])
                tasks.append(task.copy())
        return tasks

    def split_and_shuffle(self, tweets, ratio, seed):
        """
        Split and shuffle tweets according to specified ratio and seed
        :param tweets: tweets to shuffle
        :param ratio: ratio of the splitting
        :param seed: seed to use
        :return: return the splitted dataset
        """
        split_index = int(len(tweets)*ratio)
        train, test = tweets[:split_index], tweets[split_index:]
        np.random.seed(seed)
        np.random.shuffle(train)
        np.random.seed(seed)
        np.random.shuffle(test)
        return train, test

    def get_generator(self, training_set, FLAGS):
        """
        Creating a generator for the training batches
        :param training_set: training set
        :param FLAGS: flags containing parameters used to create batches (size, etc...)
        """
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
