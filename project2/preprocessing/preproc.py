import re
import nltk

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

class PreprocessTweets:

    lancaster = None
    wordnet = None
    tknz = None

    def __init__(self):
        self.lancaster = LancasterStemmer()
        self.wordnet = WordNetLemmatizer()
        self.tknz = RegexpTokenizer(r'\w+')

    def stem_sentence(self, sentence):
        stems = [self.stem(w) for w in sentence.split()]
        return " ".join(stems)

    def stem(self, word):
        return self.lancaster.stem(word) 

    def tokenize(self, sentence):
        return " ".join(self.tknz.tokenize(sentence))

    def stopwords_parser(self, sentence):
        return sentence

    def lemmatize_sentence(self, sentence):
        lems = [self.lemmatize(w) for w in sentence.split()]
        return " ".join(lems)

    def lemmatize(self, word):
        try:
            return self.wordnet.lemmatize(word).lower()
        except Exception as e:
            print(e)
            return word