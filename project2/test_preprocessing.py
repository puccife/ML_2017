sentence = "imagine all the people"

from preprocessing.preproc import PreprocessTweets
from utils.preprocessing import clean_tweets

print(clean_tweets(sentence))

pt = PreprocessTweets()

sentence = pt.stem_sentence(sentence)
print(sentence)

sentence = pt.lemmatize_sentence(sentence)
print(sentence)

sentence = pt.tokenize(sentence)
print(sentence)
