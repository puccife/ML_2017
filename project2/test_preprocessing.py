sentence = "watching off their rockets on nbc . it's hilariousss"

from preprocessing.preproc import PreprocessTweets

pt = PreprocessTweets()

sentence = pt.stem_sentence(sentence)
print(sentence)

sentence = pt.lemmatize_sentence(sentence)
print(sentence)

sentence = pt.tokenize(sentence)
print(sentence)
