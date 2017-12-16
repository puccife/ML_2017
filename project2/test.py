import gensim
import nltk
#nltk.download('punkt')
test_string = "I can't it's don't i'm won't <3 ... didn't"
x = nltk.word_tokenize(test_string)
a = list(gensim.utils.tokenize(test_string))
print(a)
print(x)
