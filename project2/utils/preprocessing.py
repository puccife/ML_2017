import re
from string import digits
from autocorrect import spell
import gensim
from nltk.corpus import stopwords

# English Contractions that are most common and most found in the training set.
contractions_dict = {
    "<user>":"",
    "<url>":"",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "I'm": "I am",
    "i'm": "I am",
    "I've": "I have",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "that'll":"that will",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "here's" : "here is",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "<3" : "love",
    "women's": "women",
    "men's": "men",
    "everyone's": "everyone is",
    ":p":"face with tongue",
    "d:":"face with tongue",
    ":d":"grinning face",
    ";d":"winking face",
    ":/":"hesitant face",
    ";/":"hesitant face",
    "=/":"hesitant face",
    "#yougetmajorpointsif": "you get major points if",
    "alll":"all",
    "youuu":"you",
    "pleaseee":"please",
    "#thoughtsduringschool":"thoughts during school",
    "#sadtweet":"sad tweet",
    "today's":"today",
    "#smartnokialumia":"smart nokia lumia",
    "#waystomakemehappy":"ways to make me happy",
    "today's":'today is',
    "ughhh":"frustration",
    "ugh":"frustration",
    "yeahhh":"yeah",
    ";p":"winking face with tongue",
    "wahhh":"wow",
    "#cantsayno":"can not say no",
    "loveee":"love",
    "yayyy":"approval",
    "heyyy":"hey",
    "omggg":"oh my god",
    "#ifindthatattractive":"i find that attractive",
    "whyyy":"why",
    "someone's":"someone is",
    "america's":"america is",
    "#fcblive":"football club barcelona live",
    "knowww":"know",
    "#harrypotterchatuplines":"harry potter chat up lines",
    "waaa":"greeting",
    "misss":"miss",
    "thanksss":"thanks",
    "ya'll":"you all",
    "#yougetmajorpoints":"you get major points",
    "#theluckyone":"the lucky one",
    "welll":"well",
    "shhh":"quiet",
    "#muchlove":"much love",
    "#adoptakuola":"adopt a kuola",
    "u're":"you are",
    "#thingsiwanttohappen":"things i want to happen",
    "friend's":"friend",
    "#aday":"a day",
    "#thevoice":"the voice",
    "mannn":"man",
    "#mentionto":"mention to",
    "life's":"life is",
    "spiral-bound":"spiral bound",
    ":-d":"grinning face",
    "ewww":"disgust",
    "yeaaa":"yea",
    "#justsaying":"just saying",
    "uhhh":"surprise",
    "awhhh":"cute",
    "everything's":"everything is",
    "#pbbteens":"pinoy big brother teens",
    "people's":"people",
    "u'll":"you will",
    "]:":"sad face",
    ":-p":"face with tongue",
    "#youcangetmajorpointsif":"you can get major points if",
    "ittt":"it",
    "#happytweet":"happy tweet",
    "#dfamily":"the family",
    "nowww":"now",
    "#cantwait":"can not wait",
    "babyyy":"baby",
    "#waystobeginsex":"ways to begin sex",
    ":'d":"face of joy",
    "=d":"grinning face",
    "wayyy":"way",
    "#thevoiceuk":"the voice united kingdom",
    "#rip":"rest in peace",
    "#christianpickuplines":"christian pick up lines",
    "#throwbackthursday":"throw back thursday",
    "#itssadthat":"it is sad that",
    "playaway":"play away",
    "#mybiggestfearis":"my biggest fear is",
    "uppp":"up",
    "#cleancloud":"clean cloud",
    "okayyy":"okay",
    "#sadtimes":"sad times",
    "girlll":"girl",
    "#missyou":"miss you",
    "world-bank":"world bank",
    "punk'd":"punked",
    "cuteee":"cute",
    "fuckkk":"fuck",
    "#neversaynever":"never say never",
    "everrr":"ever",
    "damnnn":"damn",
    "shittt":"shit",
    "#thinklikeaman":"think like a man",
    "#nahhh":"no",
    "#boyfriendvideo":"boyfriend video",
    "#sosad":"so sad",
    "#ohwell":"oh well",
    "#youknowicarewhen":"you know i care when",
    "#ificanthaveyou":"if i can not have you",
    "justsayin'":"just saying",
    "#firstworldproblems":"first world problems",
    "wtfff":"what the fuck",
    "#sorrynotsorry":"sorry not sorry",
    "#sadface":"sad face",
    "#bestsmells":"best smells",
    "#realtalk":"real talk",
    "#smh":"shaking my head",
    "#heartbroken":"heart broken",
    "#factsaboutme":"facts about me",
    "#badtimes":"bad times",
    "#soexcited" :"so excited",
    "#sadday":"sad day",
    "#notcool":"not cool",
    "#sotired":"so tired",
    "#iwish":"i wish",
    "#notgood":"not good",
    "#fingerscrossed":"fingers crossed",
    "#goodday":"good day",
    "#sadlife":"sad life",
    "#sohappy":"so happy",
    "#happygirl":"happy girl",
    "#happybirthday":"happy birthday",
    "#soproud":"so proud",
    "#lovinglife":"loving life",
    "#thuglife":"thug life",
    "didnt":"did not",
     }
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

## Initialize Stopwords
stopWords = stopwords.words("english")
## Remove words that denote sentiment
for w in ['no', 'not', 'nor', 'only', 'against', 'up', 'down', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
'haven', 'isn', 'ain', 'aren', 'mightn', 'mustn', 'needn',
'wasn', 'weren', 'wouldn','shouldn']:
    stopWords.remove(w)


def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def remove_stopwords_from_tweet(tweet):
    tokens = tweet.split()
    for word in tokens:
        if word in stopWords:
            tokens.remove(word)
    return ' '.join(tokens)

def clean_str(string):
    """
    Cleans the tweet from emoji
    :param string: takes a tweet
    :return: A cleaned tweet
    """
    string = re.sub(r":\)","grinning face", string)
    string = re.sub(r":'\)","face of joy", string)
    string = re.sub(r"=\)","grinning face", string)
    string = re.sub(r"=\(","sad face", string)
    string = re.sub(r"\(8","grinning face", string)
    string = re.sub(r":\|","expressionless face", string)

    return string.strip()


def clean_tweets(tweet):
    """
    Cleans the tweet from digits,contractions, and transforming it to lower case
    :param string: takes a tweet
    :return: A cleaned tweet
    """
    remove_digits = str.maketrans('', '', digits)
    tweet= tweet.translate(remove_digits)
    tweet= tweet.lower()
    tweet= expand_contractions(tweet)
    tweet = tweet.lower()
    tweet = clean_str(tweet)
    tweet_tokenized = list(gensim.utils.tokenize(tweet))
    tweet = ' '.join(tweet_tokenized)
#    tweet = remove_stopwords_from_tweet(tweet)
    return tweet
