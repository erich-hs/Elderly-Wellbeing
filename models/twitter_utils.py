import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import tokenization
from sklearn import preprocessing
import preprocessor as p
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download
nltk.download('wordnet')
nltk.download('stopwords')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.metrics import BinaryAccuracy
import math


# Capturing new keywords with Hashtag Extraction using Regex
def hashtag_extract(tweet):
    """
    Extract hashtag from text strings within individual tweets.

    Argument: tweet -- String.
    Returns: hashtag -- List of lowercase strings. Hashtags contained on tweet.
    """
    hashtag = re.findall(r"#(\w+)", tweet)
    hashtag = list(map(str.lower, hashtag))
    return hashtag


# Text cleaning - Removing URLs, mentions, etc using tweet-preprocessor package
def tweet_clean(tweet):
    """
  Clean tweet with tweet-preprocessor p.clean() removing unwanted characters,
  user mentions, punctuations and setting to lower case text.

  Argument: tweet -- Ttext string.
  Returns: cleaned_tweet -- Cleaned tweet text string.
  """
    # Remove user mentions, symbols and unwanted characters
    tweet = p.clean(tweet)

    # Remove digits and setting lower case text
    tweet = tweet.replace('\d+', '').lower()

    # Remove punctuations
    cleaned_tweet = re.sub(r'[^\w\s]', '', tweet)

    return cleaned_tweet


# Stemming, Lemmatization and Tokenization using nltk package
def tweet_preprocess(tweet):
    """
  Process tweets with stemming, lemmatization, tokenization, and removing stopwords.
  Stopwords dictionary = English

  Arguments: tweet -- String.
  Returns: processed_tweed -- List of strings with stemmed lemmatized tokenized
  words contained in tweet, without stopwords.
  """
    stemmer = nltk.SnowballStemmer('english')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokenizer = TweetTokenizer()
    stop_words = set(stopwords.words('english'))  # Create stopwords set

    # Stemming (having -> have)
    tweet = stemmer.stem(tweet)

    # Lemmatization ('dogs' -> 'dog') and
    # Tokenization ('good muffins cost $10' -> ['good', 'muffins', 'cost', '$', '10'])
    tweet = [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(tweet)]

    # Removing stop words (a, an, our, etc.)
    processed_tweet = []
    for word in tweet:
        if word not in stop_words:
            processed_tweet.append(word)

    return processed_tweet


def sequentialize(tweets, max_words=5000, words_per_element=50, enforce_words_per_element=False):
    """
  Sequentialize tweets into an (m, x) matrix, where:
  m = number of tweet samples
  x = dimension of individual tweet object, after tokenized

  Arguments:
  tweets -- preprocessed tweets. Each individual element should be contained into a list.
            Use tweet_preprocess() function before sequentialize()
  max_words -- Int. Hyperparameter where only the most common (max_words - 1) will be kept,
            based on word frequency
  words_per_element -- Int. Default = 50. Maximum dimension of individual tweet vectors,
            padded with zeros. Also seen as number of features per training/testing element.
            Automatically set to the maximum number needed per individual tweet, after a
            first sequentialize iteration.
  enforce_words_per_element -- Boolean. Default = False. When 'True', enforces the inputed
            words_per_element value into the final tweet vector.

  Returns:
  sequences_matrix -- Numpy array of shape (m, x).
  """
    # max_words = 5000
    # Hyperparameter: Only the most common (max_words - 1) will be kept, based on word frequency

    global max_len
    max_len = words_per_element
    # max_len = 5
    # Number of words to pad and fit into feature X matrix

    tok = Tokenizer(num_words=max_words)
    X = tweets

    tok.fit_on_texts(X)  # Updates internal vocabulary based on list of texts
    sequences = tok.texts_to_sequences(X)  # Tokenize word sequences
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

    # Checking token counts per tweet
    word_counts = []
    for i in range(0, sequences_matrix.shape[0]):
        word_counts.append(np.count_nonzero(sequences_matrix[i, :]))

    if (max(word_counts) < max_len) and not enforce_words_per_element:
        max_len = max(word_counts)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
        print('max_words_per_element value updated:', max_len)

    global highest_word_token
    highest_word_token = max(list(max(i) for i in sequences_matrix[:, ]))

    print(f'Highest word token: {highest_word_token}')

    return sequences_matrix
