
import nltk
import matplotlib.pyplot as plt
import pandas as pd

# text analysis
text = """The Securities and Exchange Board of India (Sebi)'s new mandate in margin trading, which was brought into effect last year in a phased manner, has increased upfront requirement to 100 from Wednesday. Sebi hiked the upfront margin requirement to 50 from 25 from 1 March 2021 and further to 75 in June."""

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

remove_link = re.sub(f'https?:\/\/.*[\r\n]*', '', text)
remove_link = re.sub(r'#', '', remove_link)
print(remove_link)

print('\033[92m' + text)
print('\033[92m' + remove_link)

from nltk.tokenize import sent_tokenize

text = """As per SEBI regulations on peak margins, starting Tuesday, June 1, 2021, intraday leverages will be reduced to ensure 75 of the margin required is collected for all Equity and derivative positions."""
nltk.download('punkt')
tokenized_text = sent_tokenize(text)
print(tokenized_text)

from nltk.tokenize import word_tokenize
tokenized_word = word_tokenize(text)

print(tokenized_word)

# frequency distribution
from nltk.probability import FreqDist
fredist = FreqDist(tokenized_word)
fredist.most_common(4)

#plotting frequency distribution
import matplotlib.pyplot as plt
fredist.plot(30, cumulative = False, color = 'blue')
plt.show()

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)

filtered_sentence = []
for word in tokenized_word:
  if word not in stop_words:
    filtered_sentence.append(word)
print('Tokenized Sentence : \n', tokenized_word)
print('\nFiltered Sentence : \n', filtered_sentence)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
stemmed_sentence = []
for word in filtered_sentence:
  stemmed_sentence.append(ps.stem(word))

print('Filtered Sentence : \n', filtered_sentence)
print('\nStemmed Sentence : \n', stemmed_sentence)

# stemming and lemmatization

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

ps = PorterStemmer()
word = 'crying'
print('Lemmatized Word  :  ', lemmatizer.lemmatize(word, 'v'))
print('Stemmed word  :  ', ps.stem(word))