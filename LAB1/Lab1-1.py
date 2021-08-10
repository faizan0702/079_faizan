import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random

nltk.download('twitter_samples')

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


print('number of positive tweets:' , len(all_positive_tweets))
print('number of negative tweets:' , len(all_negative_tweets))

print('\n thee type of all_positive _tweets is : ', type(all_positive_tweets))
print('the type of a tweet entry is: ' , type(all_negative_tweets[0]))

fig = plt.figure(figsize=(5,5))

labels = 'ML-BSB-Lec' , 'ML-HAP-Lec' , 'ML-HAP-Lab'

sizes = [40,35,25]

plt.pie(sizes , labels=labels , autopct='%.2f%%',
shadow = True , startangle = 90)

plt.axis('equal')

# plt.show()

fig = plt.figure(figsize=(5,5))
labels = 'Positives','Negatives'
sizes = [len(all_positive_tweets) , len(all_negative_tweets)]

plt.pie(sizes , labels= labels , autopct ='%1.1f%%' , 
shadow=True , startangle = 90)

plt.axis('equal')

# plt.show()

# print('\033[92m'+ all_positive_tweets[random.randint(0,5000)])
# print('\033[91m' +all_negative_tweets[random.randint(0,5000)])

tweet = all_positive_tweets[2277]
# print(tweet)

nltk.download('stopwords')


import re 
import string 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

print('\033[92m' +tweet)
print('\033[94m')

tweet2 = re.sub(r'https?:\/\/.*[\r\n]*' ,'',tweet)

tweet2 = re.sub(r'#' , '' ,tweet2)

print(tweet2)


tokenizer = TweetTokenizer(preserve_case=False)

tweet_tokens = tokenizer.tokenize(tweet2)

print()
print('Tokenized string')
print(tweet_tokens)

stopwords_english = stopwords.words('english')

print('stop words\n')
print(stopwords_english)

print('\nPunctuation\n')
print(string.punctuation)


print()
print('\033[92m')
print(tweet_tokens)
print('\033[94m')

tweets_clean =[]

for word in tweet_tokens:
    if(word not in stopwords_english and 
    word not in string.punctuation):
        tweets_clean.append(word)

print('remove stop words and punction:')
print(tweets_clean)


#stemming 
stemmer = PorterStemmer()


tweets_stem = []

for word in tweets_clean:
    stem_word = stemmer.stem(word)
    tweets_stem.append(stem_word)

print('stemmed words:')
print(tweets_stem)