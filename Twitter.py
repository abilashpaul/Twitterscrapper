from tweepy import API 
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tweepy
import re
import itertools
import collections
import nltk

ACCESS_TOKEN = "216573480-xrb4TrxptyOsvilB4d4Pt2qamvTI0Ly7mSLZNScs"
ACCESS_TOKEN_SECRET = "T5fg8uZ6tg0saP5y7xYhvtCp9CbUci5G7AJ62WWN3hASI"
CONSUMER_KEY = "j6geRiaWWDOI4Sh0JoRa3POOE"
CONSUMER_SECRET = "n1DYy0xPhJsFEKOG0jAeaBkHySvAMjq3FCz0BTzrGsbr5QN2D8"

screen_name = "realDonaldTrump"
#screen_name = "katyperry"
#screen_name = "BarackObama"

# # # # TWITTER CLIENT # # # #
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets


class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        #print(auth)
        return auth

class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """
    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])

        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df

 
if __name__ == '__main__':

 twitter_client = TwitterClient()
 tweet_analyzer = TweetAnalyzer()


    #api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
api = twitter_client.get_twitter_client_api()
    
    # Twitter allows a max of 200 tweets only
tweets = api.user_timeline(screen_name, count=200)

df = tweet_analyzer.tweets_to_data_frame(tweets)
print("Analysis for Twiter Screen Name: " ,screen_name)
    # Get average length over all tweets:
print("\nAverage length of tweets: ", round(np.mean(df['len']))) 

    # Get the number of likes for the most liked tweet:
print("# of Likes for most liked tweet: ", np.max(df['likes']))

    # Get the number of retweets for the most retweeted tweet:
print("# of Likes for most retweeted tweet: ", np.max(df['retweets']))

    # Layered Time Series:
time_likes = pd.Series(data=df['likes'].values, index=df['date'])
time_likes.plot(figsize=(16, 4), label="likes", legend=True)

time_retweets = pd.Series(data=df['retweets'].values, index=df['date'])
time_retweets.plot(figsize=(16, 4), label="retweets", legend=True)
plt.show()

    # Text Analysis
all_tweets = [tweet.text for tweet in tweets]
all_tweets[:5]  
    
    
    # remove URL
def remove_url(txt):
    return (" ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split()))

all_tweets_no_urls = [remove_url(tweet) for tweet in all_tweets]
all_tweets_no_urls[:15]
     
    
    # convert to lower case and split into unique
words_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_urls]
    

    # List of all words across tweets
all_words_no_urls = list(itertools.chain(*words_in_tweet))

    # Create counter
counts_no_urls = collections.Counter(all_words_no_urls)
counts_no_urls.most_common(15)
      

    # remove stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]
tweets_nsw[0] 
      
    
all_words_nsw = list(itertools.chain(*tweets_nsw))
counts_nsw = collections.Counter(all_words_nsw)
counts_nsw.most_common(15)

  # Word filter - 

collection_words = ['says','live']
tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in tweets_nsw]
tweets_nsw[15]

    # Flatten list of words in clean tweets
all_words_nsw_nc = list(itertools.chain(*tweets_nsw_nc))

    # Create counter of words in clean tweets
counts_nsw_nc = collections.Counter(all_words_nsw_nc)
counts_nsw_nc.most_common(15)
clean_tweets_ncw = pd.DataFrame(counts_nsw_nc.most_common(15),
                            columns=['words', 'count'])
#clean_tweets_ncw.head()

clean_tweets_nsw = pd.DataFrame(counts_nsw.most_common(15),
                             columns=['words', 'count'])

      # Plot horizontal bar graph
fig, ax = plt.subplots(figsize=(8, 8))
clean_tweets_nsw.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Tweets")

plt.show()



    # Plot horizontal bar graph
fig, ax = plt.subplots(figsize=(8, 8))
clean_tweets_ncw.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color="purple")

ax.set_title("Common Words Found in Tweets (Without Collection Words)")

plt.show()





