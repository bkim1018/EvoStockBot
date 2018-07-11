import sys, os
def insertIntoPath(newPath):
    if newPath not in sys.path:
        sys.path.insert(0, newPath)
insertIntoPath(os.path.join(os.getcwd(), '..', '..', 'playground'))
insertIntoPath(os.path.join(os.getcwd(), '..', 'price'))

import tweepy
import numpy as np
import pandas as pd
import time
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from datetime import datetime
import re
from dateutil import parser


def getTwitterApi():
    consumer_token = "FSdktl3u8Q4xhmEHeC7tt4Q6P"
    consumer_secret = "pErt27djL4HCcBt0z9PU6w7zVFygfVsEP1Aqx2gQ70uYibqP7p"
    access_key = "1902120404-I1QEwvP7uLEBRJefMMPdPSf4uChDnsNvzO8hFcj"
    access_secret = "RZAp4AlaXoszBXqp8gkMJM1ed6TJzTBFCDS99248hV9i4"


    auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    return tweepy.API(auth)

def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except (tweepy.RateLimitError, tweepy.TweepError):
            print("Rate limit exceeded, I'm passing out for a bit peace")
            time.sleep(15 * 60)


def makeCleanRep(origSource, removeLink=True):
    """
    Returns a list of the sid, text, and date of a given source.
    """

    sid = origSource._json['id']
    text = origSource._json['full_text']
    date = origSource._json['created_at']

    if removeLink == True:
        nonLinks = []
        for word in text.split(" "):
            if "http" not in word:
                nonLinks.append(word)
        text = " ".join(nonLinks)

    return [sid, text, date]


def getCurrentTwitterData(api):
    allData = []
    ids = []
    texts = []
    dates = []
    sources = []
    handles = ["technology", "business", "WSJbusiness", "BBCWorldBiz", "ForbesTech", "CNBCtech"]
    c = 0

    for handle in handles:
#         print("Starting", handle,  "...")
        hc = 0
        cur = tweepy.Cursor(api.user_timeline, screen_name = handle, tweet_mode='extended').items(100)
        for source in limit_handled(cur):
            clean = makeCleanRep(source)
            if clean[1] not in texts:
                ids.append(clean[0])
                texts.append(clean[1])
                dates.append(clean[2])
                sources.append(handle)
                c += 1
                hc += 1
#         print("Gathered", hc, "tweets for", handle)

    sia = SIA()
    df = pd.DataFrame({"IDs":ids, "Texts":texts, "Dates":dates, "Sources": sources}, columns=["IDs", "Texts", "Dates", "Sources"])
    df['Sentiment'] = pd.Series([sia.polarity_scores(text)['compound'] for text in df['Texts']])
    return df


def filterTwitterData(data, company_regex):
    """
    :param csv: csv file name
    :param company_regex: regex to select company
    :return: dataframe with only dates and sentiment of tweets sorted by date
    """
    company_tweets = data.loc[data['Texts'].str.contains(company_regex, na=False)]
    company_tweets = company_tweets[['Dates', 'Sentiment', 'IDs', 'Sources']]
    to_datetime_data = [parser.parse(date) for date in company_tweets['Dates']]
    company_tweets['Datetime'] = to_datetime_data
    company_tweets = company_tweets.sort_values(by = 'Datetime', ascending=False)
    return company_tweets[['Datetime', 'IDs', 'Sentiment', 'Sources']]