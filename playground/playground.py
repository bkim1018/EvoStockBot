from utils import *
import numpy as np
import pandas as pd
from datetime import datetime
import re
from dateutil import parser

SLIDING_LENGTH = 45
SENTIMENT_LENGTH = 10

def evaluate_fitness(bot,data,senti_data):
    """
    bot = trading bot instance
    data = historical price data of a certain stock [price, volume]
    senti_data = historical twitter sentiment of a stocks company

    evaluates the fitness of each neural net
    :return: returns amount of money generated
    one-days data and 10 minute intervals
    """
    time = data.index.values
    useful_data = data.values[:, [0, 4]]


    for cur_day in range(useful_data[SLIDING_LENGTH:].shape[0]):
        data_slice = useful_data[cur_day:cur_day + SLIDING_LENGTH]
        data_slice = normalize_data(data_slice).flatten()

        current_time = time[cur_day + SLIDING_LENGTH] ## must convert to datetime format for comparison
        sentiment = get_sentiment(current_time,senti_data,SENTIMENT_LENGTH)

        data_slice = list(data_slice)

        data_slice.append(sentiment)
        print(current_time,sentiment)
    #
    #     bot.make_decision(data_slice)
    #
    # fitness = bot.money + bot.shares * useful_data[-1][0] ## show me the money
    return fitness

def normalize_data(slidiing_window_data):
    """
    Normalizes the data such that each point after the first is a percent difference of the first point
    :param slidiing_window_data: sliding window of the
    :return:
    """
    start = slidiing_window_data[0]
    normalized_data = (slidiing_window_data - start)/start
    return normalized_data


def get_sentiment(time,senti_data,LAST_COUNT):
    """
    :param time: current time
    :param senti_data: sorted sentiment data
    :param LAST_COUNT: number of desiered last tweets
    :return: sum of their sentiments
    """

    before_data = senti_data.loc[senti_data['Datetime'] < time]
    sub = before_data.iloc[-LAST_COUNT:]
    sum_sentiment = sum(sub['Sentiment'])
    return sum_sentiment

def get_company_tweets(csv,company_regex):
    """
    :param csv: csv file name
    :param company_regex: regex to select company
    :return: dataframe with only dates and sentiment of tweets sorted by date
    """
    data = pd.read_csv(csv)
    company_tweets = data.loc[data['Texts'].str.contains(company_regex, na=False)]
    company_tweets = company_tweets[['Dates', 'Sentiment']]
    to_datetime_data = [parser.parse(date) for date in company_tweets['Dates']]
    company_tweets['Datetime'] = to_datetime_data
    company_tweets = company_tweets.sort_values(by = 'Datetime')

    return company_tweets