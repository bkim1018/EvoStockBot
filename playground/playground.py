from utils import *

def evaluate_fitness(bot,data,senti_data):
    """
    bot = trading bot instance
    evaluates the fitness of each neural net
    :return: returns amount of money generated
    one-days data and 10 minute intervals
    """
    time = data.index.values
    useful_data = data.values[:, [0, 4]]
    SLIDING_LENGTH = 45
    SENTIMENT_LENGTH = 10

    for cur_day in range(useful_data[SLIDING_LENGTH:].shape[0]):
        data_slice = useful_data[cur_day:cur_day + SLIDING_LENGTH]
        data_slice = normalize_data(data_slice).flatten()
        current_time = time[cur_day + SLIDING_LENGTH]

        ### TODO : sentiment of that time
        ### TODO : append sum of sentiment to data_slice

        bot.make_decision(data_slice)
    fitness = bot.money + bot.shares * useful_data[-1][0]
    return fitness

def normalize_data(slidiing_window_data):
    """
    Normalizes the data such that each point after the first is a percent difference of the first point
    :param slidiing_window_data: sliding window of the
    :return:
    """
    start = slidiing_window_data[0]
    normalized_data = (slidiing_window_data - start)/float(start)
    return normalized_data


def get_sentiment(time,senti_data):
    pass
