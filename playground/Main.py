import playground
import price_data
import training_bot_class
import requests
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

NUM_GENERATIONS = 1000
NUM_BOTS = 100
## generate 100 random bots
bots = []
data = []

FB_price_data = price_data.get_google_finance_intraday('FB', period=300, days=70)
fb_tweets = playground.get_company_tweets(TWITTER_PATH, '(F|f)acebook')
sentiments = playground.evaluate_fitness(bot, dfs, fb_tweets)

data.append(('FB',FB_price_data,sentiments))

### add other companies to data

for i in NUM_BOTS:
    bots.append(training_bot_class(1000,'FB'))

for generation_index in NUM_GENERATIONS:
    average_fitness = []
    current_average = 0

    for bot in bots:
        current_average += total_fitness(bot,data)
    current_average = current_average / float(NUM_BOTS)

    average_fitness.append(current_average)
    print(current_average)

    bots = playground.next_generation(bots)

def total_fitness(bot,data):
    """
    :param bot:
    :param data:
    :return:
    """
    for company_name,company_price,company_senti in data:
        company_fit = playground.evaluate_fitness(bot, company_price, company_senti)
        bot.add_fitness(company_fit)
        bot.reset_attributes(1000,company_name)

    return bot.get_fitness()
