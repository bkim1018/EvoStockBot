import playground
import price_data
import training_bot_class
import requests
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

NUM_GENERATIONS = 1000
NUM_BOTS = 20
START_AMT = 1000
histPathStr = os.path.join(os.getcwd(), '..', 'data', 'twitter', 'historical.csv')
## generate 100 random bots
bots = []
data = []

FB_price_data = price_data.get_google_finance_intraday('FB', period=300, days=70)
fb_tweets = playground.get_company_tweets(TWITTER_PATH, '(F|f)acebook')
data.append(('FB',FB_price_data,fb_tweets))

MSFT_price_data = price_data.get_google_finance_intraday('MSFT', period=300, days=70)
msft_tweets = playground.get_company_tweets(TWITTER_PATH, '(M|m)icrosoft')
data.append(('MSFT',MSFT_price_data,msft_tweets))

apple_price = price_data.get_google_finance_intraday('AAPL', period=300, days=70)
apple_tweets = playground.get_company_tweets(TWITTER_PATH, '(A|a)pple')
data.append(('AAPL',apple_price,apple_tweets))

### add other companies to data

for i in range(NUM_BOTS):
    bots.append(TradingBot(1000,'FB', utils.createDummy()))

for generation_index in range(NUM_GENERATIONS):
    #     utils.saveGeneration(bots, generation_index)
    startTime = time.time()
    average_fitness = []
    current_average = 0

    for bot in bots:
        current_average += total_fitness(bot, data)
    current_average = current_average / float(NUM_BOTS)
    #     utils.saveGeneration()

    average_fitness.append(current_average)
    growth = (current_average - START_AMT) / START_AMT
    print(generation_index, 'a', current_average, '    g', growth, 'Time: ', time.time() - startTime)

    botRef = bots
    bitErrRate = 1.0 / (current_average)
    bots = utils.getNextGen(bots, bitErrRate)
    for bot in botRef:
        del bot

    for bot in bots:
        bot.reset_attributes(1000, '', resetFitness=True)

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
