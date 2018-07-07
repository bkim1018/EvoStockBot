import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'playground'))

import playground
from trading_bot_class import *

def total_fitness(bot,data):
    """
    :param bot:
    :param data:
    :return:
    """
    for company_name,company_price,company_senti in data:
        company_fit = playground.evaluate_fitness(bot, company_price, company_senti)
        bot.add_fitness(company_fit)
        bot.reset_attributes(1000, company_name)
    print(bot.get_fitness())
    return bot.get_fitness()