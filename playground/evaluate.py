import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'playground'))

import playground
from trading_bot_class import *

import concurrent.futures

def evaluateFitness(bot, data, startAmt, p=False):
    """
    :param bot: bot to evaluate over data
    :param data: list of datasets to evaluate over
    :return:
    """
    for company_name,company_price,company_senti in data:
        company_fit = playground.evaluate_fitness(bot, company_price, company_senti)
        bot.add_fitness(company_fit)
        bot.reset_attributes(startAmt, company_name)

    if p:
        print(bot.get_fitness())

    return bot

def evaluateGeneration(gen, data, startAmt, skipPercent, pFlag=False, numWorkers=5):
    numBots = len(gen)
    assert ((numBots * skipPercent) % 1 == 0)
    bots = gen[:numBots * skipPercent].copy()

    if pFlag:
        for i in range(numBots * skipPercent):
            print(bots[i].get_fitness())

    for i, bot in enumerate(bots):
        if i >= (numBots * skipPercent):
            bots.append(evaluateFitness(bot, data, startAmt, skipPercent))

    with concurrent.futures.ProcessPoolExecutor() as exe:
        try:
            bots = list(exe.map(evaluateFitness, bots, chunksize=numBots/numWorkers))
        except Exception:
            print("Received exception while calling evaluateGeneration. SKipping this generation. ")
            return gen
    return bots


if __name__ == '__main__':
    evaluateFitness(sys.argv)