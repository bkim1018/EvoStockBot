def evaluate_fitness(bot,data):
    """
    bot = trading bot instance
    evaluates the fitness of each neural net
    :return: returns amount of money generated
    """
    for row in data:
        bot.make_decision(row)
    fitness = bot.money + bot.shares * row[-1]
    return fitness




