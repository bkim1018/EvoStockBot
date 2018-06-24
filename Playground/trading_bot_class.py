import keras
<<<<<<< HEAD:playground/trading_bot_class.py
import numpy as np
from math import floor
=======

>>>>>>> 45328cd7c93685238d928bb71a26f0dfd70cd17e:Playground/trading_bot_class.py
class TradingBot:
    """
    Instance that will store import information relevant it's portofolio
    """
<<<<<<< HEAD:playground/trading_bot_class.py
    def __init__(self,starting_money, company,neural_net):
=======
    def __init__(self, starting_money, company):
>>>>>>> 45328cd7c93685238d928bb71a26f0dfd70cd17e:Playground/trading_bot_class.py
        # Each rocket has an (x,y) position.
        self.money = starting_money
        self.shares = 0  # start with no shares
        self.company = company
        self.buy_state = True  # True = can buy stock
        self.neural_net = neural_net  # generate random neural net
        self.last_sell = 0
        self.last_buy = 0
        self.fitness = 0

    def sell(self,stock_price):
        print('Tried to sell')
        if self.buy_state == False:
            print('Sold')
            self.buy_state = True
            self.money = self.money + self.shares * stock_price
            self.shares = 0
            self.last_trade = stock_price

    def buy(self,stock_price):
<<<<<<< HEAD:playground/trading_bot_class.py
        print('Tried to buy')
        if self.buy_state == True:
            print('Bought')
=======
        if self.buy_state == True:
>>>>>>> 45328cd7c93685238d928bb71a26f0dfd70cd17e:Playground/trading_bot_class.py
            self.buy_state = False
            self.shares = floor(self.money/stock_price)
            self.money = self.money - self.shares * stock_price
            self.last_buy = stock_price

    def reset_attributes(self,new_money,new_company):
        self.money = new_money
        self.shares = 0  # start with no shares
        self.company = new_company
        self.buy_state = True  # True = can buy stock
        self.last_trade = 0

    def add_fitness(self,new_fitness):
        self.fitness += new_fitness

    def make_decision(self,data):
        prediction = self.neural_net.predict(data)[0]
        index = np.where(prediction == np.amax(prediction))[0][0]

        return(index)



    def get_fitness(selfs):
        return self.fitness

<<<<<<< HEAD:playground/trading_bot_class.py
    def setNet(self, wMat):
        self.neural_net.set_weights(wMat)
=======
    def set_net(self, wMat):
        self.neural_net = wMat
>>>>>>> 45328cd7c93685238d928bb71a26f0dfd70cd17e:Playground/trading_bot_class.py

    # def mutate(self):
    #     self.neural_net = utils.mutate(self.neural_net)