import keras
import numpy as np
from math import floor
import utils

class TradingBot:
    """
    Instance that will store import information relevant it's portofolio
    """
    def __init__(self,starting_money, company,neural_net):
        # Each rocket has an (x,y) position.
        self.money = starting_money
        self.shares = 0  # start with no shares
        self.company = company
        self.buy_state = True  # True = can buy stock
        self.neural_net = neural_net  # generate random neural net
        self.last_sell = 0
        self.last_buy = 0
        self.fitness = 0

    def sell(self,stock_price, debug_mode = False):
        # print('Tried to sell')
        if self.buy_state == False:
            self.buy_state = True
            self.money = self.money + self.shares * stock_price
            self.shares = 0
            self.last_sell = stock_price
            if debug_mode:
                return 1
            # print('Sold at ', stock_price, '    Money', self.money + self.shares * stock_price)
        if debug_mode:
            return 2

    def buy(self,stock_price, debug_mode = False):
        # print('Tried to buy')
        if self.buy_state == True:
            self.buy_state = False
            self.shares = floor(self.money/stock_price)
            self.money = self.money - self.shares * stock_price
            self.last_buy = stock_price
            if debug_mode:
                return 3
            # print('Bought at', stock_price, '    Money', self.money + self.shares * stock_price)
        if debug_mode:
            return 4

    def reset_attributes(self,new_money,new_company, resetFitness=False):
        self.money = new_money
        self.shares = 0  # start with no shares
        self.company = new_company
        self.buy_state = True  # True = can buy stock
        self.last_buy = 0
        self.last_sell = 0
        if resetFitness:
            self.fitness = 0

    def add_fitness(self,new_fitness):
        self.fitness += new_fitness

    def make_decision(self,data):
        prediction = self.neural_net.predict(data)[0]
        try:
            index = np.where(prediction == np.amax(prediction))[0][0]
            return index
        except IndexError:
            # print("RECEIVED INDEX ERROR YOU FKIN IDIOT")
            # print('data = ' , data)
            # print('prediction: ', self.neural_net.predict(data))
            return 2


    def get_fitness(self):
        return self.fitness

    def setNet(self, wMat):
        self.neural_net.set_weights(wMat)


    def mutate(self, bitErrRate):
        self.setNet(utils.mutate(self.neural_net.get_weights(), bitErrRate))