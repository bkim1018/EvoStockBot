import keras
import utils

class TradingBot(starting_money, company):
    """
    Instance that will store import information relevant it's portofolio
    """
    def __init__(self):
        # Each rocket has an (x,y) position.
        self.money = starting_money
        self.shares = 0 # start with no shares
        self.company = company
        self.buy_state = True # True = can buy stock
        self.neural_net = 0 # generate random neural net

    def sell(self,stock_price):
        if buy_state == False:
            self.buy_state = Trueu
            self.money = self.money + self.shares * stock_price
            self.shares = 0

    def buy(self,stock_price):
        if buy_state == True:
            self.buy_state = False
            self.shares = floor(self.money/stock_price)
            self.money = self.money - self.shares * stock_price

    def setNet(self, wMat):
        # set weight matrix for the bot's neural net
        pass


    def make_decision(self, data):
        pass

    def mutate(self):
        self.neural_net = utils.mutate(self.neural_net)