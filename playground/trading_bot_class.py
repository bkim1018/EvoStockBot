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
        self.last_sell = 0
        self.last_buy = 0
        self.fitness = 0

    def sell(self,stock_price):
        if buy_state == False:
            self.buy_state = True
            self.money = self.money + self.shares * stock_price
            self.shares = 0
            self.last_trade = stock_price

    def buy(self,stock_price):
        if buy_state == True:
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
        prediction = self.neural_net.predict(data)
        index = prediction.index(max(prediction))
        return(index)

    def get_fitness(selfs):
        return self.fitness

    def setNet(self, wMat):
        # set weight matrix for the bot's neural net
        pass



    def mutate(self):
        self.neural_net = utils.mutate(self.neural_net)