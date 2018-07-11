# Utility Functions
import os
import struct
import numpy as np
import random
import pickle
from trading_bot_class import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

START_AMT = 1000
saveFilepath = "botSaves/"

def saveGeneration(bots, numBots):
    # clear the save folder
    for the_file in os.listdir(saveFilepath):
        file_path = os.path.join(saveFilepath, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    for i in range(numBots):
        filepath = saveFilepath+'bot%d.h5' % i
        bots[i].neural_net.save(filepath)

    # with open("generations/savedGeneration %d.pkl" % number, 'wb') as outputStream:  # Overwrites any existing file.
    #     pickle.dump(gen, outputStream, pickle.HIGHEST_PROTOCOL)

def loadGeneration():
    bots = []
    for the_file in os.listdir(saveFilepath):
        file_path = os.path.join(saveFilepath, the_file)
        bots.append(keras.models.load_model(file_path))
    return bots

    # with open('savedGeneration.pkl', 'rb') as inputStream:
    #     return pickle.load(inputStream)


def createDummy():
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.

    model.add(Dense(10, input_dim=94))
    model.add(Activation('sigmoid'))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


getBin = lambda x: x > 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]
def floatToBinary(num):
    """ Convert float to a binary string. """
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

def binaryToFloat(b):
    """ Convert binary string to a float. """
#     b = b.replace(" ", "")  # remove spaces
    bf = int_to_bytes(int(b, 2), 4)  # 4 bytes needed for np.float32
    return struct.unpack('>f', bf)[0]

def int_to_bytes(n, minlen=0):  # Helper function
    """ Int/long to byte string.
        Python 3.2+ has a built-in int.to_bytes() method that could be
        used instead, but the following is portable."""
    nbits = n.bit_length() + (1 if n < 0 else 0)  # +1 for any sign bit.
    nbytes = (nbits+7) // 8  # Number of whole bytes.
    b = bytearray()
    for _ in range(nbytes):
        b.append(n & 0xff)
        n >>= 8
    if minlen and len(b) < minlen:  # Zero padding needed?
        b.extend([0] * (minlen-len(b)))
    return bytearray(reversed(b))  # High bytes first

def weightMatToBin(wMat):
    """
    :param wMat: a weight matrix
    :return: 1D binary array
    """
    flat = np.concatenate([a.flatten() for a in wMat])
    binStr = "".join(list(map(floatToBinary, flat)))
    binArr = np.fromstring(" ".join(binStr), dtype=int, sep=' ')
    return binArr

def binToWeightMat(bin, structure):
    """
    :param bin: binary array
    :param structure: list of shapes of arrays for weights of each layer
    :return: correctly structured weight matrix
    """
    # generate flat array
    binChunks = np.split(bin, bin.size / 32)
    binStrArr = list(map(lambda x: np.array_str(x)[1:-1].replace(" ", ""), binChunks))
    flat = np.array(list(map(binaryToFloat, binStrArr)))

    # reshape according to provided structure
    cur = 0
    arrs = []
    for s in structure:
        if len(s) == 2:
            count = s[0] * s[1]
        else:
            count = s[0]
        arrs.append(flat[cur:cur + count].reshape(s))
        cur += count

    return arrs

def create_offspring(p1,p2):
    """
    :param p1: trading bot parent 1
    :param p2: trading bot parents 2
    :return: child, new instance of TradingBot made from parents
    """
    p1Mat = p1.neural_net.get_weights()
    p2Mat = p2.neural_net.get_weights()
    structure = [a.shape for a in p1Mat]
    if [a.shape for a in p1Mat] != [a.shape for a in p2Mat]:
        print("Shapes of parent weight matrices must be the same.")
        return -1

    p1Bin = weightMatToBin(p1Mat)
    p2Bin = weightMatToBin(p2Mat)
    splitInd = np.where(np.random.randint(8, size=p1Bin.size) == 1)[0]
    p1Split = np.split(p1Bin, splitInd)
    p2Split = np.split(p2Bin, splitInd)


    # combine parent dna to make child
    cArrs = []
    for i in range(len(p1Split)):
        curParent = np.random.randint(2)
        if curParent == 0:
            cArrs.append(p1Split[i])
        else:
            cArrs.append(p2Split[i])
    cBin = np.concatenate(cArrs)
    child = TradingBot(START_AMT, p1.company, createDummy())
    child.neural_net.set_weights(binToWeightMat(cBin, structure))

    return child


def mutate(wMat, bitErrRate):
    """
    :param wMat: weight matrix to be mutated
    :param bitErrRate: bit error rate in [0.0,1.0)
    :return: new mutated weight matrix
    """
    structure = [a.shape for a in wMat]
    binArr = weightMatToBin(wMat)
    rand = np.random.random(size=binArr.size)
    toChange = np.where(rand < bitErrRate)
    binArr[toChange] = 1 - binArr[toChange]
    return binToWeightMat(binArr, structure)


def getNextGen(curGen, bitErrRate):
    """
    :param curGen: list of trading bots from current generation
    :return: List of the next generation of neural nets
    """
    curGen.sort(key=lambda x: x.fitness, reverse=True)
    count = len(curGen)
    newGen = curGen[:int(count*7/20)].copy()  # keep top 35%
    topHalf = curGen[:int(count/2)].copy()
    botHalf = curGen[int(count/2):].copy()
    random.shuffle(topHalf)
    random.shuffle(botHalf)
    newGen += botHalf[:int(count/20)]  # keep random 5% from bottom 50
    for i in range(int(count/4)):    # get 25% children from top 50
        child = create_offspring(topHalf[i], topHalf[i+int(count/4)])
        child.mutate(bitErrRate)
        newGen.append(child)


    random.shuffle(topHalf)
    random.shuffle(botHalf)
    for i in range(int(count*3/20)):  # get 15% children from mix of top50/bot50
        child = create_offspring(topHalf[i], botHalf[i])
        child.mutate(bitErrRate)
        newGen.append(child)

    random.shuffle(topHalf)
    for i in range(int(count/5)):    # get 20% children from top 50
        child = create_offspring(topHalf[i], topHalf[i+int(count/4)])
        child.mutate(bitErrRate)
        newGen.append(child)

    # print(type(topHalf[:int(count/10)].copy()[0].mutate))
    # print((isinstance(create_offspring(topHalf[i], topHalf[i]), type(topHalf[i]))))
    # print(type(topHalf[0]))
    # for i in range(int(count/10)):
    #     bot = create_offspring(topHalf[i], topHalf[i])  # fix me later
    #     bot.mutate(bitErrRate)
    #     newGen.append(bot)


    return newGen








