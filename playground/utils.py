# Utility Functions
import struct
import numpy as np
import random
import trading_bot_class.py

START_AMT = 100
COMPANIES = ["FB"]

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
    :param p1: weight matrix for parent1's neural net
    :param p2: weight matrix for parent2's neural net
    :return: child, new instance of TradingBot made from parents
    """
    structure = [a.shape for a in p1]
    if [a.shape for a in p1] != [a.shape for a in p2]:
        print("Shapes of parent weight matrices must be the same.")
        return -1

    p1Bin = weightMatToBin(p1)
    p2Bin = weightMatToBin(p2)
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

    child = TradingBot(START_AMT, COMPANIES)
    child.setNet(binToWeightMat(cBin, structure))

    return child


def mutate(wMat, bitErrRate):
    """
    :param wMat: weight matrix to be mutated
    :param bitErrRate: bit error rate in [0.0,1.0)
    :return: new mutated weight matrix
    """
    structure = structure = [a.shape for a in wMat]
    binArr = weightMatToBin(wMat)
    rand = np.random.random(size=binArr.size)
    toChange = np.where(rand < bitErrRate)
    binArr[toChange] = 1 - binArr[toChange]
    return binToWeightMat(binArr, structure)


def getNextGen(curGen):
    """
    :param curGen: list of trading bots from current generation
    :return: List of the next generation of neural nets
    """
    sortedGen = curGen.sort(key=lambda x: x.fitness)
    count = len(curGen)

    newGen = sortedGen[:count/2].copy()  # keep top 50%
    topShuffled = random.shuffle(sortedGen[:count / 2])
    botShuffled = random.shuffle(sortedGen[count / 2:])
    newGen += botShuffled[:count / 10]  # keep random 10% from bottom 50
    for i in range(count/4):    # get 25% children from top 50
        child = create_offspring(topShuffled[i], topShuffled[i+count/2])
        newGen.append(child)
        child.mutate()

    topShuffled = random.shuffle(topShuffled)
    botShuffled = random.shuffle(botShuffled)
    for i in range(count*3/20):  # get 15% children from mix of top50/bot50
        child = create_offspring(topShuffled[i], botShuffled[i])
        child.mutate()
        newGen.append(child)

    topShuffled = random.shuffle(topShuffled)
    newGen += [b.mutate() for b in topShuffled[:count/10].copy()]  # get 10% mutated copies from top50

    return newGen








