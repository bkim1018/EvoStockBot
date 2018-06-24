# Utility Functions
import struct
import numpy as np

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


def create_offspring(p1,p2):
    """
    :param p1: weight matrix for parent1's neural net
    :param p2: weight matrix for parent2's neural net
    :return: weight matrix for child's neural net or -1 on error
    """
    structure = [a.shape for a in p1]
    if [a.shape for a in p1] != [a.shape for a in p2]:
        print("Shapes of parent weight matrices must be the same.")
        return -1
    p1Flat = np.concatenate([a.flatten() for a in p1])
    p2Flat = np.concatenate([a.flatten() for a in p2])
    p1BinStr = "".join(list(map(floatToBinary, p1Flat)))
    p2BinStr = "".join(list(map(floatToBinary, p2Flat)))
    p1Bin = np.fromstring(" ".join(p1BinStr), dtype=int, sep=' ')
    p2Bin = np.fromstring(" ".join(p2BinStr), dtype=int, sep=' ')
    splitInd = np.where(np.random.randint(8, size=p1Bin.size) == 1)[0]
    p1Split = np.split(p1Bin, splitInd)
    p2Split = np.split(p2Bin, splitInd)

    # generate child's flat array
    cArrs = []
    for i in range(len(p1Split)):
        curParent = np.random.randint(2)
        if curParent == 0:
            cArrs.append(p1Split[i])
        else:
            cArrs.append(p2Split[i])
    cBin = np.concatenate(cArrs)
    cBinChunks = np.split(cBin, cBin.size / 32)
    cBinStrChunks = list(map(lambda x: np.array_str(x)[1:-1].replace(" ", "") , cBinChunks))
    cFlat = np.array(list(map(binaryToFloat, cBinStrChunks)))

    # reshape child's arrays
    cur = 0
    cArrs = []
    for s in structure:
        if len(s) == 2:
            count = s[0] * s[1]
        else:
            count = s[0]
        cArrs.append(cFlat[cur:cur+count].reshape(s))
        cur += count

    return cArrs


def mutate(neural_net):
    """
    :param net: neural net to be mutated
    :return: new net with random mutation
    """
    pass

def get_top_performers(current_generation):
    """
    :param current_generation: current generation of neural nets
    :return: top X performers of the playground
    """
    pass

def next_generation(current_generation):
    """
    :param current_generation: List of current neural nets
    :return: List of the next generation of neural nets
    """
    pass
