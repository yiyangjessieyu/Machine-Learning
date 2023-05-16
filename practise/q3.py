#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'getMinLength' function below.
#
# The function is expected to return an INTEGER.
# The function accepts STRING seq as parameter.
#

def getMinLength(seq): # BABAAABABABBAAABB

    AB, BB = 'AB', 'BB'

    def get_removal_count(seq):

        if seq == '':
            return 0

        elif seq == AB or seq == BB:
            return 1

        else:
            if AB in seq or BB in seq:
                i = seq.index(AB) or seq.index(BB)
                seq = seq[:i] + seq[i + 2:]
                print(seq)
                if seq:
                    return 1 + get_removal_count(seq)
                else:
                    return 1

    return len(seq) - (2 * get_removal_count(seq))


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    seq = input()

    result = getMinLength(seq)

    fptr.write(str(result) + '\n')

    fptr.close()
