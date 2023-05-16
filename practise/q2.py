#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'getkthSmallestTerm' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. INTEGER_ARRAY arr
#  2. LONG_INTEGER k
#

import itertools

# def getkthSmallestTerm(arr, k):
#     # Write your code here
#     combinations = []
#     sorted(arr)
#
#     for num1 in arr:
#         for num2 in arr:
#             combinations.append((num1, num2))
#
#     # Optional: If you want to eliminate duplicate combinations
#     combinations = list(set(combinations))
#
#     print(combinations[k-1])

def getkthSmallestTerm(arr, k):
    # Write your code here
    combinations = sorted(itertools.product(arr, repeat=2))
    return combinations[k - 1]


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr_count = int(input().strip())

    arr = []

    for _ in range(arr_count):
        arr_item = int(input().strip())
        arr.append(arr_item)

    k = int(input().strip())

    result = getkthSmallestTerm(arr, k)

    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
