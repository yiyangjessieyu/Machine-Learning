#!/bin/python3
import itertools
import math
import os
import random
import re
import sys


#
# Complete the 'countTeams' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. INTEGER_ARRAY rating
#  2. 2D_INTEGER_ARRAY queries
#

def countTeams(rating, queries):
    # Write your code here
    query_results = []
    print(rating)
    print(queries)

    for l, r in queries:
        print("============")
        print(l, r)
        team_count = 0
        employees_tracker = rating[l - 1: r]
        print(employees_tracker)

        pairs = list(itertools.product(employees_tracker, repeat=2))
        print(pairs)

        for i, pair in enumerate(pairs):
            employeeA, employeeB = pair
            if employeeA == employeeB:
                team_count += 1
                del pairs[i]

                if (employeeB, employeeA) in pairs:
                    other_i = pairs.index((employeeB, employeeA))
                    del pairs[other_i]

        query_results.append(team_count)

    return query_results


def matching_IDs(rating, IDA, IDB):
    return rating[IDA] == rating[IDB]


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    rating_count = int(input().strip())

    rating = []

    for _ in range(rating_count):
        rating_item = int(input().strip())
        rating.append(rating_item)

    queries_rows = int(input().strip())
    queries_columns = int(input().strip())

    queries = []

    for _ in range(queries_rows):
        queries.append(list(map(int, input().rstrip().split())))

    result = countTeams(rating, queries)

    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
