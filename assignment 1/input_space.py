from itertools import product


def input_space(domains):
    return set(product(*domains))

domains = [
    {0, 1, 2},
    {True, False},
    {9, 8}
]

print(len(sorted(input_space(domains))))
for element in sorted(input_space(domains)):
    print(element)
