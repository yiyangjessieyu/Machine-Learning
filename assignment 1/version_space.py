from itertools import product

def all_possible_functions(X):
    input_space = list(X)
    all_functions = set()
    bool_tups = list(product([True, False], repeat=len(X)))

    for bool_tup in bool_tups:  # bool_tup is result from 1 function.

        # function that will produce a element from bool_tup (T/F) depending on input
        def f(x, bool_tup=bool_tup):
            i = input_space.index(x)
            bool = bool_tup[i]
            # print("bool_tup", bool_tup, id(bool_tup))
            # print("x, i, bool", x, i, bool)
            return bool

        all_functions.add(f)

    return all_functions


def consistent(h, D):
    return all([h(x) == y for x, y in D])

def version_space(H, D):
    """
    takes a set of hypotheses H, and a training data set D, and return its version space

    :param H: elements of H are hypotheses.
    Each hypothesis (function) takes an input object and returns True or False (i.e. a binary classifier).

    :param D: elements of D are 2-tuples of the form (x, y) where
    x is an input object (hypothesis function  input)
    y is either True or False (hypothesis function  output)

    :return: a set which will be a subset of (or equal to) H
    """
    return {h for h in H if consistent(h, D)}


X = {"green", "purple"}  # an input space with two elements
D = {("green", True)}  # the training data is a subset of X * {True, False}
F = all_possible_functions(X)
H = F  # H must be a subset of (or equal to) F

VS = version_space(H, D)

print(len(VS))

for h in VS:
    for x, y in D:
        if h(x) != y:
            print("You have a hypothesis in VS that does not agree with the set D!")
            break
    else:
        continue
    break
else:
    print("OK")
