# Representation-dependent functions are defined outside of the main CEA
# function. This allows CEA to be representation-independent. In other words
# by defining the following functions appropriately, you can make CEA work with
# any representation.
from itertools import product


def decode(code):
    """Takes a code and returns the corresponding hypothesis."""

    def h(my_input):
        input_x, input_y = my_input
        x1, y1, x2, y2 = code
        return (min(x1, x2) <= input_x <= max(x1, x2)) and (min(y1, y2) <= input_y <= max(y1, y2))

    return h


def match(code, x):
    """Takes a code and returns True if the corresponding hypothesis returns
    True (positive) for the given input."""
    return decode(code)(x)


def lge(code_a, code_b):
    """Takes two codes and returns True if code_a is less general or equal
    to code_b."""

    # Complete this for the conjunction of constraints. You do not need to
    # decode the given codes.

    return all(code_b[i] == True for i in range(len(code_a)) if code_a[i] == True)


def initial_S(domains):
    """Takes a list of domains and returns a set where each element is a
    code for the initial members of S."""
    F = all_possible_functions(domains)
    images = set()
    for h in F:
        hx_tup = tuple(h(x) for x in domains)
        if not all(hx_tup):
            images.add(hx_tup)
    # Return an appropriate set
    return images

def all_possible_functions(X):

    input_space = list(X)

    all_functions = set()

    bool_tups = list(product([True, False], repeat=len(X)))

    for bool_tup in bool_tups:
        # bool_tup is result from 1 function.

        # function that will
        # produce a element from bool_tup (T/F) depending on input
        def f(x, bool_tup=bool_tup):

            i = input_space.index(x)
            bool = bool_tup[i]
            # print("bool_tup", bool_tup, id(bool_tup))
            # print("x, i, bool", x, i, bool)
            return bool

        all_functions.add(f)

    return all_functions


def initial_G(domains):
    """Takes a list of domains and returns a set where each element is a
    code for the initial members of G."""

    F = all_possible_functions(domains)
    images = set()
    for h in F:
        hx_tup = tuple(h(x) for x in domains)
        if all(hx_tup):
            images.add(hx_tup)
    # Return an appropriate set
    return images


def minimal_generalisations(code, x):
    """Takes a code (corresponding to a hypothesis) and returns the set of all
    codes that are the minimal generalisations of the given code with respect
    to the given input x."""

    # Return an appropriate set


def minimal_specialisations(cc, domains, x):
    """Takes a code (corresponding to a hypothesis) and returns the set of all
    codes that are the minimal specialisations of the given code with respect
    to the given input x."""

    # Return an appropriate set


def cea_trace(domains, D):
    S_trace, G_trace = [], []
    S = initial_S(domains)
    G = initial_G(domains)

    trace = S + G # Append S and G (or their copy) to corresponding trace list

    for input_x, output_y in D: # For each training example in D, with input_x and output_y, do:
        if output_y:  # if positive

            G.remove() # Remove from G any hypotheses that do not match d
            # For each hypothesis s in S that does not match d

        else:  # if negative

    # Complete

    # Append S and G (or their copy) to corresponding trace list

    return S_trace, G_trace

domains = [
    {'red', 'blue'}
]

training_examples = [
    (('red',), True)
]

S_trace, G_trace = cea_trace(domains, training_examples)
print(len(S_trace), len(G_trace))
print(all(type(x) is set for x in S_trace + G_trace))
S, G = S_trace[-1], G_trace[-1]
print(len(S), len(G))
