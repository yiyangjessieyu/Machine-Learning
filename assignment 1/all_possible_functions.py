from itertools import product

def all_possible_functions(X):

    input_space = list(X)

    all_functions = set()

    bool_tups = list(product([True, False], repeat=len(X)))

    for bool_tup in bool_tups:
        # bool_tup is result from 1 function.

        # function that will
        # produce a element from bool_tup depending on input
        def f(x, bool_tup=bool_tup):

            i = input_space.index(x)
            bool = bool_tup[i]
            # print("bool_tup", bool_tup, id(bool_tup))
            # print("x, i, bool", x, i, bool)
            return bool

        all_functions.add(f)

    return all_functions


X = {"green", "purple"}  # an input space with two elements
F = all_possible_functions(X)

# Let's store the image of each function in F as a tuple
images = set()
for h in F:
    images.add(tuple(h(x) for x in X))

for image in sorted(images):
    print(image)

