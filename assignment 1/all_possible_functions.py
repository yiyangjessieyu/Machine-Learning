from itertools import product

def all_possible_functions(X):

    X = list(X)

    F = set()

    Y = list(product([True, False], repeat=len(X)))

    for y in Y:
        # y is result from 1 function.

        # function that will
        # produce a element from y (T/F) depending on input
        def f(x, y=y):

            i = X.index(x)
            bool = y[i]
            return bool

        F.add(f)

    return F


X = {"green", "purple"}  # an input space with two elements
F = all_possible_functions(X)

# Let's store the image of each function in F as a tuple
images = set()
for h in F:
    images.add(tuple(h(x) for x in X))

for image in sorted(images):
    print(image)

