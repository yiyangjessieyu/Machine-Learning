def decode(code):

    def h(input):
        input_x, input_y = input
        x1, y1, x2, y2 = code
        return (x1 <= input_x <= x2) and (y1 <= input_y <= y2)

    return h

# import itertools
#
# h = decode((-1, -1, 1, 1))
#
# for x in itertools.product(range(-2, 3), repeat=2):
#     print(x, h(x))
# x
import itertools

h1 = decode((1, 4, 7, 9))
h2 = decode((7, 9, 1, 4))
h3 = decode((1, 9, 7, 4))
h4 = decode((7, 4, 1, 9))

for x in itertools.product(range(-2, 11), repeat=2):
    if len({h(x) for h in [h4]}) != 1:
        print("Inconsistent prediction for", x)
        break
else:
    print("OK")
