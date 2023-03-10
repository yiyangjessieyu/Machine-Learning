def decode(code):

    def h(input, code=code):
        input_x, input_y = input
        x1, y1, x2, y2 = code
        isTrue = (min(x1, x2) <= input_x <= max(x1, x2)) and (min(y1, y2) <= input_y <= max(y1, y2))
        return isTrue

    return h

import itertools

h = decode((-1, -1, 1, 1))

for x in itertools.product(range(-2, 3), repeat=2):
    print(x, h(x))
x

# import itertools
#
# h1 = decode((1, 4, 7, 9))
# h2 = decode((7, 9, 1, 4))
# h3 = decode((1, 9, 7, 4))
# h4 = decode((7, 4, 1, 9))
#
# for x in itertools.product(range(-2, 11), repeat=2):
#     if len({h(x) for h in [h1, h2, h3, h4]}) != 1:
#         print("Inconsistent prediction for", x)
#         break
# else:
#     print("OK")
