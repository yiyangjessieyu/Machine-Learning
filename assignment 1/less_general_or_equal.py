def less_general_or_equal(ha, hb, X):
    return all([ha(x) <= hb(x) for x in X])


X = list(range(1000))


def h2(x):
    return x % 2 == 0


def h3(x):
    return x % 3 == 0


def h6(x):
    return x % 6 == 0


H = [h2, h3, h6]

for ha in H:
    for hb in H:
        print(ha.__name__, "<=", hb.__name__, "?", less_general_or_equal(ha, hb, X))
        # ha_outputs = [ha(x) for x in X]
        # hb_outputs = [hb(x) for x in X]
        # print("ha_outputs", sum(ha_outputs))
        # print(ha_outputs)
        # print("hb_outputs", sum(hb_outputs))
        # print(hb_outputs)
    #   if ha == h3 and hb == h2:
