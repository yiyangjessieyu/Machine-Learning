def less_general_or_equal(ha, hb, X):

    ha_outputs = [ha(x) for x in X]
    hb_outputs = [hb(x) for x in X]

    # print("ha_output")
    # for ha_output in ha_outputs:
    #     print(ha_output)
    #
    # print("hb_output")
    # for hb_output in hb_outputs:
    #     print(hb_output)

    return ha_outputs <= hb_outputs and all(ha_x == hb_x for ha_x in ha_outputs for hb_x in hb_outputs if
                                            ha_x <= hb_x <= ha_x)

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
        ha_outputs = [ha(x) for x in X]
        hb_outputs = [hb(x) for x in X]
        print("ha_outputs", sum(ha_outputs))
        # print(ha_outputs)
        print("hb_outputs", sum(hb_outputs))
        # print(hb_outputs)
     #   if ha == h3 and hb == h2:

