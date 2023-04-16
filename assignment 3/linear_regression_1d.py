def dot_product(a, b, n):
    return sum([a[i] * b[i] for i in range(n)])


def linear_regression_1d(data):
    """
    Where   x is vector of feature values, and
            y is the vector of response values, and
            n is the length of these vectors.
    :param data: list of pairs (feature value x, response value y)
    :return: list of pairs (gradient m, intercept c)
    """

    sum_x, sum_y, n = 0, 0, 0
    for x, y in data:
        sum_x += x
        sum_y += y
        n += 1

    # m = (n x.y - ∑x ∑y) / (n x.x - (∑x)**2), the slope of the line of least squares fit.
    x, y = zip(*data)
    m = (n * dot_product(x, y, n) - sum_x * sum_y) / (n * dot_product(x, x, n) - sum_x ** 2)

    # c = (∑y - m∑x)/n, the intercept of the line of least squares fit.
    c = (sum_y - m * sum_x) / n

    return (m, c)


data = [(1, 4), (2, 7), (3, 10)]
m, c = linear_regression_1d(data)
print(m, c)
print(4 * m + c)
