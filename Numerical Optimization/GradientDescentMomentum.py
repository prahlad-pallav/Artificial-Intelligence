import numpy as np
from matplotlib import pyplot as plt


# here is the cost function

def f(x):
    return x * x

def df(x):
    return 2 * x

# gradient descent algorithm -> n is the no. of iterations and alpha is the learning rate
# we track the results ( x and f(x) values as well)

def gradient_descent(start, end, n, alpha=0.1, momentum=0.0):
    x_values = []
    y_values = []
    x = np.random.uniform(start, end)

    for i in range(n):
        x = x - alpha * df(x) - momentum * x
        x_values.append(x)
        y_values.append(f(x))
        print('#%d f(%s) = %s' % (i, x, f(x)))

    return [x_values, y_values]

if __name__ == "__main__":
    solutions, scores = gradient_descent(-1, 1, 50, 0.1, momentum=0.3)
    inputs = np.arange(-1, 1.1, 0.1)
    plt.plot(inputs, f(inputs))
    plt.plot(solutions, scores, '.-', color="green")
    plt.show()


