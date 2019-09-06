import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import math


data = """2	15.11
4	11.36
6	9.77
8	9.09
10	8.48
15	7.69
20	7.33
25	7.06
30	6.7
40	6.43
50	6.16
60	5.99
70	5.77
80	5.64
90	5.39
110	5.09
130	4.87
150	4.6
160	4.5
170	4.36
180	4.2
"""


def load_data():
    v = data.splitlines()
    x = [v[i].split('\t')[0] for i in range(len(v))]
    y = [v[i].split('\t')[1] for i in range(len(v))]
    return x, y


# part a
def a():
    regr = linear_model.LinearRegression()
    x, y = load_data()

    # make x, y as log x, log y
    print(x)
    x = [math.log(float(x[i])) for i in range(len(x))]
    print(x)
    y = [math.log(float(y[i])) for i in range(len(y))]

    # make x, y as np array
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    regr.fit(x, y)

    plt.scatter(x.tolist(), y.tolist(),  color='black')
    plt.plot(x.tolist(), regr.predict(x).tolist(), color='blue', linewidth=2)
    plt.title('Regression line in log-log coordinates')
    plt.xlabel('Log of Hours')
    plt.ylabel('Log of Sulfate')
    plt.show()


# part b
def b():
    regr = linear_model.LinearRegression()
    x, y = load_data()

    # make x, y as log x, log y
    print(x)
    x = [math.log(float(x[i])) for i in range(len(x))]
    print(x)
    y = [math.log(float(y[i])) for i in range(len(y))]

    # make x, y as np array
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    regr.fit(x, y)

    yy = regr.predict(x)

    # reverse log to plot original coordinate
    x = x.tolist()
    x = [math.exp(x[i][0]) for i in range(len(x))]

    y = y.tolist()
    y = [math.exp(y[i][0]) for i in range(len(y))]

    yy = yy.tolist()
    yy = [math.exp(yy[i][0]) for i in range(len(yy))]

    plt.scatter(x, y,  color='black')
    plt.plot(x, yy, color='blue', linewidth=2)
    plt.title('Regression curve in original coordinates')
    plt.xlabel('Hours')
    plt.ylabel('Sulfate')
    plt.show()

# part c
def c_1():
    regr = linear_model.LinearRegression()
    x, y = load_data()

    # make x, y as log x, log y
    print(x)
    x = [math.log(float(x[i])) for i in range(len(x))]
    print(x)
    y = [math.log(float(y[i])) for i in range(len(y))]

    # make x, y as np array
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    regr.fit(x, y)
    yy = regr.predict(x)
    plt.title('Residual plot in log-log coordinates')
    plt.scatter(y.tolist(), (y - yy).tolist())
    plt.show()

def c_2():
    regr = linear_model.LinearRegression()
    x, y = load_data()

    # make x, y as log x, log y
    print(x)
    x = [math.log(float(x[i])) for i in range(len(x))]
    print(x)
    y = [math.log(float(y[i])) for i in range(len(y))]

    # make x, y as np array
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    regr.fit(x, y)
    yy = regr.predict(x)

    y_original = [math.exp(y[i][0]) for i in range(len(y))]

    y_residuals = [math.exp(y[i][0] - yy[i][0]) for i in range(len(y))]

    plt.title('Residual plot in original coordinates')
    plt.scatter(y_original, y_residuals)
    plt.show()


if __name__ == "__main__":
    a()
    b()
    c_1()
    c_2()



