
import glmnet_python
from glmnet_python import glmnet


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from numpy import genfromtxt

import sys
sys.path.append('../test')
sys.path.append('../lib')
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
from glmnet_python import glmnet; from glmnet_python import glmnetPlot
from glmnet_python import glmnetPrint; from glmnet_python import glmnetCoef; from glmnet_python import glmnetPredict
from glmnet_python import cvglmnet; from glmnet_python import cvglmnetCoef
from glmnet_python import cvglmnetPlot; from glmnet_python import cvglmnetPredict

def load_data():

    data = genfromtxt('abalone.data', delimiter=',', dtype='str')
    # get age by adding 1.5 to the rings
    data[:, -1] = (data[:, -1].astype('float64') + 1.5).astype('str')
    return data



# part a
def a(isplot=True):

    regr = linear_model.LinearRegression()
    data = load_data()

    # make x, y as np array
    x = data[:, 1:data.shape[1]-1].reshape(-1, data.shape[1]-2).astype('float64')
    y = data[:, data.shape[1]-1:].reshape(-1, 1).astype('float64')

    #
    regr.fit(x, y)
    yy = regr.predict(x)

    if isplot:
        _, ax = plt.subplots()
        plt.title('Residual plot (without including gender) of linear regression for predicting age')
        ax.scatter(y.tolist(), (y-yy).tolist(), c='b', alpha=0.25)
        ax.grid(True)
        plt.show()

    return x, y

# part b
def b(isplot=True):

    regr = linear_model.LinearRegression()
    data = load_data()

    # make x, y as np array
    gender = data[:, 0].tolist()
    gender_map = []
    for i in range(len(gender)):
        if gender[i][0] == 'M':
            gender_map.append(1)
        elif gender[i][0] == 'I':
            gender_map.append(0)
        elif gender[i][0] == 'F':
            gender_map.append(-1)
    data[:, 0] = np.array(gender_map)
    x = data[:, 0:data.shape[1]-1].reshape(-1, data.shape[1]-1).astype('float64')
    y = data[:, data.shape[1]-1:].reshape(-1, 1).astype('float64')

    regr.fit(x, y)
    yy = regr.predict(x)

    if isplot:
        _, ax = plt.subplots()
        plt.title('Residual plot (with including gender) of linear regression for predicting age')
        ax.scatter(y.tolist(), (y-yy).tolist(), c='b', alpha=0.25)
        ax.grid(True)
        plt.show()

    return x, y


def c(isplot=True):
    regr = linear_model.LinearRegression()
    data = load_data()

    # make x, y as np array
    x = data[:, 1:data.shape[1]-1].reshape(-1, data.shape[1]-2).astype('float64')
    y = data[:, data.shape[1]-1:].reshape(-1, 1).astype('float64')

    y = np.log(y)

    #
    regr.fit(x, y)
    yy = regr.predict(x)

    y_original = np.exp(y)
    yy_original = np.exp(yy)

    if isplot:
        _, ax = plt.subplots()
        plt.title('Residual plot (without including gender) of linear regression for predicting log of age')
        ax.scatter(y_original.tolist(), (y_original-yy_original).tolist(), c='b', alpha=0.25)
        ax.grid(True)
        plt.show()

    return x, y


def d(isplot=True):

    regr = linear_model.LinearRegression()
    data = load_data()

    # make x, y as np array
    gender = data[:, 0].tolist()
    gender_map = []
    for i in range(len(gender)):
        if gender[i][0] == 'M':
            gender_map.append(1)
        elif gender[i][0] == 'I':
            gender_map.append(0)
        elif gender[i][0] == 'F':
            gender_map.append(-1)
    data[:, 0] = np.array(gender_map)
    x = data[:, 0:data.shape[1]-1].reshape(-1, data.shape[1]-1).astype('float64')
    y = data[:, data.shape[1]-1:].reshape(-1, 1).astype('float64')

    y = np.log(y)
    #
    regr.fit(x, y)
    yy = regr.predict(x)

    y_original = np.exp(y)
    yy_original = np.exp(yy)

    if isplot:
        _, ax = plt.subplots()
        plt.title('Residual plot (with including gender) of linear regression for predicting log of age')
        ax.scatter(y_original.tolist(), (y_original- yy_original).tolist(), c='b', alpha=0.25)
        ax.grid(True)
        plt.show()

    return x, y


def f():
    a_x, a_y = a(False)
    b_x, b_y = b(False)
    c_x, c_y = c(False)
    d_x, d_y = d(False)

    cv_a = cvglmnet(x=a_x, y=a_y, alpha=0)
    cv_b = cvglmnet(x=b_x, y=b_y, alpha=0)
    cv_c = cvglmnet(x=c_x, y=c_y, alpha=0)
    cv_d = cvglmnet(x=d_x, y=d_y, alpha=0)

    f = plt.figure()
    axa = f.add_subplot(2,2,1)
    axa.title.set_text("Regularization constant plot for predicting age without gender")
    cvglmnetPlot(cv_a)
    axb = f.add_subplot(2,2,2)
    axb.title.set_text("Regularization constant plot for predicting age with gender")
    cvglmnetPlot(cv_b)
    axc = f.add_subplot(2,2,3)
    axc.title.set_text("Regularization constant plot for predicting log of age without gender")
    cvglmnetPlot(cv_c)
    axd = f.add_subplot(2,2,4)
    axd.title.set_text("Regularization constant plot for predicting log of age with gender")
    cvglmnetPlot(cv_d)
    plt.show()


if __name__ == "__main__":
    a()
    b()
    c()
    d()
    f()




