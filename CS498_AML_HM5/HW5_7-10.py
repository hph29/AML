import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import math
import seaborn as sns



data = """77	28.5	33.5	100	38.5	114	85	178	37.5	53	58
85.5	29.5	36.5	107	39	119	90.5	187	40	52	59
63	25	31	94	36.5	102	80.5	175	33	49	57
80.5	28.5	34	104	39	114	91.5	183	38	50	60
79.5	28.5	36.5	107	39	114	92	174	40	53	59
94	30.5	38	112	39	121	101	180	39.5	57.5	59
66	26.5	29	93	35	105	76	177.5	38.5	50	58.5
69	27	31	95	37	108	84	182.5	36	49	60
65	26.5	29	93	35	112	74	178.5	34	47	55.5
58	26.5	31	96	35	103	76	168.5	35	46	58
69.5	28.5	37	109.5	39	118	80	170	38	50	58.5
73	27.5	33	102	38.5	113	86	180	36	49	59
74	29.5	36	101	38.5	115.5	82	186.5	38	49	60
68	25	30	98.5	37	108	82	188	37	49.5	57
80	29.5	36	103	40	117	95.5	173	37	52.5	58
66	26.5	32.5	89	35	104.5	81	171	38	48	56.5
54.5	24	30	92.5	35.5	102	76	169	32	42	57
64	25.5	28.5	87.5	35	109	84	181	35.5	42	58
84	30	34.5	99	40.5	119	88	188	39	50.5	56
73	28	34.5	97	37	104	82	173	38	49	58
89	29	35.5	106	39	118	96	179	39.5	51	58.5
94	31	33.5	106	39	120	99.5	184	42	55	57
"""


def load_data():
    v = data.splitlines()
    x = [v[i].split('\t')[1:] for i in range(len(v))]
    y = [v[i].split('\t')[0] for i in range(len(v))]
    return x,y


# part a
def a():

    regr = linear_model.LinearRegression()
    x, y = load_data()

    # make x, y as np array
    x = np.array(x).reshape(-1, 10).astype('float64')
    y = np.array(y).reshape(-1, 1).astype('float64')

    regr.fit(x, y)
    yy = regr.predict(x)

    _, ax = plt.subplots()
    plt.title('Residual plot in original coordinates')
    ax.scatter(y.tolist(), (y-yy).tolist(), c='b', alpha=0.7, label='original coordinate')
    ax.legend()
    ax.grid(True)
    plt.show()
    return y.tolist(), (y-yy).tolist()


# part b
def b():
    regr = linear_model.LinearRegression()
    x, y = load_data()

    print(y)
    y_cube_root = [float(y[i]) ** (1. /3) for i in range(len(y))]
    print(y)
    # make x, y as np array
    x = np.array(x).reshape(-1, 10).astype('float64')
    y_cube_root = np.array(y_cube_root).reshape(-1, 1).astype('float64')

    regr.fit(x, y_cube_root)
    yy_cube_root = regr.predict(x)

    y_original = np.array([y_cube_root.tolist()[i][0] ** 3 for i in range(len(y_cube_root))]).reshape(-1, 1).astype('float64')
    yy_original = np.array([ yy_cube_root.tolist()[i][0] ** 3 for i in range(len(yy_cube_root))]).reshape(-1, 1).astype('float64')

    _, ax = plt.subplots()
    plt.title('Residual plot of cube root of body mass in cube root coordinates')
    ax.scatter(y_cube_root.tolist(), (y_cube_root-yy_cube_root).tolist(), c='b', alpha=0.7, label='cube root coordinate')
    ax.legend()
    ax.grid(True)
    plt.show()

    b, c = a()
    _, ax = plt.subplots()
    plt.title('Residual plot from regress cube root of body mass vs body mass')
    ax.scatter(y_original.tolist(), (y_original-yy_original).tolist(), c='r', alpha=0.7, label='original coordinate from regress cube root of body mass')
    ax.scatter(b, c, c='b', alpha=0.7, label='original coordinate from regress body mass')
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":

    a()
    b()



