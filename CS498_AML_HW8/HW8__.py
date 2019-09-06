import numpy as np
from matplotlib import pyplot as plt
import mnist

def show(image):
    plt.imshow(image)
    plt.show()

# get first 20 images and binarize images
images = mnist.train_images()
X = images[0:20]
X = np.where(X < 128, -1, 1)

# noise images
noises = np.genfromtxt('NoiseCoordinates.csv', delimiter=',', skip_header=1, usecols=range(1, 16))
for i in range(len(noises)):
    if i % 2 == 1:
        continue
    image_id = int(i / 2)
    for j in range(noises.shape[1]):
        X[image_id][int(noises[i, j]), int(noises[i + 1, j])] *= -1

# boltzman machine
Q = np.genfromtxt('InitialParametersModel.csv', delimiter=',')
order = np.genfromtxt('UpdateOrderCoordinates.csv', delimiter=',', skip_header=1, usecols=range(1, 785))

EQ = np.zeros((20, 28, 28))
for i in range(20):
    EQ[i] = Q

# define constant
theta_HH = 0.8
theta_HX = 2.0
e = 10e-10
size = 10
EQ = EQ[0:size]
noiseX = X[size:2 * size]
order = order[2 * size: 2 * 2 * size, :]
num_img = len(EQ)

# List to store the energy after each iteration
energy_list = list()
# iter 10 times
for iters in range(10):

    # update E-Q
    E_q_log_h_x = np.zeros((num_img, 1))
    E_q_log_Q = np.zeros((num_img, 1))

    for img_id in range(num_img):
        for row in range(EQ.shape[1]):
            for col in range(EQ.shape[2]):

                term1 = 0
                if row != 0:
                    term1 += theta_HH * (2 * EQ[img_id][row, col] - 1) * (2 * EQ[img_id][row-1, col] - 1)
                if col != 0:
                    term1 += theta_HH * (2 * EQ[img_id][row, col] - 1) * (2 * EQ[img_id][row, col-1] - 1)
                if row != 27:
                    term1 += theta_HH * (2 * EQ[img_id][row, col] - 1) * (2 * EQ[img_id][row+1, col] - 1)
                if col != 27:
                    term1 += theta_HH * (2 * EQ[img_id][row, col] - 1) * (2 * EQ[img_id][row, col+1] - 1)

                E_q_log_h_x[img_id] += term1 + theta_HX * (2 * EQ[img_id][row, col] - 1) * noiseX[img_id][row, col]

                E_q_log_Q[img_id] += EQ[img_id, row, col] * np.log((EQ[img_id, row, col] + e)) + \
                                     (1 - EQ[img_id, row, col]) * np.log(((1 - EQ[img_id, row, col]) + e))

    new_EQ = E_q_log_Q - E_q_log_h_x

    energy_list.append(new_EQ)

    # update pi
    for i in range(0, order.shape[0]):
        if i % 2 == 1: continue
        for j in range(order.shape[1]):
            image_id = int(i / 2)
            row = int(order[i, j])
            col = int(order[i + 1, j])
            x1 = 0
            x2 = 0

            if row != 0:
                t = theta_HH * (2 * EQ[image_id, row-1, col] - 1)
                x1 += t
                x2 -= t
            if col != 0:
                t = theta_HH * (2 * EQ[image_id, row, col-1] - 1)
                x1 += t
                x2 -= t
            if row != 27:
                t = theta_HH * (2 * EQ[image_id, row+1, col] - 1)
                x1 += t
                x2 -= t
            if col != 27:
                t = theta_HH * (2 * EQ[image_id, row, col+1] - 1)
                x1 += t
                x2 -= t

            X = noiseX[image_id, row, col]

            new_pi = np.exp(x1 + (theta_HX * X)) / (np.exp(x1 + (theta_HX * X)) + np.exp(x2 + -theta_HX * X))
            EQ[image_id, row, col] = new_pi


# output energy list
energy = np.array(energy_list).reshape(10, 10).T
np.savetxt("energyResult.csv", energy, delimiter=',')

# get hidden image
hidden_img = np.zeros((28, 280))

for img_id in range(len(EQ)):
    hidden_img[:, img_id * 28: (img_id + 1) * 28] = EQ[img_id]
show(hidden_img)
map_img = np.where(hidden_img >= 0.5, 1, 0)
show(map_img)
np.savetxt("denoised.csv", map_img, fmt='%d', delimiter=',')
np.savetxt("energy.csv", energy[0:2, 0:2], delimiter=',')

















