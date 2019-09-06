import mnist
from matplotlib import pyplot as plt
import numpy as np

def binarize(images):
    for image in images:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] < 128:
                    image[i][j] = 0
                else:
                    image[i][j] = 1

def main():
    images = mnist.train_images()[0:20]
    image_list = [images[i] for i in range(len(images))]
    #print(images[0])
    binarize(image_list)
    #print("****************************\n", images[0])
    #draw(image_list[0])
    #print(type(images))
    #print(images.shape)
    add_noise(image_list)
    #draw(image_list[1])
    boltzman_machine()

def add_noise(images):
    noises = np.genfromtxt('NoiseCoordinates.csv', delimiter=',', skip_header=1, usecols=range(1, 16))
    for i in range(noises.shape[0]):
        if i % 2 == 1:
            continue
        image_id = int(i / 2)
        for j in range(noises.shape[1]):
            #print("image_id", image_id)
            #print("x", noises[i, j])
            #print("y", noises[i+1][j])
            #print("test", noises[i+1, j])
            value = images[image_id][int(noises[i, j]), int(noises[i+1, j])]
            images[image_id][int(noises[i, j]), int(noises[i+1, j])] = 1 - value


def boltzman_machine():

    iter = 10
    theta_H = 0.2
    theta_X = 0.8
    e = 10e-10

    Q = np.genfromtxt('InitialParametersModel.csv', delimiter=',')

    updateOrder = np.genfromtxt('UpdateOrderCoordinates.csv', delimiter=',',
                                skip_header=1, usecols=range(1, 1 + 28*28))

    E_Q = list()
    for i in range(20):
        E_Q.append(Q)
    print(E_Q)

    energy = []

    for i in range(iter):

        for img in         







def draw(image):
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':

    main()
