from PIL import Image
import numpy as np


class ImageSegmentation:

    def __init__(self, image_file):
        image = Image.open(image_file)
        self.filename = image_file
        self.num_row = image.size[1]
        self.num_col = image.size[0]
        self.data = np.array(image.getdata())
        self.identifier = 0

    def run(self):
        for num_cluster in [10, 20, 50]:
            self.em(num_cluster)

    def run_five_times(self):
        for i in range(5):
            self.identifier = i
            self.em(20)


    def guss(self, a, b):
        distances = np.linalg.norm((a - b), axis = 1)**2
        return np.exp(-0.5*distances)

    def em(self, num_cluster):
        row, col = self.data.shape

        # init result matrix
        matrix = np.ones(shape=self.data.shape)
        std0 = np.std(self.data[:, 0])/2
        std1 = np.std(self.data[:, 1])/2
        std2 = np.std(self.data[:, 2])/2
        matrix[:, 0] = self.data[:, 0]/std0
        matrix[:, 1] = self.data[:, 1]/std1
        matrix[:, 2] = self.data[:, 2]/std2

        #intialize the clusters
        pi = np.ones(num_cluster) / num_cluster
        p = np.zeros(shape=(col, num_cluster))
        init_center = np.random.randint(0, row, num_cluster)

        for index, center in enumerate(init_center):
            p[:, index] = matrix[center]

        w = np.ones(shape=(row, num_cluster))
        num_iteration = 50
        for i in range(num_iteration):

            # e
            for j in range(num_cluster):
                w[:, j] = pi[j] * np.exp(-1/2 * np.linalg.norm((matrix - p[:, j]), axis=1)**2)
            w = (w.T / w.sum(axis=1)).T

            # m
            p = np.dot(matrix.T, w)
            p = p / np.sum(w, axis=0)
            pi = np.sum(w, axis=0) / row

        claszz = np.argmax(w, axis=1)
        for i in range(num_cluster):
            idx = np.where(claszz == i)
            matrix[idx, :] = p[:, i]

        matrix[:, 0] = matrix[:, 0] * std0
        matrix[:, 1] = matrix[:, 1] * std1
        matrix[:, 2] = matrix[:, 2] * std2
        matrix = matrix.astype(int)

        out = Image.new("RGB", (self.num_col, self.num_row))
        for i in range(self.num_row):
            for j in range(self.num_col):
                out.putpixel((j, i), tuple(matrix[i*self.num_col+j]))
        name = self.filename.split(".")[0] + "_" + str(num_cluster) + str(self.identifier) + "." + self.filename.split(".")[1]
        out.save(name)

if __name__ == '__main__':
    ImageSegmentation("smallsunset.jpg").run()
    ImageSegmentation("smallstrelitzia.jpg").run()
    ImageSegmentation("RobertMixed03.jpg").run()

    ImageSegmentation("smallsunset.jpg").run_five_times()

