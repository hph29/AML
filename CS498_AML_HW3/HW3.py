from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

dict_by_category = {}
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

###
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def group_by_category(dict):
    for i in range(len(dict[b'labels'])):
        category = dict[b'labels'][i]
        data = dict[b'data'][i]
        if category not in dict_by_category:
            dict_by_category[category] = [data]
        else:
            dict_by_category[category].append(data)


def load_data():
    list = [unpickle("cifar-10-batches-py/data_batch_1"),
            unpickle("cifar-10-batches-py/data_batch_2"),
            unpickle("cifar-10-batches-py/data_batch_3"),
            unpickle("cifar-10-batches-py/data_batch_4"),
            unpickle("cifar-10-batches-py/test_batch")]

    for i in range(len(list)):
        group_by_category(list[i])

    assert reduce((lambda a, b: a + b), [len(dict_by_category[i]) for i in range(10)]) == 50000


def p1_compute_mean():
    mean = np.mean(dict_by_category[0], axis=0)
    expected = int(reduce((lambda x, y: x + y), map(lambda x: x[0] / len(dict_by_category[0]), dict_by_category[0])))
    actual = int(mean[0])
    assert expected == actual
    mean_image = []
    for i in range(len(dict_by_category)):
        mean = np.mean(dict_by_category[i], axis=0)
        mean_image.append(mean)
        # print("Mean image for category %s is \n%s" % (i, np.mean(dict_by_category[i], axis=0)))
    return mean_image


def p1_compute_error():
    mean = p1_compute_mean()
    to_plot = []
    for i in range(len(dict_by_category)):
        centered = dict_by_category[i] - mean[i]
        pca = PCA(n_components=20)
        pca.fit(centered)
        error = np.var(dict_by_category[0], ddof=1, axis=0).sum() - pca.explained_variance_.sum()
        to_plot.append(error)
        # print("Error for category %s is \n%s" % (i, error))

    # draw bar chart for error
    plt.bar(np.arange(len(dict_by_category)), to_plot, align='center')
    plt.xticks(np.arange(len(dict_by_category)), [labels[i] for i in range(10)])
    plt.ylabel('error (sum of dropped eigenvalues)')
    plt.title('Error of each category using the first 20 principal components against the category')
    plt.show()


def p2_plot():
    mean = p1_compute_mean()

    distance_matrix = cdist(np.array(mean), np.array(mean))
    print('Distance_Matrix: ')
    print(distance_matrix)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(distance_matrix)
    c = ['purple', 'cyan', 'blue', 'green', 'yellow', 'red', 'orange', 'brown', 'black', 'gray']
    [plt.scatter(proj[i, 0], proj[i, 1], label=labels[i], c=c[i]) for i in range(10)]
    plt.legend()
    plt.show()


def p3_plot():
    mean = p1_compute_mean()
    error = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            pca = PCA(n_components=20)
            # get 20 pc for b
            centered_b = dict_by_category[j] - mean[j]
            pca.fit(centered_b)
            pc_b = pca.n_components_
            a_b = mean[i] + (np.dot(np.transpose(pc_b), (dict_by_category[i] - mean[i])) * pc_b)
            centered_a = dict_by_category[i] - mean[i]
            pca.fit(centered_a)
            pc_a = pca.n_components_
            b_a = mean[j] + (np.dot(np.transpose(pc_a), (dict_by_category[j] - mean[j])) * pc_a)

            pca.fit(a_b)
            error_a_b = np.var(dict_by_category[i], ddof=1, axis=0).sum() - pca.explained_variance_.sum()
            pca.fit(b_a)
            error_b_a = np.var(dict_by_category[j], ddof=1, axis=0).sum() - pca.explained_variance_.sum()

            error[i][j] = 1/2 * (error_a_b + error_b_a)
    print(error.shape)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(error)
    c = ['purple', 'cyan', 'blue', 'green', 'yellow', 'red', 'orange', 'brown', 'black', 'gray']
    [plt.scatter(proj[i, 0], proj[i, 1], label=labels[i], c=c[i]) for i in range(10)]
    plt.legend()
    plt.show()

if __name__ == "__main__":
    load_data()
    p1_compute_mean()
    p1_compute_error()
    p2_plot()
    p3_plot()
