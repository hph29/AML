from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq, kmeans
import os
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import random
import numpy as np

def hw4_p1_load_data():
    array = np.genfromtxt('data.csv', delimiter=',', dtype=str)
    return array[:,0], array[1:,1:].astype(float)


def hw4_p1_part1():
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    label, feature = hw4_p1_load_data()

    for s in ['single', 'complete', 'average']:

        linkage_matrix = linkage(feature, s)
        dendrogram(
            linkage_matrix,
            truncate_mode="lastp",  # show only the last p merged clusters
            p=40,  # show only the last p merged clusters
            show_leaf_counts=True,  # numbers in brackets are counts, others idx
            leaf_rotation=60.,
            leaf_font_size=8.,
            show_contracted=True,  # to get a distribution impression in truncated branches
            labels=label
        )
        plt.title('Hierarchical Clustering Dendrogram For Method ' + s)
        plt.show()


def hw4_p1_part2():
    label, feature = hw4_p1_load_data()
    x = []
    y = []
    for i in range(1, 26):
        model = KMeans(n_clusters=i, random_state=1).fit(feature)
        interia = model.inertia_
        x.append(i)
        y.append(interia)
    plt.plot(x, y)
    plt.show()


def hw4_p2(k, size):
    feature_pre_row = 3 * size
    whole_data = np.empty((0, feature_pre_row))
    # recursive list files in folder
    files = [os.path.join(root, f) for root, _, files in os.walk('HMP_Dataset') for f in files]
    # filter out unrelated files
    files = list(filter(filter_out_unrelated_file, files))
    # shuffle file paths
    random.Random(2).shuffle(files)
    num_file_for_train = int(0.8 * len(files))
    # split data set into train and test
    train_files = files[:num_file_for_train]
    test_files = files[num_file_for_train:]

    # read each file in and generate data set
    # convert file into segments and build k cluster
    for file in files:
        array = np.loadtxt(file)
        # remove extra rows to crave shape of data set
        extra = array.shape[0] % size
        array = array[:-extra]
        array = array.reshape(-1, feature_pre_row)
        # append single file
        whole_data = np.append(whole_data, array, axis=0)
    # apply kmeans to cluster data
    codebook, distortion = kmeans(whole_data, k)
    # create histogram for random forest to train data
    train_xx = []
    train_yy = []
    for train_file in train_files:
        array = np.loadtxt(train_file)
        extra = array.shape[0] % size
        array = array[:-extra]
        array = array.reshape(-1, feature_pre_row)
        # vector quantize train_file
        closest_centroid, distortion = vq(array, codebook)
        # create histogram
        hist = [0] * k
        for i in range(len(closest_centroid)):
            hist[closest_centroid[i]] = hist[closest_centroid[i]] + 1
        train_xx.append(hist)
        # add label info to array
        category = train_file.split('\\')[1]
        train_yy.append(category)

    test_xx = []
    test_yy = []
    for test_file in test_files:
        array = np.loadtxt(test_file)
        extra = array.shape[0] % size
        array = array[:-extra]
        array = array.reshape(-1, feature_pre_row)
        # vector quantize test_file
        closest_centroid, distortion = vq(array, codebook)
        # create histogram
        hist = [0] * k
        for i in range(len(closest_centroid)):
            hist[closest_centroid[i]] = hist[closest_centroid[i]] + 1
        test_xx.append(hist)
        category = test_file.split('\\')[1]
        # add label info to array
        test_yy.append(category)

    # convert label to number for training model using random forest
    train_yy = list(map(convert_category_to_number, train_yy))
    test_yy = list(map(convert_category_to_number, test_yy))

    # init random forest model
    reg = RandomForestClassifier()
    # train model
    reg.fit(train_xx, train_yy)
    # predict test data set
    predict = reg.predict(test_xx)
    # print confusion matrix
    print("confusion_matrix for k = %s and size = %s:" % (k, size))
    print(confusion_matrix(test_yy, predict))
    # calculate error rate for the predict result
    correct_num = 0
    for i in range(len(predict)):
        if int(round(predict[i])) == test_yy[i]:
            correct_num = correct_num + 1
    error_rate = correct_num / len(predict)
    print("error rate for k = %s and size = %s: %s" % (k, size, str(error_rate)))

    return error_rate


def convert_category_to_number(x):
    if "Brush_teeth" == x:
        return 1
    elif "Climb_stairs" == x:
        return 2
    elif "Comb_hair" == x:
        return 3
    elif "Descend_stairs" == x:
        return 4
    elif "Drink_glass" == x:
        return 5
    elif "Eat_meat" == x:
        return 6
    elif "Eat_soup" == x:
        return 7
    elif "Getup_bed" == x:
        return 8
    elif "Liedown_bed" == x:
        return 9
    elif "Pour_water" == x:
        return 10
    elif "Sitdown_chair" == x:
        return 11
    elif "Standup_chair" == x:
        return 12
    elif "Use_telephone" == x:
        return 13
    elif "Walk" == x:
        return 14

def filter_out_unrelated_file(f):
    return "MODEL" not in f and "README" not in f and "MANUAL" not in f and "display" not in f

def hw4_p2_plot():
    k_dependent_error = []
    k_range = range(150,  150 * 5 + 1, 150)
    size_range = range(16, 16 * 5 + 1, 16)
    for k in k_range:
        size_dependent_error = []
        for size in size_range:
            size_dependent_error.append(hw4_p2(k, size))
        k_dependent_error.append(size_dependent_error)
    plt.plot(list(size_range), k_dependent_error[0], 'r',
             list(size_range), k_dependent_error[1], 'b',
             list(size_range), k_dependent_error[2], 'g',
             list(size_range), k_dependent_error[3], 'm',
             list(size_range), k_dependent_error[4], 'y')
    plt.show()



if __name__ == "__main__":
    hw4_p1_part1()
    hw4_p1_part2()
    hw4_p2_plot()

