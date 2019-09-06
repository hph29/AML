import numpy as np
from numpy import genfromtxt

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from math import log
from math import exp
from scipy.misc import logsumexp

from sklearn.mixture import GaussianMixture

class EMTopicModel:

    def __init__(self, file):
        with open(file) as f:
            self.num_doc = int(f.readline().strip())
            self.len_vocabulary = int(f.readline().strip())
            self.num_words = int(f.readline().strip())
            self.num_topic = 30
            self.topic_prob = np.zeros(self.num_topic)
            self.words_prob = np.zeros((self.num_topic, self.len_vocabulary))
        self.data = genfromtxt(file, skip_header=3, delimiter=' ', dtype=int)

    def vectorize_init_features(self):
        self.features = []
        self.feature_pairs = []
        for i in range(self.num_doc):
            # init input feature for each document there is a list of word count for each word in vocabulary list
            feature = np.zeros(self.len_vocabulary)
            entry_for_document_i = self.data[np.where(self.data[:, 0] == i + 1)]
            # assign word count to input feature
            feature[entry_for_document_i[:, 1] - 1] = entry_for_document_i[:, 2]
            entry_for_document_i[:, 1] = entry_for_document_i[:, 1] - 1
            self.feature_pairs.append(list(zip(entry_for_document_i[:, 1], entry_for_document_i[:, 2])))
            self.features.append(feature)
            #print(feature.tolist())
        # print(self.features)
        # print(self.feature_pairs)
        # print("********************************")


    def init_kmeans(self):
        self.k_means_label = KMeans(n_clusters=self.num_topic).fit_predict(self.features)
        # print(list(self.k_means_label))


    def init_em_params(self):
        for i in range(self.num_topic):
            self.topic_prob[i] = list(self.k_means_label).count(i) / self.num_topic
            doc_ids = [a for a, b in enumerate(self.k_means_label) if b == i]
            word_vector = np.array([1 / self.num_words] * self.len_vocabulary)
            [np.add(word_vector, self.features[doc_id]) for doc_id in doc_ids]
            self.words_prob[i] = word_vector / np.sum(word_vector)
        # print(self.topic_prob)
        # print(self.words_prob)

    def em(self):
        likelihood = float('-inf')
        while True:
            w_i_j = list()  # for each document, we store numTopics values in this
            for feature_pair in self.feature_pairs:
                log_vec = list()
                for j in range(self.num_topic):
                    words_prob_delta = self.words_prob[j]
                    m_log = log(self.topic_prob[j])
                    for word_id, counts in feature_pair:
                        m_log += counts * log(words_prob_delta[word_id])
                    log_vec.append(m_log)

                log_vec = list(map(lambda x: x - max(log_vec), log_vec))
                log_vec = list(map(lambda x: x - logsumexp(log_vec), log_vec))

                w_j = [exp(x) for x in log_vec]
                w_i_j.append(w_j)

            # calculate delta likelihood, and check for convergence
            new_likelihood = 0.
            for i in range(len(self.features)):
                for j in range(self.num_topic):
                    words_prob = self.words_prob[j]
                    inner = log(self.topic_prob[j])
                    for word_id, counts in self.feature_pairs[i]:
                        inner += counts * log(words_prob[word_id])

                    new_likelihood += inner * w_i_j[i][j]

            print('Likelihood= ', new_likelihood)
            # convergence criteria
            if new_likelihood < likelihood or (new_likelihood - likelihood) / abs(new_likelihood) < 0.000001:
                return

            likelihood = new_likelihood

            # update the parameters
            for j in range(self.num_topic):
                # pie-j
                self.topic_prob[j] = sum(list(map(lambda x: x[j], w_i_j))) / self.num_doc

                # p-j
                num = np.array([1/self.num_words] * self.len_vocabulary)  # smooths data
                denom = 0
                for i in range(len(self.features)):
                    num = np.add(num, self.features[i] * w_i_j[i][j])
                    denom += sum(self.features[i]) * w_i_j[i][j]

                self.words_prob[j] = num / denom

    def run(self):
        self.vectorize_init_features()
        self.init_kmeans()
        self.init_em_params()
        self.em()

        print(self.topic_prob)
        print(sum(self.topic_prob))
        for w in self.words_prob:
            s = sorted(list(zip(w, list(range(self.len_vocabulary)))), reverse=True)
            print(s[0:10])
            print([x[1] for x in s[0:10]])
            print(sum(w))
            print('\n')

if __name__ == '__main__':

    EMTopicModel("docword.nips.txt").run()

