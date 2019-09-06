from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt


class EMTopicModel:

    def __init__(self, file, voc_file):
        with open(file) as f:
            self.num_doc = int(f.readline().strip())
            self.num_vocabulary = int(f.readline().strip())
            self.num_words = int(f.readline().strip())

        self.vocabulary_list = genfromtxt(voc_file, delimiter=' ', dtype=str)

        self.num_topic = 30
        # int feature with size (num_doc, num_voc)
        self.feature = np.zeros(shape=(self.num_doc, self.num_vocabulary))
        data = genfromtxt(file, skip_header=3, delimiter=' ', dtype=int)
        # populate data
        for row in data:
            doc_id = row[0] - 1
            word_id = row[1] - 1
            word_count = row[2]
            self.feature[doc_id][word_id] = word_count

    def init_em_parameter(self):
        # init prob (1, 30)
        self.pi = np.ones(self.num_topic) / self.num_topic

        # (12419 x 30)
        self.p = np.zeros(shape=(self.num_vocabulary, self.num_topic))
        # (1 x 30)
        init_center = np.random.randint(0, self.num_doc - 1, self.num_topic)

        for index, doc_id in enumerate(init_center):
            # (30)        word count for each doc        count of all words within that doc_id
            self.p[:, index] = (self.feature[doc_id, :] + 1) / (np.sum(self.feature[doc_id, :]) + self.num_vocabulary)

        self.w = np.ones(shape=(self.num_doc, self.num_topic))

    def em(self):
        num_iteration = 200

        for i in range(num_iteration):
            # e
            self.w = np.dot(self.feature, np.log(self.p)) + np.log(self.pi)
            self.w = (self.w.T - self.w.max(axis=1)).T
            self.w = (self.w.T - np.log(np.sum(np.exp(self.w), axis=1))).T
            self.w = np.exp(self.w)

            # m
            self.pi = np.sum(self.w, axis=0) / self.num_doc
            self.p = np.dot(self.feature.T, self.w) + 1.0 / self.num_vocabulary
            sum = np.sum(self.p, axis=0)
            self.p = self.p / sum

        self.clazz = np.argmax(self.w, axis=1)

    def plot(self):
        prob_matrix = np.zeros(shape=self.num_topic)
        for cluster in self.clazz:
            prob_matrix[cluster] += 1
        prob_matrix /= prob_matrix.sum()

        plt.figure()
        plt.title('Probability of topic being selected')
        plt.xlabel('Topic Id')
        plt.ylabel('Probability')
        plt.bar(range(1, 31), prob_matrix)
        plt.show()

    def list_common_word(self):
        cluster_vec = np.zeros(shape=(self.num_topic, self.num_vocabulary))
        for index, cluster in enumerate(self.clazz):
            cluster_vec[cluster, :] += self.feature[index]

        for i in range(self.num_topic):
            common_word_pre_topic = []
            for j in range(10):
                word_id = np.argmax(cluster_vec[i, :])
                common_word_pre_topic.append(self.vocabulary_list[word_id])
                cluster_vec[i, word_id] = 0
            print("Topic %s: " % i, common_word_pre_topic)

    def run(self):
        self.init_em_parameter()
        self.em()
        self.plot()
        self.list_common_word()

if __name__ == '__main__':

    EMTopicModel("docword.nips.txt", "vocab.nips.txt").run()

