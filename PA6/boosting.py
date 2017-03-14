import numpy as np
from copy import copy

class Booster:
    MODEL_OCCURS = "occurs"
    MODEL_NOT_OCCURS = "not_occurs"
    def __init__(self):
        self._dict = self.read_dic("hw6dictionary.txt")
        self._minimum_error_learners = [] # Number of boosting rounds, consisting of [alpha/error,word_index,model(h_occurs/h_not_occurs)]
    def read_data(self,filename):
        # reads and returns test train and label data

        print("Loading %s ..." % filename)

        filehandle = open(filename, "r")
        line = filehandle.readline()
        data = []

        while line != "":
            line = line.split()
            data += [line]
            line = filehandle.readline()

        data = np.array(data)

        print(filename, "loaded : Dim", data.shape)
        return data[:, :-1].astype(int), data[:, -1].astype(int)
    def read_dic(self, filename):
        # Creates dictionary of words
        dict = {}

        filehandle = open(filename,"r")
        file = filehandle.read()
        file = file.split("\n")
        for i in range(len(file)):
            if len(file[i])>0:
                dict[i] = file[i]
        return dict
    def h_occurs(self, col):
        # 1 if word occurs in dict, else -1
        col[np.where(col == 0)] = -1

        return col
    def h_not_occurs(self, col):
        # 1 if word does not occur in dict, else -1
        col[np.where(col==1)] = -1
        col[np.where(col==0)] = 1
        return col
    def indicator(self,h, label):
        # 1 if classified incorrectly, 0 otherwise
        ind_equal = np.where(h == label)[0]
        ind_not_equal = np.where(h != label)[0]
        for i in ind_not_equal:
            h[i] = 1
        for c in ind_equal:
            h[c] = 0
        return h
    def err(self,indicator_col,weights):
        return np.dot(indicator_col,weights.T)
    def alpha(self,error):
        return 1/2 * np.log((1-error)/error)
    def calc_weights(self,prev_weights,alpha,h,label, normalization):
        weights = prev_weights * np.exp(-alpha*label*h) / normalization
        return weights
    def normalization(self,indicator_h, alpha, weights):
        correct = np.where(indicator_h==0)[0]
        incorrect = np.where(indicator_h==1)[0]
        norm = 0
        for c in correct:
            norm += np.exp(-alpha) * weights[c]
        for i in incorrect:
            norm += np.exp(alpha) * weights[i]
        return norm
    def build_weak_learners(self, data, weights,label):
        weak_learners_error_occurs = []  # 4003 learners
        weak_learners_error_not_occurs = []
        # For each word/column
        for c in range(data.shape[1]):
            # Build two weak learner classifiers
            h_occurs = self.h_occurs(copy(data[:,c]))
            h_not_occurs = self.h_not_occurs(copy(data[:,c]))
            indicator_col_occurs = self.indicator(copy(h_occurs),label)
            indicator_col_not_occurs = self.indicator(copy(h_not_occurs),label)

            # Compute and store the error
            weak_learners_error_occurs += [self.err(indicator_col_occurs,weights)]
            weak_learners_error_not_occurs += [self.err(indicator_col_not_occurs,weights)]

        return weak_learners_error_occurs+weak_learners_error_not_occurs
    def boost(self, times, data, label):
        weights = np.ones(data.shape[0])/data.shape[0]
        for t in range(times):
            # Build all the classifiers (4003 * 2) models and keep the one with the minimum error

            weak_learners_error = self.build_weak_learners(data, weights, label)
            min_learner_ind = weak_learners_error.index(min(weak_learners_error))
            if (min_learner_ind >= data.shape[1]):
                # min learner in not_occurs model
                alpha = self.alpha(weak_learners_error[min_learner_ind])
                min_learner_ind -= data.shape[1]
                self._minimum_error_learners += [[alpha, min_learner_ind, Booster.MODEL_NOT_OCCURS]]
                h = self.h_not_occurs(copy(data[:,min_learner_ind]))
                indicator_h = self.indicator(copy(h),label)
                norm = self.normalization(indicator_h,alpha,weights)
                weights = self.calc_weights(weights, alpha, h,label, norm)
            else:
                alpha = self.alpha(weak_learners_error[min_learner_ind])
                self._minimum_error_learners += [[alpha, min_learner_ind, Booster.MODEL_OCCURS]]
                h = self.h_occurs(copy(data[:,min_learner_ind]))
                indicator_h = self.indicator(copy(h), label)
                norm = self.normalization(indicator_h, alpha, weights)
                weights = self.calc_weights(weights,alpha,h,label,norm)
    def test(self, data, label):
        final_h = np.zeros(data.shape[0])
        for learner_data in self._minimum_error_learners:
            alpha = learner_data[0]
            word_ind = learner_data[1]
            if (learner_data[2] == Booster.MODEL_OCCURS):
                h = self.h_occurs(copy(data[:,word_ind]))
            elif (learner_data[2] == Booster.MODEL_NOT_OCCURS):
                h = self.h_not_occurs(copy(data[:,word_ind]))
            final_h += h * alpha
        return (label.shape[0] - np.count_nonzero(np.sign(final_h) == label))/label.shape[0]
    def weak_learners(self, num_rounds):
        words = []
        for l in range(len(self._minimum_error_learners)):
            if l == num_rounds:
                break
            words += [self._dict[self._minimum_error_learners[l][1]]]
        return words

if __name__ == "__main__":
    b = Booster()
    input_data, input_label = b.read_data("hw6train.txt")
    test_data, test_label = b.read_data("hw6test.txt")
    b.boost(10, input_data, input_label)
    print(b.test(input_data, input_label))
    print(b.weak_learners(10))
