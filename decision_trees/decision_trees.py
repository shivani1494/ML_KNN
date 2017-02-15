import numpy as np

class decision_tree:
    def __init__(sf, filenames):
        sf._input_data = load(filenames[0])
        sf._label_index = 0
        sf._num_features = 0
        sf.__rootnode = None
        sf._num_examples = 0

    def load(sf,filename):
        filehandle = open(filename, "r")
        line = filehandle.readline()
        data = []
        while line != "":
            line = line.split()
            data += [line]
            line = filehandle.readline()
        
        data = np.array(data)
        data = data.astype(float)

        sf._label_index = data.shape[1]-1
        sf._num_features = sf._label_index
        sf._num_examples = data.shape[0]

        return data

    def train(sf):
        sf._rootnode = sf_input_data
        feature, threshold = sf.maximize_IG(X)


    def maximize_IG(sf,X):
        max_ig = None
        feature = 0
        threshold = 0
        for f in range(sf._num_features):
            t = sf.get_threshold(X,f)
            ig = sf.information_gain(X,f,t)
            if max_ig == None or ig > best_ig:
                best_ig = ig
                feature = f
                threshold = t
        return feature, threshold

    def information_gain(sf, X, feature, threshold):
        # uncertainty (entropy) reduced in data X when knowing Z
        return sf.entropy(X) - sf.conditional_entropy(X,feature,threshold)


    def entropy(sf, X):
        nonzero = np.count_nonzero(X[:,sf._label_index])

        return -nonzero/sf._num_examples*np.log(nonzero/sf._num_examples) -(sf._num_examples-nonzero)/sf._num_examples*np.log((sf._num_examples-nonzero)/sf._num_examples)

    def conditional_entropy(sf, X, feature, threshold):
        ind_greater, ind_smaller = sf.examples_greater_than_threshold(X, feature, threshold)

        greater = entropy(X[ind_greater,:]) # H(X,Z = 0)
        smaller = entropy(X[ind_smaller,:]) # H(X,Z = 1)

        prob_greater = len(greater)/sf._num_examples # P(Z=0)
        prob_smaller = len(smaller)/sf._num_examples # P(Z=1)

        return prob_greater * greater + prob_smaller * smaller

    def examples_threshold(sf, X, feature, threshold):
        greater =  X[:,feature] >= threshold # samples Z = 0
        smaller = X[:,feature] <= threshold # samples Z = 1
        return np.where(greater), np.where(smaller)

    def get_threshold(sf,X,feature):
        return np.mean(X[:,feature],axis=0)

class Node:
    def __init__(sf,data):
        sf.data = data
        sf.pure = sf.check_purity()

    def 