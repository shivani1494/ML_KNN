import numpy as np

class DecisionTree:
    def __init__(sf, filenames):
        sf._label_index = 0
        sf._num_features = 0
        sf._input_data = load(filenames[0])

        sf._feature_name = sf.feature_names("hw3features.txt")
        sf._rootnode = None
        sf.train()

    def load(sf,filename):
        print("Loading %s ..." %filename)

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

        print(filename,"loaded : Dim",data.shape)
        return data

    def train(sf):
        print("Start training")
        sf._rootnode = Node(None, sf._input_data, None)
        current_node = sf._rootnode
        while True:
            print(current_node)
            if not(current_node.pure):
                feature, threshold = sf.maximize_IG(current_node.data)
                split(current_node, feature, threshold)

                current_node = current_node.leftchild

            elif current_node.parent != None:
                if current_node.parent.rightchild != current_node:
                    current_node = current_node.rightchild
                elif current_node.parent.parent != None:
                    pass

    def split(sf, node, feature, threshold):
        # Creates two branches if the feature was not already used
        ind_greater, ind_smaller = sf.examples_threshold(node.data,feature,threshold)
        if not(sf.feature_in_branch(node,feature)):
            node_smaller = Node(node, node.data[ind_smaller], feature, threshold, "left")
            node_greater = Node(node, node.data[ind_greater], feature, threshold, "right")
        else :
            pass # Choose label of the majority


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
        num_examples = data.shape[0]

        return -nonzero/sf._num_examples*np.log(nonzero/sf._num_examples) -(sf._num_examples-nonzero)/sf._num_examples*np.log((sf._num_examples-nonzero)/sf._num_examples)

    def conditional_entropy(sf, X, feature, threshold):
        num_examples = data.shape[0]
        ind_greater, ind_smaller = sf.examples_greater_than_threshold(X, feature, threshold)

        greater = entropy(X[ind_greater,:]) # H(X,Z = 0)
        smaller = entropy(X[ind_smaller,:]) # H(X,Z = 1)

        prob_greater = len(greater)/sf._num_examples # P(Z=0)
        prob_smaller = len(smaller)/sf._num_examples # P(Z=1)

        return prob_greater * greater + prob_smaller * smaller

    def examples_threshold(sf, X, feature, threshold):
        greater =  X[:,feature] > threshold # samples Z = 0
        smaller = X[:,feature] <= threshold # samples Z = 1
        return np.where(greater), np.where(smaller)

    def get_threshold(sf,X,feature):
        return np.mean(X[:,feature],axis=0)

    def feature_in_branch(sf, node, feature):
        # Checks if a feature is already being used in the branch
        if (node.parent != None):
            if node.parent.feature == feature:
                return False
            else :
                return feature_in_branch(node.parent, feature)
        else :
            return True

    def feature_names(sf, filename):
        filehandle = open(filenam, "r")

    class Node:
        def __init__(sf, parent, data, feature, threshold, pos):
            sf.rightchild = None
            sf.leftchild = None
            sf.data = data
            sf.label = None
            sf.feature = feature
            sf.threshold = threshold
            sf.pure = sf.check_purity()
            if parent:
                sf.parent = parent
                if pos == "left":
                    sf.parent.leftchild = sf
                elif pos == "right":
                    sf.parent.rightchild = sf

        def __str__(sf):
            p = sf.parent.feature if p.parent else None
            return "f<%i> p<%i>" %(f,p)

        def check_purity():
            same_label_count = np.count_nonzero(sf.data[:,sf._label_index])
            if same_label_count == 0:
                sf.label = 1
            elif same_label_count == sf.data.shape[0]:
                sf.label = 0
            else :
                return False
            return True

if __name__ == "__main__":
    tree = DecisionTree()