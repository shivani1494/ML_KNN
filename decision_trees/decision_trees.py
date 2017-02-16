import numpy as np

class DecisionTree:
    def __init__(sf, filenames, input_data=None):
        sf._num_features = 0
        if input_data != None:
            sf._input_data = input_data
            sf._num_features = input_data.shape[1]-1
        else:
            sf._input_data = sf.load(filenames[0])
            sf._feature_names = sf.get_feature_names("hw3features.txt")
        sf._rootnode = None

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

        print(filename,"loaded : Dim",data.shape)
        return data

    def train(sf):
        print("Start training")
        sf._rootnode = Node(sf._input_data, None, None)
        current_node = sf._rootnode
        iteration = 0
        while True:
            print("iter:",iteration); iteration += 1;
            if not(current_node.pure):
                print("Node Not Pure")
                # Set current node to pure
                # (we are not going to compute the tree for this node again)
                current_node.pure = True
                feature, threshold = sf.maximize_IG(current_node)

                if feature == None and threshold == None:
                    print("Features Used in the branch,",end=" ")
                    # All the features used in that branch
                    # "Actual" label of all the data in current_node is the majority label
                    # (Node Purity is checked when created)

                    label_index = current_node.data.shape[1]-1
                    nonzero = np.count_nonzero(current_node.data[:,label_index])
                    if (nonzero > current_node.data[:,label_index].shape[0]-nonzero):
                        # Actual label for right branch
                        current_node.label_if_less = 1
                        print("Label Right",current_node.label_if_less)
                    else:
                        # Actual label for left branch
                        current_node.label_if_less = 0
                        print("Label Left",current_node.label_if_less)

                else:
                    print("Features Not used")
                    # Features not being used
                    current_node.threshold = threshold
                    current_node.feature = feature

                    sf.split(current_node, feature, threshold)
                    print(current_node)

                    print("Building Left Branch")
                    # Build Left Branch
                    current_node = current_node.leftchild
                

            elif current_node.parent:

                if current_node.parent.rightchild and current_node.parent.rightchild != current_node:

                    # Build sibling (rightchild) branch
                    current_node = current_node.parent.rightchild
                elif current_node.parent != None:
                    # Go to grandfather
                    current_node = current_node.parent

            else:
                break

    def split(sf, node, feature, threshold):
        # Creates two branches
        label = node.data.shape[1]-1
        ind_greater, ind_smaller = sf.examples_threshold(node.data,feature,threshold)
        
        node_smaller = Node(node.data[ind_smaller,:], node, "left")
        node_greater = Node(node.data[ind_greater,:], node, "right")


    def maximize_IG(sf,node):
        X = node.data
        max_ig = None
        feature = None
        threshold = None
        for f in range(sf._num_features):
            # Checks if the feature is already in use in the branch
            if not(sf.feature_in_branch(node,f)):
                print("# F",f)
                t = sf.get_threshold(X,f)
                ig = sf.information_gain(X,f,t)
                if max_ig == None or ig > best_ig:
                    best_ig = ig
                    feature = f
                    threshold = t
        print("F Chosen,",feature)
        return feature, threshold

    def information_gain(sf, X, feature, threshold):
        # uncertainty (entropy) reduced in data X when knowing Z
        return sf.entropy(X) - sf.conditional_entropy(X,feature,threshold)


    def entropy(sf, X):
        # Log(0) = -Inf
        print("Here",X.shape[0])
        print(X)
        label_index = X.shape[1]-1
        nonzero = np.count_nonzero(X[:, label_index])
        num_examples = X.shape[0]

        # Division by 0 otherwise
        if nonzero == num_examples or nonzero == 0:
            return 0

        return -nonzero/num_examples*np.log(nonzero/num_examples) -(num_examples-nonzero)/num_examples*np.log((num_examples-nonzero)/num_examples)

    def conditional_entropy(sf, X, feature, threshold):
        num_examples = X.shape[0]
        ind_greater, ind_smaller = sf.examples_threshold(X, feature, threshold)

        greater_entropy = sf.entropy(X[ind_greater,:]) # H(X,Z = 0)
        smaller_entropy = sf.entropy(X[ind_smaller,:]) # H(X,Z = 1)

        prob_greater = X[ind_greater,:].shape[0]/num_examples # P(Z=0)
        prob_smaller = X[ind_smaller,:].shape[0]/num_examples # P(Z=1)

        return prob_greater * greater_entropy + prob_smaller * smaller_entropy

    def examples_threshold(sf, X, feature, threshold):
        greater =  X[:,feature] > threshold # samples Z = 0
        smaller = X[:,feature] <= threshold # samples Z = 1
        return np.where(greater)[0], np.where(smaller)[0]

    def get_threshold(sf,X,feature):
        return np.mean(X[:,feature],axis=0)

    def feature_in_branch(sf, node, feature):
        # Checks if a feature is already being used in the branch
        if (node.parent != None):
            if node.parent.feature == feature:
                return True
            else :
                return sf.feature_in_branch(node.parent, feature)
        else :
            return False

    def get_feature_names(sf, filename):
        filehandle = open(filename, "r")
        feature_names = {}
        index = 0
        for line in filehandle:
            feature_names[index] = line
            index += 1
        sf._num_features = index

    def print_final_structure(sf, node = None, first = True):
        if first and sf._rootnode:
            print("***START***")
            node = sf._rootnode
        if node:
            if node.leftchild:
                node.label_if_less = node.leftchild.label_if_less
            if node.parent:
                node.threshold = node.parent.threshold
            print(node)
            sf.print_final_structure(node.leftchild, False)
            sf.print_final_structure(node.rightchild, False)   

class Node:
        def __init__(sf, data, parent, pos):
            sf.rightchild = None
            sf.leftchild = None
            sf.data = data
            sf.label_if_less = None
            sf.feature = None
            sf.threshold = None
            sf.pure = sf.check_purity()
            sf.parent = None
            if parent:
                sf.parent = parent
                if pos == "left":
                    sf.parent.leftchild = sf
                elif pos == "right":
                    sf.parent.rightchild = sf

        def __str__(sf):
            p = sf.parent.feature if sf.parent else None
            return "f<%s> <= t<%s> : l<%s> || p<%s>" %(sf.feature, sf.threshold, sf.label_if_less, p)

        def check_purity(sf):
            # All data has the same label, so the label is going to be the same as the data
            same_label_count = np.count_nonzero(sf.data[:,sf.data.shape[1]-1])

            if same_label_count == 0:
                sf.label_if_less = 0
            elif same_label_count == sf.data.shape[0]:
                sf.label_if_less = 1
            else :
                return False
            return True

if __name__ == "__main__":
    # 2 x 6 matrix
    # index 0 ==> feature
    # index 1 ==> label
    # threshold = 1
    #testX = np.array([[-1,1],[-2,0],[2,1],[-4,0],[3,1],[6,0]])

    # 2 features
    testX = np.array([[1,1,1], [-1,-1,1], [-1,1,0], [1,-1,0]])
    
    tree = DecisionTree(None,testX)

    # Check if feature_in_branch works
    # Comment sf.pure in __init__ in Node
    """n = Node([1], None, None)
    n.feature = 5
    n2 = Node([1],n, "left")
    n3 = Node([1], n, "right")
    n4 = Node([1], n2, "left")

    print(tree.feature_in_branch(n4,4))"""

    #tree = DecisionTree(["hw3train.txt"])
    tree.train()
    tree.print_final_structure()

    #print(tree.entropy(testX))
    #print(tree.conditional_entropy(testX,0,1))
