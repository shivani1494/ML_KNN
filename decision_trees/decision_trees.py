import numpy as np
import copy

class DecisionTree:
    def __init__(sf, filenames, input_data=None):
        sf._num_features = 0
        if input_data != None:
            sf._input_data = input_data
            sf._num_features = input_data.shape[1]-1
        else:
            sf._input_data = sf.load(filenames[0])
            # sf._num_features calculated in get_feature_names
            sf._feature_names = sf.get_feature_names("hw3features.txt")
        sf._rootnode = None

        sf.predictions = sf.reset_predictions();

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
            #print("iter:",iteration); iteration += 1;
            if not(current_node.pure):
                #print("Node Not Pure")
                # Set current node to pure
                # (we are not going to compute the tree for this node again)
                current_node.pure = True
                feature, threshold = sf.maximize_IG(current_node)

                if feature == None and threshold == None:
                    # Only called when features_in_branch is being used 
                    # All the features used in that branch
                    # "Actual" label of all the data in current_node is the majority label
                    # (Node Purity is checked when node is created)

                    label_index = current_node.data.shape[1]-1
                    nonzero = np.count_nonzero(current_node.data[:,label_index])
                    if (nonzero > current_node.data[:,label_index].shape[0]-nonzero):
                        # Actual label for this branch
                        current_node.label = 1
                        #print("Label Right",current_node.label)
                    else:
                        # Actual label for this branch
                        current_node.label = 0
                        #print("Label Left",current_node.label)
                    
                    # Set parent Label
                    if current_node.parent.leftchild == current_node:
                        current_node.parent.label_if_less = current_node.label
                    elif current_node.parent.rightchild == current_node:
                        current_node.parent.label_if_great = current_node.label

                else:
                    #print("Features Not used")
                    # Features not being used
                    current_node.threshold = threshold
                    current_node.feature = feature

                    sf.split(current_node, feature, threshold)
                    #print(current_node)

                    #print("Building Left Branch")
                    # Build Left Branch
                    current_node = current_node.leftchild
                

            elif current_node.parent:
                #print("Label for pure",current_node.label)

                if current_node.parent.rightchild and current_node.parent.rightchild != current_node:

                    #print("Building Right Branch")
                    # Build sibling (rightchild) branch
                    current_node = current_node.parent.rightchild
                elif current_node.parent != None:
                    # Go to grandfather
                    #print("Go to parent")
                    current_node = current_node.parent

            else:
                break
            

    def split(sf, node, feature, threshold):
        # Creates two branches
        label = node.data.shape[1]-1
        ind_greater, ind_smaller = sf.indices_feature_leq_threshold(node.data,feature,threshold)
        
        node_smaller = Node(node.data[ind_smaller,:], node, "left")
        node_greater = Node(node.data[ind_greater,:], node, "right")


    def maximize_IG(sf,node):
        X = node.data
        max_ig = None
        feature = None
        threshold = None

        
        for f in range(sf._num_features):
            # Checks if the feature is already in use in the branch
            #if not(sf.feature_in_branch(node,f)):
                #print("# F",f)
            X_f = np.array(X[:,f], copy=True) # Deep copy
            X_f.sort()
            for split in range(X_f.shape[0]-1):
                t = (X_f[split] + X_f[split+1]) / 2

            #t = sf.get_threshold(X,f)
                ig = sf.information_gain(X,f,t)
                #print(ig)
                if max_ig == None or ig >= max_ig:
                    max_ig = ig
                    feature = f
                    threshold = t
        #print("F Chosen,",feature)
        return feature, threshold

    def information_gain(sf, X, feature, threshold):
        # uncertainty (entropy) reduced in data X when knowing Z
        return sf.entropy(X) - sf.conditional_entropy(X,feature,threshold)


    def entropy(sf, X):
        # Log(0) = -Inf
        #print("Here",X.shape[0])
        #print(X)
        label_index = X.shape[1]-1
        nonzero = np.count_nonzero(X[:, label_index])
        num_examples = X.shape[0]

        # Division by 0 otherwise
        if nonzero == num_examples or nonzero == 0:
            return 0

        return -nonzero/num_examples*np.log(nonzero/num_examples) -(num_examples-nonzero)/num_examples*np.log((num_examples-nonzero)/num_examples)

    def conditional_entropy(sf, X, feature, threshold):
        num_examples = X.shape[0]
        ind_greater, ind_smaller = sf.indices_feature_leq_threshold(X, feature, threshold)

        greater_entropy = sf.entropy(X[ind_greater,:]) # H(X,Z = 0)
        smaller_entropy = sf.entropy(X[ind_smaller,:]) # H(X,Z = 1)

        prob_greater = X[ind_greater,:].shape[0]/num_examples # P(Z=0)
        prob_smaller = X[ind_smaller,:].shape[0]/num_examples # P(Z=1)

        return prob_greater * greater_entropy + prob_smaller * smaller_entropy

    def indices_feature_leq_threshold(sf, X, feature, threshold):
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

    def print_final_structure(sf, node = None, root = True, level = 0):
        if root and sf._rootnode:
            print("***START***")
            node = sf._rootnode
        if node and level < 3: ###################################### Lvl down
            print(node)
            sf.print_final_structure(node.leftchild, False, level+1)
            sf.print_final_structure(node.rightchild, False, level+1)

    def predict(sf,X, node, root=True):
        # Does not preserve the order of the data
        if root:
            node = sf._rootnode
        elif not(node):
            return

        greater, smaller = sf.indices_feature_leq_threshold(X,node.feature,node.threshold)

        if node.label_if_less == None:
            sf.predict(X[smaller,:], node.leftchild, False)
        else :
            smaller_label = node.label_if_less

            # Add label
            smaller = np.hstack((X[smaller,:], np.ones((X[smaller,:].shape[0], 1)) * smaller_label))
            # Add to predictions
            sf.predictions = np.vstack((sf.predictions, smaller))
            
        
        if node.label_if_great == None:
            sf.predict(X[greater,:], node.rightchild, False)
        else:
            greater_label = node.label_if_great
            
            greater = np.hstack((X[greater,:], np.ones((X[greater,:].shape[0], 1)) * greater_label))
            sf.predictions = np.vstack((sf.predictions, greater))


    def reset_predictions(sf):
        # An array of dimensions 1 x num_featues + correct_prediction + prediction
        # Need to add an all 0 row in order to append the predictions later
        sf.predictions = np.zeros((1,sf._num_features+2))

    def deep_copy(sf):
        tree = copy.copy(sf)
        # Copy Root and link roots
        tree._rootnode = copy.copy(sf._rootnode)
        tree._rootnode.copy = sf._rootnode
        sf._rootnode.copy = tree._rootnode

        # Copy children
        if sf._rootnode.rightchild:
            rightchild = copy.copy(sf._rootnode.rightchild)
            rightchild.parent = tree._rootnode
            tree._rootnode.rightchild = rightchild

            sf.copy_nodes(sf._rootnode.rightchild, tree._rootnode.rightchild)
            
        if sf._rootnode.leftchild:
            leftchild = copy.copy(sf._rootnode.leftchild)
            leftchild.parent = tree._rootnode
            tree._rootnode.leftchild = leftchild
            
            sf.copy_nodes(sf._rootnode.leftchild, tree._rootnode.leftchild)

        return tree

    def copy_nodes(sf,node, ncop):
        node.copy = ncop
        ncop.copy = node
        if node.rightchild:
            rightchild = copy.copy(node.rightchild)
            rightchild.parent = ncop
            ncop.rightchild = rightchild
            sf.copy_nodes(node.rightchild, ncop.rightchild)
        if node.leftchild:
            leftchild = copy.copy(node.leftchild)
            leftchild.parent = ncop
            ncop.leftchild = leftchild
            sf.copy_nodes(node.leftchild, ncop.rightchild)



class Node:
        def __init__(sf, data, parent, pos):
            sf.copy = None # Used to keep track of the changes made when pruning
            sf.marked = False # Used to keep track of the path
            sf.rightchild = None
            sf.leftchild = None
            sf.data = data
            sf.label_if_less = None # Only for parents of leaf nodes
            sf.label_if_great = None # Only for parents of leaf nodes
            sf.label = None # Only for Leaf Nodes
            sf.feature = None
            sf.threshold = None
            sf.parent = None
            if parent:
                sf.parent = parent
                if pos == "left":
                    sf.parent.leftchild = sf
                elif pos == "right":
                    sf.parent.rightchild = sf
            sf.pure = sf.check_purity()

        def __str__(sf):
            #if sf.label != None:
                # leaf node
            #    return ""

            p = None
            if (sf.parent):
                p = str(sf.parent.feature)
                if sf.parent.leftchild == sf:
                    p += "l"
                else:
                    p += "r"

            return "f<%s> ? t<%s> : ll<%s> lg<%s> || p<%s> || #%i" %(sf.feature, sf.threshold, sf.label_if_less, sf.label_if_great, p, sf.data.shape[0])

        def check_purity(sf):
            # All data has the same label, so the label is going to be the same as the data
            same_label_count = np.count_nonzero(sf.data[:,sf.data.shape[1]-1])

            if same_label_count == 0:
                sf.label = 0
            elif same_label_count == sf.data.shape[0]:
                sf.label = 1
            else :
                return False
            # Set Parent Label
            if sf.parent:
                if sf.parent.leftchild == sf:
                    sf.parent.label_if_less = sf.label
                elif sf.parent.rightchild == sf:
                    sf.parent.label_if_great = sf.label
            return True

def prune(tree, treeTest, val_data, node):
        # Breadth First (Yes No Branches)
        # tree is the original, treeTest a copy
        # node from treeTest currently being checked
        # root_node initially

        feature = node.feature
        threshold = node.threshold
        X = node.data

        # Assign labels according to the majority
        nonzero = np.count_nonzero(X[:,-1])
        if X.shape[0]-nonzero < nonzero:
            label = 1
        else:
            label = 0

        if not(node.parent): # Root Node
            node.leftchild.label = label
            node.rightchild.label = label
            node.label_if_less = label
            node.label_if_great = label

        else: # Any other node , do not take node if leaf node (if it is already labeled)
              # Node checked in find_node

            node.label = label
            if node.parent.leftchild == node:
                node.parent.label_if_less = label
            elif node.parent.rightchild == node:
                node.parent.label_if_great = label
        node.rightchild = None
        node.leftchild = None

        tree.reset_predictions()
        treeTest.reset_predictions()
        tree.predict(val_data,None)
        treeTest.predict(val_data,None)

        correct = np.count_nonzero(tree.predictions[:,-1] == tree.predictions[:,-2])

        error_tree = val_data.shape[0] - correct
        correct_test = np.count_nonzero(treeTest.predictions[:,-1] == treeTest.predictions[:,-2])
        error_treeTest = val_data.shape[0] - correct_test


        if error_treeTest <= error_tree:
            print("Val Error Tree", error_tree)
            print("Val Error treeTest", error_treeTest)
            return treeTest, node
        else:
            node.copy.marked = True # Original Node marked
            treeTest = tree.deep_copy()
            next_node = find_node(tree._rootnode)
            if not(next_node):
                return treeTest, next_node
            else:
                return prune(tree, treeTest, val_data, next_node)
def find_node(node):
    # Breadth First Search (Yes No)
    # Search from top, original tree
    not_searched = [node]

    while not_searched :
        node = not_searched.pop(0)

        if node.leftchild: # Has children
            if node.label_if_great == None: # leftchild is not leaf:
                not_searched.append(node.rightchild)
            if node.label_if_less == None: # rightchild is not leaf
                not_searched.append(node.leftchild)
        if node.marked:
            node.marked = False
            return not_searched.pop(0).copy
            
if __name__ == "__main__":
    # 2 x 6 matrix
    # index 0 ==> feature
    # index 1 ==> label
    # threshold = 1
    #testX = np.array([[-1,1],[-2,0],[2,1],[-4,0],[3,1],[6,0]])

    # 2 features
    #testX = np.array([[1,1,1], [-1,-1,1], [-1,1,0], [1,-1,0]])
    
    #tree = DecisionTree(None,testX)
    # 2 features
    #X = np.array([[-1,1,1]])


    # Training
    tree = DecisionTree(["hw3train.txt"])
    tree.train()
    tree.print_final_structure()

    # Training Error
    tree.reset_predictions()
    X = tree.load("hw3train.txt")
    tree.predict(X,None)
    print(np.count_nonzero(tree.predictions[:,-1] == tree.predictions[:,-2]))
    
    # Test Error
    tree.reset_predictions()
    Xtest = tree.load("hw3test.txt")
    tree.predict(Xtest,None)
    print(np.count_nonzero(tree.predictions[:,-1] == tree.predictions[:,-2]))

    #### Pruning
    Xtest = tree.load("hw3test.txt")
    Xval = tree.load("hw3validation.txt")
    treeTest = tree.deep_copy()
    print("First prune")
    treeTest, node = prune(tree, treeTest, Xval, treeTest._rootnode)
    print(node, node.label)
    # Error
    treeTest.reset_predictions()
    treeTest.predict(Xtest,None)
    print(np.count_nonzero(treeTest.predictions[:,-1] == treeTest.predictions[:,-2]))    

    # Second Pruning
    tree = treeTest
    treeTest = tree.deep_copy()
    print("Second prune")
    treeTest, node = prune(tree, treeTest, Xval, treeTest._rootnode)
    print(node, node.label)
    # Error    
    treeTest.reset_predictions()
    treeTest.predict(Xtest,None)
    print(np.count_nonzero(treeTest.predictions[:,-1] == treeTest.predictions[:,-2]))    

