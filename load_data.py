import pandas as pd
import numpy as np

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("iris.csv")
data.head(10)

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None,
                info_gain=None, value=None):
        
        #For decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        #For leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):

        #Initialize root
        self.root = None

        #Stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        #Split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            #Get best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            #check if IG is positive
            if best_split["info_gain"] > 0:
                #recur_left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                #recur_right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                #return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                             left_subtree, right_subtree, best_split["info_gain"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        #dictionary to store best split
        best_split = {}
        max_info_gain = -float("inf")

        #loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            #loop over all the feature values present in the data
            for threshold in possible_thresholds:
                #get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                #check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    #split labels into all, left and right of threshold
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    #compute info gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    #update best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        #return the best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        #split left and right
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        #define child weights
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - ((weight_l * self.gini_index(l_child))
                                               + (weight_r * self.gini_index(r_child)))
        else:
            gain = self.entropy(parent) - ((weight_l * self.entropy(l_child))
                                               + (weight_r * self.entropy(r_child)))
        return gain
    
    #calculating entropy (self.entropy)
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            entropy += -p_cls * np.log2(cls)
        return entropy
    
    #(self.gini_index)
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini
    
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    #recursively print out tree
    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_ "+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft: " % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright: " % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    #trains the tree (mimicking scikit's 'fit')
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    #predict new dataset
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, X, tree):
        if tree.value != None: return tree.value
        feature_val = X[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(X, tree.left)
        else:
            return self.make_prediction(X, tree.right)


X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train, Y_train)
classifier.print_tree()


