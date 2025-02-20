import numpy as np
import pandas as pd

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
