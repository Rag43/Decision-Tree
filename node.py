class Node():
    def _init_(self, feature_index=None, threshold=None, left=None, right=None,
                info_gain=None, value=None):
        
        #For decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        #For leaf node
        self.value = value
