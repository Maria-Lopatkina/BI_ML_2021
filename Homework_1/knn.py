import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        dist = np.zeros((len(X), len(self.train_X)))
        for i in range(len(X)):
            s = 0
            for j in range(len(self.train_X)):
                s = sum(abs(X[i] - self.train_X[j]))
                dist[i][j] = s
                s = 0
        return dist


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        dist = np.zeros((len(X), len(self.train_X)))
        for i in range(len(X)):
            a = abs(self.train_X - X[i])
            dist[i] = sum(a.transpose())
        return dist


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        dist = np.abs(X[:, None, :] - self.train_X[None, :, :]).sum(axis=-1)
        return dist

    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.full(n_test, None)
        for el in range(n_test):
            one_sample = list(distances[el])
            sorted_ls = sorted(one_sample)
            k_list = []
            for i in range(self.k):
                min_dist = sorted_ls[i]
                in_y_train = self.train_y[one_sample.index(min_dist)]
                k_list.append(in_y_train)
            prediction[el] = most_common(k_list)
        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        # prediction = np.zeros(n_test, np.int)
        prediction = np.full(n_test, None)
        for el in range(n_test):
            one_sample = list(distances[el])
            sorted_ls = sorted(one_sample)
            k_list = []
            for i in range(self.k):
                min_dist = sorted_ls[i]
                in_y_train = self.train_y[one_sample.index(min_dist)]
                k_list.append(in_y_train)
            prediction[el] = most_common(k_list)
        return prediction


def most_common(lst):
    return max(set(lst), key=lst.count)