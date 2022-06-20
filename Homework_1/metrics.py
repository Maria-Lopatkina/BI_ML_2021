import numpy as np
from sklearn.metrics import confusion_matrix


def binary_classification_metrics(y_pred, y_true, metric):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    conf_m = confusion_matrix(y_pred, y_true)
    TP = conf_m[0][0]
    TN = conf_m[1][1]
    FP = conf_m[1][0]
    FN = conf_m[0][1]
    metr = {"accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None}
    try:
        metr["accuracy"] = round((TP + TN) / (TP + TN + FP + FN), 2)
        metr["precision"] = round(TP / (TP + FP), 2)
        metr["recall"] = round(TP / (TP + FN), 2)
        metr["f1"] = round((TP) / (TP + 0.5 * (FP + FN)), 2)
    except ZeroDivisionError as e:
        print("Division by zero")
    return metr[metric]

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    TruePN = 0
    for el in range(len(y_pred)):
        if y_pred[el] == y_true[el]:
            TruePN += 1
    accuracy = TruePN / len(y_pred)
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    ymean = sum(y_true) / len(y_true)
    s1 = 0
    s2 = 0
    for el in range(len(y_true)):
        s1 += (y_true[el] - y_pred[el])**2
        s2 += (y_true[el] - ymean)**2
    r2 = 1 - s1 / s2
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    s = 0
    for el in range(len(y_true)):
        s += (y_true[el] - y_pred[el])**2
    mse = 1 / len(y_true) * s
    return mse



def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    s = 0
    for el in range(len(y_true)):
        s += abs(y_true[el] - y_pred[el])
    mae = 1 / len(y_true) * s
    return mae
