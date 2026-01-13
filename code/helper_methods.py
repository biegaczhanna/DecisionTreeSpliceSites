#
#   Author: Hanna Biegacz
#   This file contains helper methods for the main.py file, 
#   such as data splitting and model evaluation.
#

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from algorithms.DecisionTree import DecisionTree
import numpy as np
import random
from itertools import product

def run_grid_search(data, cv_folds=None):
    best_accuracy = 0
    best_config = {}

    train_sizes = [0.8,] # np.arange(0.1, 1.0, 0.1)
    depths = (5, 7,  10, 15, 20, 40)
    min_samples = (2, 10, 30, 50, 70, 90, 110)
    min_gains = (0.0, 0.01, 0.1, 0.2)

    for train_set_size, d, ms, mg in product(train_sizes, depths, min_samples, min_gains):
        accuracy, tree_model = _evaluate_config(data, train_set_size, d, ms, mg, cv_folds)
        
        if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_config = {
                            "accuracy": best_accuracy,
                            "train_set_size": train_set_size,
                            "depth": d,
                            "min_samples_split": ms,
                            "min_gain": mg,
                            "tree": tree_model
                        }
    return best_config

def _evaluate_config(data, train_set_size, depth, min_samples_split, min_gain, cv_folds):
    """
    Evaluates a specific configuration using Cross-Validation or a stratified split.
    """
    if cv_folds and cv_folds > 1:
        results = perform_cross_validation(data, depth, min_samples_split, min_gain, k=cv_folds)
        mean_acc = results['mean_accuracy']
        
        train_data_rep, _ = stratified_split_data(data, 0.8) # Default fallback
        dt_rep = DecisionTree(train_data_rep, max_depth=depth, min_samples_split=min_samples_split, min_gain=min_gain)
        dt_rep.train()
        
        return mean_acc, dt_rep

    else:
        train_data, test_data = stratified_split_data(data, train_set_size)
        dt = DecisionTree(train_data, max_depth=depth, min_samples_split=min_samples_split, min_gain=min_gain)
        dt.train()
        accuracy, _, _, _ = calculate_metrics(dt, test_data)
        return accuracy, dt

def perform_cross_validation(data, depth, min_samples_split, min_gain, k=5):
    folds = k_fold_split(data, k)
    accuracies = []
    recalls = []
    precisions=[]
    conf_matrices = []
    
    for i, (train_data, test_data) in enumerate(folds):
        dt = DecisionTree(train_data, max_depth=depth, min_samples_split=min_samples_split, min_gain=min_gain)
        dt.train()
        acc, cm, rec, prec = calculate_metrics(dt, test_data)
        accuracies.append(acc)
        recalls.append(rec)
        precisions.append(prec)
        conf_matrices.append(cm)
        
    return {
        "mean_accuracy": np.mean(accuracies),
        "mean_recall": np.mean(recalls),
        "mean_precision": np.mean(precisions),
        "mean_confusion_matrix": np.mean(conf_matrices, axis=0)
    }

def calculate_metrics(tree_model, test_data):
    y_true = []
    y_pred = []

    for row in test_data:
        actual_class = row[0]
        dna_sequence = row[1]
        prediction = tree_model.predict(dna_sequence)
        
        y_true.append(actual_class)
        y_pred.append(prediction)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    unique_true = set(y_true)
    unique_pred = set(y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)

    return accuracy, conf_matrix, recall, precision


# DIFFERENT METHODS OF SPLITTING DATA

def split_data(data, train_size=0.8):
    random.shuffle(data)
    train_size = int(len(data) * train_size)
    return data[:train_size], data[train_size:]

def stratified_split_data(data, train_size=0.8):
    """
    Splits data into train and test sets while maintaining the proportion of classes.
    """
    positives = [row for row in data if row[0] == 1]
    negatives = [row for row in data if row[0] == 0]
    
    random.shuffle(positives)
    random.shuffle(negatives)
    
    train_pos_count = int(len(positives) * train_size)
    train_neg_count = int(len(negatives) * train_size)
    
    train_data = positives[:train_pos_count] + negatives[:train_neg_count]
    test_data = positives[train_pos_count:] + negatives[train_neg_count:]
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    return train_data, test_data

def k_fold_split(data, k=5):
    """
    Generates k folds for cross-validation, maintaining class stratification.
    Returns a list of tuples: (train_data, test_data) for each fold.
    """
    positives = [row for row in data if row[0] == 1]
    negatives = [row for row in data if row[0] == 0]
    
    random.shuffle(positives)
    random.shuffle(negatives)
    
    pos_chunks = [positives[i::k] for i in range(k)]
    neg_chunks = [negatives[i::k] for i in range(k)]
    
    folds = []
    for i in range(k):
        test_data = pos_chunks[i] + neg_chunks[i]
        
        train_data = []
        for j in range(k):
            if i != j:
                train_data.extend(pos_chunks[j] + neg_chunks[j])
        
        random.shuffle(train_data)
        random.shuffle(test_data)
        folds.append((train_data, test_data))
        
    return folds

