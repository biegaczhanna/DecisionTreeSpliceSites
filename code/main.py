'''
  Author: Hanna Biegacz, Ada Jacyna
  Main file for the decision tree classification
'''

from file_parsing.FileParser import FileParser as file_parser
from algorithms.DecisionTree import DecisionTree, print_tree
from helper_methods import run_grid_search, perform_cross_validation, stratified_split_data, calculate_metrics
from contextlib import redirect_stdout
import os


GRID_SEARCH = False # Change to false if you want to create a single decision tree
CROSS_VALIDATION_FOLDS = 3
GRID_SEARCH_CV_FOLDS = 3 

FILES = [
    ("Donors", "data/spliceDTrainKIS.dat.txt"),
    ("Acceptors", "data/spliceATrainKIS.dat.txt") 
]

def main():
    if GRID_SEARCH:
        for name, file_path in FILES:
            if not os.path.exists(file_path):
                print(f"Skipping {name}: File {file_path} not found.")
                continue

            print(f"\n{'='*20} Processing {name} {'='*20}")
            fp = file_parser(file_path)
            data = fp.parse()

            print(f"---> Running Grid Search")
            best_result = run_grid_search(data, cv_folds=GRID_SEARCH_CV_FOLDS)
            
            print(f"Best Accuracy found: {best_result['accuracy']:.4f}")
            print(f"Recall: {best_result['recall']:.4f}")
            print(f"Precision: {best_result['precision']:.4f}")
            print(f"Confusion Matrix:\n{best_result['confusion_matrix']}")
            print(f"Parameters: Depth={best_result['depth']}, "
                f"MinSamples={best_result['min_samples_split']}, "
                f"MinGain={best_result['min_gain']}, "
                f"TrainSetSize={best_result['train_set_size']:.1f}")

            print(f"\n---> Cross Validation for best parameters")

            results = perform_cross_validation(
                data, 
                best_result['depth'], 
                best_result['min_samples_split'], 
                best_result['min_gain'], 
                CROSS_VALIDATION_FOLDS
            )
            
            print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
            print(f"Mean Recall: {results['mean_recall']:.4f}")
            print(f"Mean Precision: {results['mean_precision']:.4f}")
            print(f"Mean Confusion Matrix:\n{results['mean_confusion_matrix']}")

            best_tree = best_result['tree']        
            with open(f"trees/{name}_best_tree.txt", 'w') as f:
                with redirect_stdout(f):
                    print_tree(best_tree.root, 0)
        
    else:
        fp = file_parser(FILES[0][1])
        data = fp.parse()
        train_data, test_data = stratified_split_data(data, 0.8)
        dt = DecisionTree(train_data, max_depth=10, min_samples_split=30, min_gain=0.0)
        dt.train()

        a, cm, r, p = calculate_metrics(dt, test_data)

        print(f"Accuracy: {a}")
        print(f"Recall: {r}")
        print(f"Precision: {p}")
        print(f"Confusion matrix:\n{cm}")

        with open(f"trees/decision_tree.txt", 'w') as f:
                with redirect_stdout(f):
                    print_tree(dt.root, 0)



main()