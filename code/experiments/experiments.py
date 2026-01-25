'''
  Author: Hanna Biegacz
  
  This file contains the script for running parameter experiments on the decision tree classifier.
  The script runs experiments for both donors and acceptors. You can find the results in the results/ folder.
  Automatically generates plots for the results and saves them in the results/plots/ folder.

'''

import os
import sys
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from file_parsing.FileParser import FileParser as file_parser
from helper_methods import evaluate_config, run_experiment
from results_plotter import plot_and_save_results



DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))

FILES = [
    ("Donors", os.path.join(DATA_DIR, "spliceDTrainKIS.dat.txt")),
    ("Acceptors", os.path.join(DATA_DIR, "spliceATrainKIS.dat.txt")) 
]

# Base configutation to set while changing the other parameters
BASE_CONFIGS = {
    "Donors": {
        "depth": 15,
        "min_samples_split": 15,
        "min_gain": 0.00
    },
    "Acceptors": {
        "depth": 25,
        "min_samples_split": 40,
        "min_gain": 0.00
    }
}


TEST_RANGES = {
    "Donors": {
        "depth": [3, 5, 7, 10, 15, 20, 40, 60, 80, 100, 120],
        "min_samples_split": [1, 2, 10, 30, 50, 70, 90, 110, 130],
        "min_gain": [0.0, 0.01, 0.02, 0.05, 0.06, 0.08, 0.09, 0.1, 0.2, 0.5]
    },
    "Acceptors": {
        "depth": [3, 5, 7, 10, 15, 20, 40, 60, 80, 100, 120],
        "min_samples_split": [1, 2, 10, 30, 50, 70, 90, 110, 130],
        "min_gain": [0.0, 0.01, 0.02, 0.05, 0.06, 0.08, 0.09, 0.1, 0.2, 0.5]
    },
}


def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    for name, file_path in FILES:
        if not os.path.exists(file_path):
            print(f"Skipping {name}: File {file_path} not found.")
            continue

        print(f"\n{'='*3} Processing {name} {'='*3}")
        fp = file_parser(file_path)
        data = fp.parse()
        
        base_config = BASE_CONFIGS[name]
        ranges = TEST_RANGES[name]
        
        analysis_results = {}

        for param_name, values_to_test in ranges.items():
            print(f"--- Testing influence of: {param_name} ---")
            param_results = []
            
            for value in values_to_test:
                current_config = deepcopy(base_config)
                current_config[param_name] = value
                
                print(f"   Testing {param_name} = {value}")
                
                res = run_experiment(
                    data, 
                    current_config["depth"], 
                    current_config["min_samples_split"], 
                    current_config["min_gain"]
                )
                param_results.append(res)
            
            analysis_results[param_name] = param_results

        output_filename = os.path.join(RESULTS_DIR, f"{name}_parameter_influence.txt")
        with open(output_filename, 'w') as f:
            for param_name, results in analysis_results.items():
                fixed_params = {k: v for k, v in base_config.items() if k != param_name}
                
                f.write(f"\n{'='*10} Influence of {param_name} {'='*10}\n")
                f.write(f"Fixed Parameters: {fixed_params}\n")
                f.write("-" * 80 + "\n")
                f.write("Accuracy \t\t\tRecall \t\t\tPrecision \t\t\tDepth \t\t\tMinSamplesSplit \t\t\tMinGain\n")
                
                for result in results:
                    f.write(f"{result['accuracy']:.4f} \t\t\t{result['recall']:.4f} \t\t\t{result['precision']:.4f} \t\t\t"
                            f"{result['depth']} \t\t\t{result['min_samples_split']} \t\t\t{result['min_gain']}\n")
                f.write("\n")
                
        print(f"Analysis complete for {name}. Results saved to {output_filename}")
        
        try:
            plot_and_save_results(output_filename)
        except Exception as e:
            print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()
