'''
    Author: Hanna Biegacz
    This code is used during the experiment process, to generate plots for the results.
    The results are saved in the results/plots/ folder.
    This code can also be run separately.
    
'''

import os
import matplotlib.pyplot as plt

def parse_results_file(file_path):
    """
    Parses the results file to extract data for each parameter influence section.
    Automatically detects the parameter name and handles the file formating.
    """
    results_data = {}
    current_parameter = None
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {}

    with open(file_path, 'r') as file_handle:
        file_lines = file_handle.readlines()
        
    line_index = 0
    while line_index < len(file_lines):
        line = file_lines[line_index].strip()
        
        if line.startswith("========== Influence of"):
            current_parameter = line.replace("==========", "").replace("Influence of", "").strip()
            
            results_data[current_parameter] = {
                "values": [],
                "accuracy": [],
                "recall": [],
                "precision": []
            }
            
            while line_index < len(file_lines) and not file_lines[line_index].strip().startswith("Accuracy"):
                line_index += 1
            line_index += 1 
            continue
            
        if current_parameter and line:
            if line.startswith("---") or line.startswith("==="):
                line_index += 1
                continue

            line_parts = line.split()
            if len(line_parts) >= 6:
                try:
                    accuracy = float(line_parts[0])
                    recall = float(line_parts[1])
                    precision = float(line_parts[2])
                    parameter_value = 0.0
                    if current_parameter == "depth":
                        parameter_value = float(line_parts[3])
                    elif current_parameter == "min_samples_split":
                        parameter_value = float(line_parts[4])
                    elif current_parameter == "min_gain":
                        parameter_value = float(line_parts[5])
                    else:
                        pass
                    
                    results_data[current_parameter]["values"].append(parameter_value)
                    results_data[current_parameter]["accuracy"].append(accuracy)
                    results_data[current_parameter]["recall"].append(recall)
                    results_data[current_parameter]["precision"].append(precision)
                except ValueError:
                    pass
        line_index += 1
    return results_data

def plot_and_save_results(file_path):
    """
    Reads the results file, generates plots for each parameter, and saves them.
    Calculates what range should be used for the y-axis.
    """
    print(f"Generating plots for {file_path}...")
    dataset_name = os.path.basename(file_path).replace("_parameter_influence.txt", "")
    output_directory = os.path.dirname(file_path)
    plots_directory = os.path.join(output_directory, "plots")
    
    if not os.path.exists(plots_directory):
        os.makedirs(plots_directory)
        
    parsed_data = parse_results_file(file_path)
    
    if not parsed_data:
        print("No data found to plot.")
        return

    for parameter_name, metrics_data in parsed_data.items():
        if not metrics_data["values"]:
            continue
            
        plt.figure(figsize=(10, 6))
        
        sorted_indices = sorted(range(len(metrics_data["values"])), key=lambda k: metrics_data["values"][k])
        
        x_values = [metrics_data["values"][i] for i in sorted_indices]
        accuracy_values = [metrics_data["accuracy"][i] for i in sorted_indices]
        recall_values = [metrics_data["recall"][i] for i in sorted_indices]
        precision_values = [metrics_data["precision"][i] for i in sorted_indices]
        
        plt.plot(x_values, accuracy_values, label='Accuracy', marker='o', linestyle='-')
        plt.plot(x_values, recall_values, label='Recall', marker='s', linestyle='--')
        plt.plot(x_values, precision_values, label='Precision', marker='^', linestyle='-.')
        
        plt.title(f'Influence of {parameter_name} on {dataset_name}')
        plt.xlabel(parameter_name)
        plt.ylabel('Score')
        
        all_y_values = accuracy_values + recall_values + precision_values
        if all_y_values:
            y_min = min(all_y_values)
            y_max = max(all_y_values)
            y_range = y_max - y_min
            
            padding = max(y_range * 0.1, 0.02)
            
            plt.ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))
        
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        output_filename = f"{dataset_name}_{parameter_name}.png"
        output_path = os.path.join(plots_directory, output_filename)
        
        plt.savefig(output_path)
        plt.close()
        print(f"  Saved plot: {output_path}")

if __name__ == "__main__":
    results_directory = "results"
    if os.path.exists(results_directory):
        for filename in os.listdir(results_directory):
            if filename.endswith("_parameter_influence.txt"):
                full_path = os.path.join(results_directory, filename)
                plot_and_save_results(full_path)
    else:
        print(f"Directory '{results_directory}' does not exist.")
