import os
import matplotlib.pyplot as plt

def parse_results_file(file_path):
    """
    Parses the results file to extract data for each parameter influence section.
    """
    data = {}
    current_param = None
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {}

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect new section
        if line.startswith("========== Influence of"):
            # Extract parameter name
            # Format: ========== Influence of PARAM_NAME ==========
            current_param = line.replace("==========", "").replace("Influence of", "").strip()
            
            data[current_param] = {
                "values": [],
                "accuracy": [],
                "recall": [],
                "precision": []
            }
            
            # Skip until data starts (skip Fixed Parameters and Headers)
            # We look for the line starting with "Accuracy" which is the header
            while i < len(lines) and not lines[i].strip().startswith("Accuracy"):
                i += 1
            i += 1 # Move past the header line
            continue
            
        # Parse data lines
        if current_param and line:
            # Check if line is a separator or empty
            if line.startswith("---") or line.startswith("==="):
                i += 1
                continue

            parts = line.split()
            # We expect at least 6 columns: Acc, Rec, Prec, Depth, MinSamples, MinGain
            if len(parts) >= 6:
                try:
                    acc = float(parts[0])
                    rec = float(parts[1])
                    prec = float(parts[2])
                    
                    # Columns: 
                    # 3: depth
                    # 4: min_samples_split
                    # 5: min_gain
                    
                    val = 0.0
                    if current_param == "depth":
                        val = float(parts[3])
                    elif current_param == "min_samples_split":
                        val = float(parts[4])
                    elif current_param == "min_gain":
                        val = float(parts[5])
                    else:
                        # Fallback if param name doesn't match standard keys strictly
                        # but usually it should be one of them.
                        pass
                    
                    data[current_param]["values"].append(val)
                    data[current_param]["accuracy"].append(acc)
                    data[current_param]["recall"].append(rec)
                    data[current_param]["precision"].append(prec)
                except ValueError:
                    # Skip lines that don't parse as floats
                    pass
        
        i += 1
        
    return data

def plot_and_save_results(file_path):
    """
    Reads the results file, generates plots for each parameter, and saves them.
    """
    print(f"Generating plots for {file_path}...")
    dataset_name = os.path.basename(file_path).replace("_parameter_influence.txt", "")
    output_dir = os.path.dirname(file_path)
    plots_dir = os.path.join(output_dir, "plots")
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    data = parse_results_file(file_path)
    
    if not data:
        print("No data found to plot.")
        return

    for param, metrics in data.items():
        if not metrics["values"]:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Sort data by x-axis value to ensure lines are drawn correctly
        sorted_indices = sorted(range(len(metrics["values"])), key=lambda k: metrics["values"][k])
        
        x_vals = [metrics["values"][i] for i in sorted_indices]
        y_acc = [metrics["accuracy"][i] for i in sorted_indices]
        y_rec = [metrics["recall"][i] for i in sorted_indices]
        y_prec = [metrics["precision"][i] for i in sorted_indices]
        
        plt.plot(x_vals, y_acc, label='Accuracy', marker='o', linestyle='-')
        plt.plot(x_vals, y_rec, label='Recall', marker='s', linestyle='--')
        plt.plot(x_vals, y_prec, label='Precision', marker='^', linestyle='-.')
        
        plt.title(f'Influence of {param} on {dataset_name}')
        plt.xlabel(param)
        plt.ylabel('Score')
        
        # Dynamic Y-axis limits
        all_y_values = y_acc + y_rec + y_prec
        if all_y_values:
            y_min = min(all_y_values)
            y_max = max(all_y_values)
            y_range = y_max - y_min
            
            # Add some padding (10% of range, or 0.02 minimum)
            padding = max(y_range * 0.1, 0.02)
            
            plt.ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))
        
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        output_filename = f"{dataset_name}_{param}.png"
        output_path = os.path.join(plots_dir, output_filename)
        
        plt.savefig(output_path)
        plt.close()
        print(f"  Saved plot: {output_path}")

if __name__ == "__main__":
    results_dir = "results"
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith("_parameter_influence.txt"):
                full_path = os.path.join(results_dir, filename)
                plot_and_save_results(full_path)
    else:
        print(f"Directory '{results_dir}' does not exist.")
