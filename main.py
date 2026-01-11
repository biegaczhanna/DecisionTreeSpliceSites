from file_parsing.FileParser import FileParser as file_parser
from algorithms.DecisionTree import DecisionTree, print_tree
from contextlib import redirect_stdout

def main():
    depth = 35
    fp = file_parser("data/spliceDTrainKIS.dat.txt")
    training_data = fp.parse()

    dt = DecisionTree(training_data, max_depth=depth)
    root = dt.train()

    print("--- BUDOWA DRZEWA ---")

    # Wywołujemy naszą nową funkcję wizualizującą
    with open('outTree.txt', 'w') as f:
        with redirect_stdout(f):
            print_tree(root, 0)

main()