'''
    Author: Ada Jacyna
    This file contains FileParser class, which is responsible for parsing the data from the file. 
    It is used in main.py to manage data from the files in the data/ directory. 
    
'''

class FileParser:
    def __init__(self, path: str):
        self.path = path
    
    def parse(self):
        pairs = []
        with open(self.path, "r") as f:
            lines = f.read().splitlines()
        
        # It is neccessary to skip the first line
        lines = lines[1:]

        for i in range(0, len(lines), 2):
            class_value = int(lines[i])
            dna = lines[i + 1]
            pairs.append((class_value, dna))
        return pairs
