import numpy as np
import os
import csv

for filename in os.listdir('./mini_dataset'):
    with open(f'./mini_dataset/{filename}') as csvfile:
        if filename[-4:] == '.csv':
            reader = csv.reader(csvfile)
            next(reader)
            
            num_lines = 0
            for line in reader:
                num_lines += 1
            
            if num_lines > 13:
                print(filename + ' ' + str(num_lines))