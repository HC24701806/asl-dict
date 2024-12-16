import numpy as np
import os
import csv

for filename in os.listdir('./dataset'):
    with open(f'./dataset/{filename}') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        num_lines = 0
        for line in reader:
            num_lines += 1
        if num_lines > 12:
            print(filename + ' ' + str(num_lines))