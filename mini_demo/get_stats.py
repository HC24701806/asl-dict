import numpy as np
import csv

dict = {}
index = 0
files = ['train', 'val', 'test']
for filename in files:
    with open(f'./ASL_Citizen/splits/{filename}.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for line in reader:
            key = line[2]
            if key not in dict:
                dict[key] = np.zeros(3, dtype=int)
            dict[key][index] += 1
    index += 1

with open('stats.txt', 'w') as out:
    for key in dict:
        out.write(key + ' ' + str(dict[key][0]) + ' ' + str(dict[key][1]) + ' ' + str(dict[key][2]) + '\n')
out.close()