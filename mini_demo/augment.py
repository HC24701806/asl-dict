import numpy as np
import csv
import os

dict = {}
with open('random_sample.csv') as sample_csv:
    reader = csv.reader(sample_csv)
    next(reader)

    for line in reader:
        dict[line[1][:-4]] = [line[0], line[2]]

with open('augmented_sample.csv', 'w') as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(['split', 'file', 'gloss'])
    for filename in os.listdir('./mini_raw_dataset'):
        if filename[-4:] != '.csv':
            continue
        filename = filename[:-4]
        with open(f'./mini_raw_dataset/{filename}.csv') as raw_file:
            reader = csv.reader(raw_file)
            next(reader)

            raw_data = []
            for line in reader:
                for i in range(1, 137):
                    raw_data.append(np.float32(line[i]))

            for x_shift in [-0.2, -0.1, 0, 0.1, 0.2]:
                for i in range(5):
                    augmented_data = np.copy(raw_data)
                    noise = np.random.normal(0, 0.005, len(augmented_data))
                    for j in range(len(augmented_data)):
                        if augmented_data[j] != 0:
                            augmented_data[j] += noise[j]
                            if j % 2 == 0:
                                augmented_data[j] += x_shift
                    
                    new_file_name = f'{filename}_{x_shift}_{i}'
                    with open(f'./mini_dataset/{new_file_name}.txt', 'w') as outfile:
                        for entry in augmented_data:
                            outfile.write(str(entry) + '\t')
                        outfile.close()
                    info = dict[filename]
                    writer.writerow([info[0], new_file_name, info[1]])