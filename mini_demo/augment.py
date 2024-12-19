import csv
import sys
sys.path.insert(1, '/Users/haolincong/Documents/GitHub/asl-dict/')
from util import shift

def augment_file(id):
    input_path = f'./mini_dataset/{id}.mp4'
    for x_shift in range(-150, 151, 150):
        for y_shift in range(-75, 76, 75):
            output_path = f'./mini_augmented_dataset/{id}_{x_shift}_{y_shift}.mp4'
            shift(input_path, output_path, x_shift, y_shift)

with open('random_sample.csv') as in_csv, open('augmented_sample.csv', 'w') as out_csv:
    reader = csv.reader(in_csv)
    writer = csv.writer(out_csv)
    next(reader)
    writer.writerow(['split', 'file', 'gloss'])

    for line in reader:
        id = line[1][:-4]
        augment_file(id)

        for x_shift in range(-150, 151, 150):
            for y_shift in range(-75, 76, 75):
                writer.writerow([line[0], f'{id}_{x_shift}_{y_shift}.mp4', line[2]])