import csv

word_sample = set({})
with open('sample_classes.txt', 'r') as in_txt:
    for line in in_txt:
        word_sample.add(line[:-1])
print(word_sample)

with open('random_sample.csv', 'w') as output:
    writer = csv.writer(output)
    writer.writerow(['split', 'file', 'gloss'])
    for split in ['train', 'val', 'test']:
        with open(f'../ASL_Citizen/splits/{split}.csv', 'r') as input:
            reader = csv.reader(input)
            next(reader)

            for line in reader:
                if line[2] in word_sample:
                    writer.writerow([split, line[1], line[2]])