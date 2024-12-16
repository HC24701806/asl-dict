import csv

word_sample = set({})
i = 0
with open('stats.txt', 'r') as in_txt:
    for line in in_txt:
        if i == 100:
            break
        word_sample.add(line.split(' ')[0])
        i += 1

with open('random_sample.csv', 'w') as output:
    writer = csv.writer(output)
    writer.writerow(['split', 'file', 'gloss'])
    for split in ['train', 'val', 'test']:
        with open(f'./ASL_Citizen/splits/{split}.csv', 'r') as input:
            reader = csv.reader(input)
            next(reader)

            for line in reader:
                if line[2] in word_sample:
                    writer.writerow([split, line[1], line[2]])