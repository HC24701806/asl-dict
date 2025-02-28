import csv

def survey():
    dict = {}
    dict['train'] = {}
    dict['val'] = {}
    dict['test'] = {}

    with open('mini_dataset.csv') as input, open('survey.csv', 'w') as output:
        reader = csv.reader(input)
        next(reader)

        for line in reader:
            split = line[0]
            label = line[4]
            if label not in dict[split]:
                dict[split][label] = 1
            else:
                dict[split][label] += 1

        writer = csv.writer(output)
        writer.writerow(['label', 'freq'])
        for label in dict['train']:
            writer.writerow([label, dict['train'][label], dict['val'][label], dict['test'][label]])