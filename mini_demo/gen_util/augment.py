import csv
import ffmpeg
import numpy as np
import scipy.stats as sci

def augment_file(id, begin, end, num_copies):
    x_shifts = np.random.randint(-200, 200, num_copies)
    y_shifts = np.random.randint(-125, 125, num_copies)
    scales = sci.truncnorm.rvs(-2, 1.333, 0.8, 0.15, num_copies)
    blurs = sci.truncnorm.rvs(-1.5, 6, 3, 2, num_copies)
    dh = sci.truncnorm.rvs(-1.25, 1.75, 150, 120, num_copies)
    db = sci.truncnorm.rvs(-2.5, 2.5, 0, 0.2, num_copies)
    flip = np.random.choice([0, 1], size=num_copies, p=[0.8, 0.2])

    frame_list = np.linspace(begin, end - 1, 8).round().astype(int)
    frame_filter = '+'.join([f'eq(n,{f})' for f in frame_list])

    for i in range(num_copies):
        stream = (
            ffmpeg.input(f'../ASL_Citizen/videos/{id}.mp4')
            .filter('select', frame_filter)
            .filter('pad', f'iw/{scales[i]}', f'ih/{scales[i]}', f'(ow-iw)/2', f'(oh-ih)/2')
            .filter('scale', f'iw*{scales[i]}', f'ih*{scales[i]}')
            .filter('crop', f'iw-{abs(x_shifts[i])}', f'ih-{abs(y_shifts[i])}', abs(min(0, x_shifts[i])), abs(min(0, y_shifts[i])))
            .filter('pad', f'iw+{abs(x_shifts[i])}', f'ih+{abs(y_shifts[i])}', max(0, x_shifts[i]), max(0, y_shifts[i]))
            .filter('scale', 224, 224)
            .filter('hue', h=dh[i], b=db[i])
            .filter('gblur', blurs[i])
        )

        if flip[i] == 1:
            stream = stream.hflip()
        stream = stream.output(f'../augmented_dataset/{id}_{i}.mp4', loglevel='quiet')
        stream.run(overwrite_output=True)

def process_file(id, begin, end):
    frame_list = np.linspace(begin, end - 1, 8).round().astype(int)
    frame_filter = '+'.join([f'eq(n,{f})' for f in frame_list])

    {
        ffmpeg.input(f'../ASL_Citizen/videos/{id}.mp4')
        .filter('select', frame_filter)
        .filter('scale', 224, 224)
        .output(f'../augmented_dataset/{id}.mp4', loglevel='quiet')
        .run(overwrite_output=True)
    }

def augment_videos():
    freq = {}
    val_freq = {}
    with open('survey.csv') as freq_csv:
        reader = csv.reader(freq_csv)
        next(reader)

        for line in reader:
            freq[line[0]] = int(line[1])
            val_freq[line[0]] = 0

    with open('mini_dataset.csv') as in_csv, open('augmented_mini_dataset.csv', 'w') as out_csv:
        reader = csv.reader(in_csv)
        writer = csv.writer(out_csv)
        next(reader)
        writer.writerow(['split', 'file', 'gloss'])

        idx = 0
        for line in reader:
            split = line[0]
            id = line[1]
            if split == 'train':
                num_copies = round(600/freq[line[4]])
                print(str(idx), id, freq[line[4]], num_copies)
                augment_file(id, int(line[2]), int(line[3]), num_copies)
                for i in range(num_copies):
                    writer.writerow([split, f'{id}_{i}', line[4]])
            else:
                if split == 'test' or val_freq[line[4]] < 7:
                    process_file(id, int(line[2]), int(line[3]))
                    writer.writerow([split, id, line[4]])
                    if split == 'val':
                        val_freq[line[4]] += 1
                        print(id, val_freq[line[4]])
            idx += 1