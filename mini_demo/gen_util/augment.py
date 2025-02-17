import csv
import ffmpeg
import numpy as np
import scipy.stats as sci

def augment_file(id, begin, end):
    x_shifts = np.random.randint(-150, 150, 20)
    y_shifts = np.random.randint(-100, 100, 20)
    scales = sci.truncnorm.rvs(-2, 1.333, 0.8, 0.15, 20)
    blurs = sci.truncnorm.rvs(-1.5, 6, 3, 2, 20)
    dh = sci.truncnorm.rvs(-1.25, 1.75, 150, 120, 20)
    db = sci.truncnorm.rvs(-2.5, 2.5, 0, 0.2, 20)

    frame_list = np.linspace(begin, end - 1, 8).round().astype(int)
    frame_filter = '+'.join([f'eq(n,{f})' for f in frame_list])

    for i in range(20):
        {
            ffmpeg.input(f'../ASL_Citizen/videos/{id}.mp4')
            .filter('select', frame_filter)
            .filter('pad', f'iw/{scales[i]}', f'ih/{scales[i]}', f'(ow-iw)/2', f'(oh-ih)/2')
            .filter('scale', f'iw*{scales[i]}', f'ih*{scales[i]}')
            .filter('crop', f'iw-{abs(x_shifts[i])}', f'ih-{abs(y_shifts[i])}', abs(min(0, x_shifts[i])), abs(min(0, y_shifts[i])))
            .filter('pad', f'iw+{abs(x_shifts[i])}', f'ih+{abs(y_shifts[i])}', max(0, x_shifts[i]), max(0, y_shifts[i]))
            .filter('scale', 'iw*224/ih', '224')
            .filter('crop', 224, 224, '(iw-224)/2', '(ih-224)/2')
            .filter('hue', h=dh[i], b=db[i])
            .filter('gblur', blurs[i])
            .output(f'../augmented_dataset/{id}_{i}.mp4', loglevel='quiet')
            .run(overwrite_output=True)
        }

def process_file(id, begin, end):
    frame_list = np.linspace(begin, end - 1, 8).round().astype(int)
    frame_filter = '+'.join([f'eq(n,{f})' for f in frame_list])

    {
        ffmpeg.input(f'../ASL_Citizen/videos/{id}.mp4')
        .filter('select', frame_filter)
        .filter('scale', 'iw*224/ih', '224')
        .filter('crop', 224, 224, '(iw-224)/2', '(ih-224)/2')
        .output(f'../augmented_dataset/{id}.mp4', loglevel='quiet')
        .run(overwrite_output=True)
    }

def augment_videos():
    with open('mini_dataset.csv') as in_csv, open('augmented_mini_dataset.csv', 'w') as out_csv:
        reader = csv.reader(in_csv)
        writer = csv.writer(out_csv)
        next(reader)
        writer.writerow(['split', 'file', 'gloss'])

        for line in reader:
            split = line[0]
            id = line[1]
            if split == 'train':
                augment_file(id, int(line[2]), int(line[3]))
                for i in range(20):
                    writer.writerow([split, f'{id}_{i}', line[4]])
            else:
                process_file(id, int(line[2]), int(line[3]))
                writer.writerow([split, id, line[4]])