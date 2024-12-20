import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import torch

# process frame (replacement for transforms)
def process(frame):
    frame = np.divide(np.subtract(np.divide(frame, 255.0), 0.45), 0.225)
    return cv2.resize(frame, (256, 256))

# get specific frame
def get_frame(input, frame_num):
    video = cv2.VideoCapture(f'../ASL_Citizen/videos/{input}.mp4')
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = video.read()
    if not ret:
        return None
    return process(frame)

# transform video data to tensor
def get_video_data(input, begin_frame, end_frame):
    frame_list = np.linspace(begin_frame, end_frame, 16).round().astype(int)
    frames = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        frames = executor.map(lambda frame_num: get_frame(input, frame_num), frame_list)

    res = np.empty((16, 256, 256, 3), dtype='float32')
    for i, f in enumerate(frames):
        res[i] = f
    res = np.transpose(res, [3, 0, 1, 2])
    return torch.from_numpy(res)

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, id_list, label_list, id_to_filename, video_info) -> None:
        super().__init__()
        self.id_list = id_list
        self.label_list = label_list
        self.id_to_filename = id_to_filename
        self.video_info = video_info

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id = self.id_list[index]
        inputs = get_video_data(self.id_to_filename[id], self.video_info[id][0], self.video_info[id][1])
        label = self.label_list[id]
        return inputs, label