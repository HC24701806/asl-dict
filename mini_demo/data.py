import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import torch

# get specific frame
def get_frame(input, frame_num):
    video = cv2.VideoCapture(f'./mini_dataset/{input}.mp4')
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = video.read()
    if not ret:
        return None
    return np.asarray(frame, dtype='float32')

# transform video data to tensor
def get_video_data(input):
    video = cv2.VideoCapture(f'./mini_dataset/{input}.mp4')
    frame_list = np.int32(np.multiply(np.random.normal(0.45, 0.25, 8), video.get(cv2.CAP_PROP_FRAME_COUNT)))
    video.release()
    frames = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        frames = executor.map(lambda frame_num: get_frame(input, frame_num), frame_list)

    res = np.empty((8, 240, 320, 3), dtype='float32')
    for i, f in enumerate(frames):
        res[i] = f
    res = np.transpose(res, [3, 0, 1, 2])
    return torch.from_numpy(res)

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, id_list, label_list, id_dict) -> None:
        super().__init__()
        self.id_list = id_list
        self.id_dict = id_dict
        self.label_list = label_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id = self.id_list[index]
        inputs = get_video_data(self.id_dict[id])
        label = self.label_list[id]
        return inputs, label