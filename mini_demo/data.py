import numpy as np
import cv2
import os
from torchvision.transforms import v2
from concurrent.futures import ThreadPoolExecutor
import torch

process = v2.Compose([
    v2.ToTensor(),
    v2.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    v2.Resize((256, 256)),
    v2.CenterCrop(256)
])

# get specific frame
def get_frame(input, frame_num):
    video = cv2.VideoCapture(f'../augmented_dataset/{input}.mp4')
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = video.read()
    video.release()
    if ret:
        return process(frame)
    return None

# transform video data to tensor
def get_video_data(input, begin_frame, end_frame):
    frame_list = np.linspace(begin_frame, end_frame - 1, 16).round().astype(int)
    frames = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        frames = executor.map(lambda frame_num: get_frame(input, frame_num), frame_list)

    res = torch.empty(16, 3, 256, 256)
    for i, f in enumerate(frames):
        if f is None: # in case video data loading failed
            print(input, frame_list[i])
            if i == 0:
                res[i] = torch.zeros(3, 256, 256)
            else:
                res[i] = res[i - 1]
        else:
            res[i] = f
    res = res.permute(1, 0, 2, 3)
    return res

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