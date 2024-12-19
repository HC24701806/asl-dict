import numpy as np
import cv2
import torch

# transform video data to tensor
def get_video_data(input):
    video = cv2.VideoCapture(f'./mini_dataset/{input}.mp4')
    frame_list = sorted(np.int32(np.multiply(np.random.normal(0.4, 0.1667, 12), cv2.CAP_PROP_FRAME_COUNT)))
    res = np.zeros((12, 240, 320, 3), dtype='float32')
    i = 0
    for frame in frame_list:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = video.read()
        if not ret:
            break
        res[i] = np.float32(np.array(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 240))))
        i += 1
    video.release()
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