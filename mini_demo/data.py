import cv2
import torch
from torchvision.transforms import v2
from concurrent.futures import ThreadPoolExecutor

process = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

# get specific frame
def get_frame(input, frame_num):
    video = cv2.VideoCapture(f'../augmented_dataset/{input}.mp4')
    if not video.isOpened:
        return None
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = video.read()
    video.release()
    if ret:
        return process(frame)
    return None

# transform video data to tensor
def get_video_data(input):
    frame_list = [0, 1, 2, 3, 4, 5, 6, 7]
    frames = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        frames = executor.map(lambda frame_num: get_frame(input, frame_num), frame_list)

    res = torch.empty(8, 3, 224, 224)
    for i, f in enumerate(frames):
        if f is None: # in case video data loading failed
            if i == 0:
                res[i] = torch.zeros(3, 224, 224)
            else:
                res[i] = res[i - 1]
        else:
            res[i] = f
    res = res.permute(1, 0, 2, 3)
    return res

class Dataset(torch.utils.data.Dataset):
    def __init__(self, id_list, label_list, id_to_filename) -> None:
        super().__init__()
        self.id_list = id_list
        self.label_list = label_list
        self.id_to_filename = id_to_filename

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id = self.id_list[index]
        inputs = get_video_data(self.id_to_filename[id])
        label = self.label_list[id]
        return inputs, label