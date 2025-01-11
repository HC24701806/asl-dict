# remember to set "export PYTORCH_ENABLE_MPS_FALLBACK=1" before running

import numpy as np
import cv2
import torch
from torchvision.transforms import v2
from concurrent.futures import ThreadPoolExecutor
from model import Model

def load_model(path):
    model = Model()
    saved = torch.load(path, map_location=torch.device('mps'))
    model.load_state_dict(saved['model'])
    model = model.to('mps')
    model = model.eval()
    return model

def predict(frames, model):
    # process data
    process = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        v2.Resize((224, 224))
    ])
    frame_list = np.linspace(0, len(frames) - 1, 16).round().astype(int)
    processed_frames = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        processed_frames = executor.map(lambda frame_num: process(frames[frame_num]), frame_list)

    data = torch.empty(1, 16, 3, 224, 224)
    for i, f in enumerate(processed_frames):
        data[0][i] = f
    data = data.permute(0, 2, 1, 3, 4)

    # predict
    
    data = data.to('mps')
    outputs = model(data)
    post_act = torch.nn.Softmax(dim=1)
    pred = post_act(outputs).topk(k=3)
    return pred

def run_model(path):
    # load model
    model = load_model(path)

    # label to class
    classes = []
    with open('sample_classes.txt') as labels_file:
        content = labels_file.readlines()
        for line in content:
            classes.append(line[:-1])

    # video input
    video = cv2.VideoCapture(0)
    frames = []
    recording = False
    pred_classes = None
    pred_probs = None
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if recording:
            frames.append(frame)
            cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
        elif pred_classes != None:
            for i in range(3):
                cv2.putText(frame, f'{classes[pred_classes[i]]}: {pred_probs[i]}', (50, 50 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow('video', frame)

        k = cv2.waitKey(1)
        if k == ord(' '):
            if recording:
                pred = predict(frames, model)
                pred_classes = pred.indices[0].to('cpu')
                pred_probs = pred.values[0].detach().to('cpu').numpy()
                frames = []
            recording = not recording
        elif k == ord('q'):
            break