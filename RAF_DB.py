import numpy as np
import torch.utils.data as data
import cv2
import os
import os.path
import face_alignment
import torch
import random
import logging
from utilz import multi_input, data_flip, data_rotate, visual_aggregation


class RAF_DB(data.Dataset):
    def __init__(self, root='',
                 train=True, out_size=224, window_size=49, cnn_only=False):
        self.image_dir = []
        label_dir = ''
        self.labels = []
        self.root = root
        self.train = train
        self.out_size = out_size
        self.window = window_size
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
        self.cnn_only = cnn_only

        with open(label_dir, 'r') as f:
            names = [line.strip().split(" ")[0] for line in f]
        with open(label_dir, 'r') as f:
            labels = [line.strip().split(" ")[1] for line in f]

        for _, _, image_names in os.walk(root):
            for name in image_names:
                if self.train:
                    if "train" in name:
                        self.image_dir.append(name)
                        self.labels.append(labels[names.index(name.replace("_aligned", ""))])
                else:
                    if "test" in name:
                        self.image_dir += [name]
                        self.labels += [labels[names.index(name.replace("_aligned", ""))]]
            break

    def __getitem__(self, item):
        image_path = os.path.join(self.root, self.image_dir[item])
        label = self.labels[item]
        label = int(label) - 1
        if label is None:
            logging.info('Label Error')
            logging.error('Error: CANNOT Find Label from Sequence: {}!'.format(image_path))
            raise ValueError('Error: CANNOT Find Label from Sequence: {}!'.format(image_path))
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        img = cv2.resize(img, dsize=(self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)
        mode = random.randint(0, 1)
        degree = random.uniform(-8, 8)
        if self.train:
            img = data_flip(img, mode=mode)  # 224*224*3
            img = data_rotate(img, degree, size_out=(224, 224))
        if self.cnn_only:
            sequence_ = img.transpose((2, 0, 1)) / 255
            seq_input = torch.from_numpy(sequence_.astype(np.float32))  # Ch, H, W
            emo_label = torch.Tensor([label]).long()
            return torch.zeros((3, 2, 1, 68)), torch.zeros((68, 3, 49, 49)), seq_input, emo_label

        lmarks = self.fa.get_landmarks(img)  # a list of 68 point, i.e. [(x_0, y_0), (x_1, y_1), ...]
        if lmarks is None:
            sequence_ = img.transpose((2, 0, 1)) / 255
            seq_input = torch.from_numpy(sequence_.astype(np.float32))  # Ch, H, W
            emo_label = torch.Tensor([label]).long()
            return torch.zeros((3, 2, 1, 68)), torch.zeros((68, 3, 49, 49)), seq_input, emo_label

        visual = visual_aggregation(img, lmarks[0], window_size=self.window)  # 68*wz*wz*3
        landmarks = lmarks[0]
        visualize = visual.transpose((0, 3, 1, 2)) / 255  # 68*3*wz*wz
        sequence_ = img.transpose((2, 0, 1)) / 255
        geo_input = multi_input(landmarks)[:, np.newaxis, :, :].transpose((0, 3, 1, 2))  # 3, T, P, C -> 3, C, T, P
        geo_input = torch.from_numpy(geo_input.astype(np.float32))  # 3, C, T, P
        vis_input = torch.from_numpy(visualize.astype(np.float32))   # P, Ch, wz, wz
        seq_input = torch.from_numpy(sequence_.astype(np.float32))  # Ch, H, W
        emo_label = torch.Tensor([label]).long()
        return geo_input, vis_input, seq_input, emo_label

    def __len__(self):
        return len(self.labels)
