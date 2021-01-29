import numpy as np
import math
import argparse
import random, torch
import cv2


def multi_input(data):
    P, C = data.shape  # 68 2

    data_new = np.zeros((3, P, C))
    # normalization
    eye_c1 = np.mean(data[36:42, :], axis=0)
    eye_c2 = np.mean(data[42:48, :], axis=0)

    dis = math.sqrt(((eye_c2 - eye_c1) ** 2).sum())
    data = data / dis

    face_c = np.mean(data[0: 17, :], axis=0)
    eyebrow_c1 = np.mean(data[17: 22, :], axis=0)
    eyebrow_c2 = np.mean(data[22: 27, :], axis=0)
    nose_c = np.mean(data[27: 36, :], axis=0)
    eye_c1 = np.mean(data[36: 42, :], axis=0)
    eye_c2 = np.mean(data[42: 48, :], axis=0)
    lips_c = np.mean(data[48: 60, :], axis=0)
    teeth_c = np.mean(data[60: 68, :], axis=0)

    data_new[0, :, :] = data  # A
    for i in range(P):
        data_new[1, i, :] = data[i, :] - data[33, :]  # R
        if i in range(0, 17):
            data_new[2, i, ] = data[i, :] - face_c
        elif i in range(17, 22):
            data_new[2, i, ] = data[i, :] - eyebrow_c1
        elif i in range(22, 27):
            data_new[2, i, ] = data[i, :] - eyebrow_c2
        elif i in range(27, 36):
            data_new[2, i, ] = data[i, :] - nose_c
        elif i in range(36, 42):
            data_new[2, i, ] = data[i, :] - eye_c1
        elif i in range(42, 48):
            data_new[2, i, ] = data[i, :] - eye_c2
        elif i in range(48, 60):
            data_new[2, i, ] = data[i, :] - lips_c
        elif i in range(60, 68):
            data_new[2, i, ] = data[i, :] - teeth_c

    return data_new


def visual_aggregation(img, landmarks, window_size=25):
    stride = int(window_size / 2) + 1
    img = np.pad(img, ((window_size, window_size), (window_size, window_size), (0, 0)), mode='constant')
    visual = np.zeros(shape=(landmarks.shape[0], window_size, window_size, 3))  # 3 is the channels (RGB)
    for i in range(landmarks.shape[0]):
        x, y = int(landmarks[i, 0]), int(landmarks[i, 1])
        crop = img[y+stride:y+stride+window_size, x+stride:x+stride+window_size, :]
        if crop.shape[0] != window_size:
            pad = window_size-(crop.shape[0] % window_size)
            crop = np.pad(crop, ((pad, 0), (0, 0), (0, 0)), mode='constant')
        if crop.shape[1] != window_size:
            pad = window_size-(crop.shape[1] % window_size)
            crop = np.pad(crop, ((0, 0), (pad, 0), (0, 0)), mode='constant')
        visual[i, :, :, :] = crop
    return visual


def data_flip(img, mode=0):  # img: W*H*(x)
    if mode == 0:
        return img
    elif mode == 1:
        return np.fliplr(img)
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.fliplr(np.flipud(img))


def data_rotate(img, degree, size_out=(224, 224)):
    w, h = img.shape[0], img.shape[1]
    center = (int(w/2), int(h/2))
    M = cv2.getRotationMatrix2D(center, degree, 1.0)  # rotation setting
    img = cv2.warpAffine(img, M, size_out)  # rotation and rescaling
    return img


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'TRUE'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'FALSE'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    data = np.random.randn(68, 2)
    multi_input(data)
    print("Done")
