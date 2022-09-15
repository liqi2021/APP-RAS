
import os
import random
import shutil
import numpy as np
import torch


class TransWrapper(object):
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, img):
        return self.seq.augment_image(img)


class RandomTransWrapper(object):
    def __init__(self, seq, p=0.5):
        self.seq = seq
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        return self.seq.augment_image(img)


max_imu = np.array([ 9.8419981,10.21207809,10.56573486, 10.91194344, 11.23107529, 11.55004311, 11.885849,   12.21730995, 12.54627323, 12.87323761])
min_imu = np.array([-1.69137157e-02, -5.93055412e-03, -3.24095320e-03, -9.20971972e-04, -5.17153597e+00, -1.02396202e+01, -1.48390427e-01, -7.01377913e-02, -3.38135324e-02, -1.66577529e-02])


def normalize_imu(imu):
    global max_imu, min_imu
    imu = (imu-min_imu)/(max_imu-min_imu)
    return imu


def normalize_speed(speed):
    maxi=12.177567
    mini=-5.12704
    speed=(speed-mini)/(maxi-mini)
    return speed


def normalize_steering(steering):
    steering=(steering+90.0)/180.0
    return steering


def normalize_image(image):
    return image/255.0


def save_checkpoint(state, id_, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename,
            os.path.join("save_models", "{}_best.pth".format(id_))
            )