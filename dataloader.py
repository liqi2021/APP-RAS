import numpy as np
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
import torch
from imgaug import augmenters as iaa
from helper import RandomTransWrapper, normalize_imu, normalize_speed, normalize_steering, normalize_image
import h5py
import glob


class Train(Dataset):
    def __init__(self, data_dir, train_eval_flag="train", sequence_len=200):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+'*.h5')
        self.data_list.sort()
        self.sequnece_len = sequence_len
        self.train_eval_flag = train_eval_flag

        self.build_transform()

    def build_transform(self):
        if self.train_eval_flag == "train":
            self.transform = T.Compose([
                T.RandomOrder([
                    RandomTransWrapper(
                        seq=iaa.GaussianBlur(
                            (0, 1.5)),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.05),
                            per_channel=0.5),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.Dropout(
                            (0.0, 0.10),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.CoarseDropout(
                            (0.0, 0.10),
                            size_percent=(0.08, 0.2),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Add(
                            (-20, 20),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Multiply(
                            (0.9, 1.1),
                            per_channel=0.2),
                        p=0.4),
                    RandomTransWrapper(
                        seq=iaa.ContrastNormalization(
                            (0.8, 1.2),
                            per_channel=0.5),
                        p=0.09),
                ]),
                T.ToTensor()])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        with h5py.File(file_name, 'r') as h5_file:
            img_rgb = np.array(h5_file['rgb_front/image']).astype(np.float32)
            img_rgb = self.transform(img_rgb)

            img_depth = normalize_image(np.array(h5_file['depth_front/image']).astype(np.float32))
            img_depth = torch.Tensor(img_depth)

            others = np.array(h5_file['others']).astype(np.float32)

            imu = normalize_imu(others[:10])

            speed = normalize_speed(others[10])

            command = others[11]

            # Acceleration, Brake, Steer
            label = [others[12],  others[14], normalize_steering(others[13])]

            data    = [
                img_rgb,
                img_depth,
                torch.Tensor(np.array(imu)),
                torch.Tensor(np.array([speed])),
                torch.Tensor(np.array([command]))
                       ]
        return data, label


if __name__ == '__main__':
        data_train = Train(
            data_dir='./data_withbrake/',
            train_eval_flag="train")

        data, label = data_train.__getitem__(1545)
        print(data)