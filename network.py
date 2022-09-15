import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
# import random
import torchvision.models as models
# from torchsummary import summary
import numpy as np
from dataloader import Train

from PIL import Image
import time
import cv2


class Model(nn.Module):
    def __init__(self):
        ###INPUT MODULES###
        # mobile net takes images of size (224x224x3) as input

        super(Model, self).__init__()
        self.mobile_net = models.mobilenet_v2(num_classes=512)
        self.image_module = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),

        )
        self.imu_module = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
        )

        self.depth_module = nn.Sequential(  # input of size: 200x88)
            nn.Conv2d(1, 32, 4, 2, 1, bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1, bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1, bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Flatten(),

            nn.Linear(3072, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(True),

            nn.Linear(512, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(True)
        )

        # print("SUMMARY DEPTH MODULE")
        # print(summary(self.depth_module,(1,88,200)))
        self.speed_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True)
        )

        self.dense_layers = nn.Sequential(
            # 512depth, 512image, 128speed, 256imu
            nn.Linear(1408, 512),
            nn.ReLU(True)
        )

        ###COMMAND BRANCHEs###
        self.straight_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )

        self.right_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )
        self.left_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            print("Block:", block)
            if block == 'mobile_net':
                pass
            else:
                for m in self._modules[block]:
                    if isinstance(m, nn.Conv2d):
                        # nn.init.normal(m.weight, mean=0, std=0.01)
                        nn.init.xavier_uniform(m.weight)
                    elif isinstance(m, nn.Linear):
                        # nn.init.normal(m.weight, mean=0, std=0.01)
                        nn.init.xavier_uniform(m.weight)
                    else:
                        pass

    def forward(self, input_data):
        image = input_data[0]
        image = self._image_module(image)
        imu = input_data[2]
        imu = self._imu_module(imu)
        depth = input_data[1][:, None, :, :]
        depth = self._depth_module(depth)
        speed = input_data[3]
        speed = self._speed_module(speed)
        command = input_data[4]

        concat = torch.cat((image, imu, depth, speed), 1)
        concat = self._dense_layers(concat)

        output = torch.Tensor()
        batch_size =len(image)
        for i in range(batch_size):
            if command[i] == 1:
                if output.shape == torch.Size([0]):
                    output = self._straight_net(concat)
                else:
                    torch.stack(tensors=(output, self._straight_net(concat)), dim=1)

            elif command[i] == 2:
                if output.shape == torch.Size([0]):
                    output = self._right_net(concat)
                else:
                    torch.stack(tensors=(output, self._right_net(concat)), dim=1)

            elif command[i] == 0:
                if output.shape == torch.Size([0]):
                    output = self._left_net(concat)
                else:
                    torch.stack(tensors=(output, self._left_net(concat)), dim=1)

        return output

    def _image_module(self, x):
        x = self.mobile_net(x)
        return self.image_module(x)

    def _imu_module(self, x):
        return self.imu_module(x)

    def _depth_module(self, x):
        return self.depth_module(x)

    def _speed_module(self, x):
        return self.speed_module(x)

    def _dense_layers(self, x):
        return self.dense_layers(x)

    def _straight_net(self, x):
        return self.straight_net(x)

    def _right_net(self, x):
        return self.right_net(x)

    def _left_net(self, x):
        return self.left_net(x)


class Solver:
    def __init__(self, use_cuda, model):
        self.MSE_loss = torch.nn.MSELoss()
        self.model = model
        self.cuda = use_cuda
        self.loss_steer = []
        self.loss_throttle = []
        self.loss_brake = []
        self.step = 0
        self.optimizer = optim.Adam(model.parameters(), lr=5e-4)

    def train_iteration(self, dataloader, tb_writer):        
        self.model.train()
        if self.cuda == True:
            self.model = self.model.cuda()

        for batch_data, batch_label in dataloader:
            if self.cuda == True:
                for j in range(len(batch_data)):
                    batch_data[j] = torch.Tensor(batch_data[j]).cuda()
                batch_label = torch.vstack(batch_label).cuda()

            output = self.model(batch_data)
            batch_label = torch.stack((batch_label[0].to(torch.float32), batch_label[1].to(torch.float32), batch_label[2].to(torch.float32)),1)
            loss = self.MSE_loss((output), batch_label)
            print(f"[Training Epoch:{self.step}] loss={loss.item()}")
            tb_writer.add_scalar('train/loss', loss.item(), self.step)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.step += 1


    def eval_iteration(self, dataloader, tb_writer, mode):
        self.model.eval()
        if self.cuda == True:
            self.model = self.model.cuda()

        for batch_data, batch_label in dataloader:
            if self.cuda == True:
                for j in range(len(batch_data)):
                    batch_data[j] = torch.Tensor(batch_data[j]).cuda()
                batch_label = torch.vstack(batch_label).cuda()

            output = self.model(batch_data)
            batch_label = torch.stack((batch_label[0].to(torch.float32), batch_label[1].to(torch.float32), batch_label[2].to(torch.float32)),1)            
            loss = self.MSE_loss((output), batch_label)
            print(f"[{mode} Epoch:{self.step}] loss={loss.item()}")
            tb_writer.add_scalar(f'{mode}/loss', loss.item(), self.step)
            self.step += 1


if __name__ == "__main__":
    # PARAMS
    batch_size = 32
    num_epochs = 5
    cuda_use = True

    print("STARTING")
    # Change train flag to "train" to do image augmentation for rgb images
    train_data = Train(data_dir='./training_data/', train_eval_flag="no_train")

    # Change the data_dir for test
    test_data = Train(data_dir='./test_data/', train_eval_flag="no_train")
    val_data = Train(data_dir='./val_data/', train_eval_flag="no_train")



    print("INITIALIZED TRAIN AND TEST SET")
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # hparams = {
    #     'learning_rate': 0.001
    # }
    model = Model()
    writer = SummaryWriter()


    solver = Solver(use_cuda=cuda_use, model=model)
    list_test_loss = []
    list_train_loss = []

    for i in range(num_epochs):
        print("##############ITERATION NO {}###############".format(i))
        solver.train_iteration(train_dataloader, writer)
        torch.save(model, "./new_model.pt")
        solver.eval_iteration(val_dataloader, writer, mode='Validation')

    #writer.add_hparams(hparam_dict=hparams, metric_dict={'train_loss': 0.1, 'val_loss': 1})