# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32

from sensor_msgs.msg import Imu
import numpy as np
from skimage.transform import resize

import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision.transforms as T
import time
import torchvision

import carla 
from carla_msgs.msg import CarlaEgoVehicleControl

class Model(nn.Module):
    def __init__(self):
        ###INPUT MODULES###
        #mobile net takes images of size (224x224x3) as input
        
        super(Model,self).__init__()
        self.mobile_net = models.mobilenet_v2(num_classes=512)
        self.image_module = nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(True),
            nn.Linear(1024,512),
            nn.ReLU(True),

            )
        self.imu_module = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            )

        self.depth_module = nn.Sequential(#input of size: 200x88)
            nn.Conv2d(1,32,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32,32,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32,64,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,64,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
          #  nn.Conv2d(64,128,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(128),
          #  nn.ReLU(True),
          #  nn.Conv2d(128,128,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(128),
          #  nn.ReLU(True),

          #  nn.Conv2d(128,256,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(256),
          #  nn.ReLU(True),
          #  nn.Conv2d(256,256,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(256),
          #  nn.ReLU(True),
            
        
            nn.Flatten(),

           #input to linear layer probably incorrect
           # nn.Linear(384 , 512),
           #  nn.Linear(3840,512),
            nn.Linear(3072,512),
            nn.Dropout(p=0.5),
            nn.ReLU(True),

            nn.Linear(512,512),
            nn.Dropout(p=0.5),
            nn.ReLU(True)
            )

        print("SUMMARY DEPTH MODULE")
        # print(summary(self.depth_module,(1,88,200)))
        self.speed_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True)
            )

        self.dense_layers = nn.Sequential(
            #512depth, 512image, 128speed, 256imu
            nn.Linear(1408,512),
            nn.ReLU(True)
            )

        ###COMMAND BRANCHEs###
        self.straight_net = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,3),  #outputlayer is supposed to have no activation function
            nn.Sigmoid()
            )

        self.right_net = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,3),  #outputlayer is supposed to have no activation function
            nn.Sigmoid()
            )
        self.left_net = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,3),  #outputlayer is supposed to have no activation function
            nn.Sigmoid()
            )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            print("Block:", block)
            if block=='mobile_net':
                pass
            else:
                for m in self._modules[block]:
                    print("M:", m)
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal(m.weight,mean=0, std=0.01)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal(m.weight,mean=0,std=0.01)
                    else:
                        pass
               # normal_init(m)

    def forward(self, input_data):
        image = input_data[0]
        image = self._image_module(image)
        imu   = input_data[2]
        imu   = self._imu_module(imu)
        depth = input_data[1]
        depth = self._depth_module(depth)
        speed = input_data[3]
        speed = self._speed_module(speed)
        command=input_data[4]
        
        concat = torch.cat((image,imu,depth,speed),1)
        concat = self._dense_layers(concat)
        
        ##########################################################
        ##1 here needs to be changed if batch size is changed!!##
        ##########################################################
        output = torch.Tensor()
        for i in range(1):
            if command[i]==1:
                if output.shape==torch.Size([0]):
                    output = self._straight_net(concat)
                else:
                    torch.stack(tensors=(output, self._straight_net(concat)),dim=1)

            elif command[i]==2:
                if output.shape==torch.Size([0]):
                    output = self._right_net(concat)
                else:
                    torch.stack(tensors=(output, self._right_net(concat)),dim=1)

            elif command[i]==0:
                if output.shape==torch.Size([0]):
                    output = self._left_net(concat)
                else:
                    torch.stack(tensors=(output, self._left_net(concat)),dim=1)

        return output

    
    def _image_module(self,x):
        x = self.mobile_net(x)
        return self.image_module(x)
        
    def _imu_module(self,x):
        return self.imu_module(x)

    def _depth_module(self,x):
        return self.depth_module(x)

    def _speed_module(self,x):
        return self.speed_module(x)

    def _dense_layers(self,x):
        return self.dense_layers(x)

    def _straight_net(self,x):
        return self.straight_net(x)

    def _right_net(self,x):
        return self.right_net(x)

    def _left_net(self,x):
        return self.left_net(x)



rgb_cropsize=(224,224,3)
depth_cropsize=(70,200)


def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = rgba.shape
        
    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8')
    
def rgba2gray(rgba):
    gray = np.mean(rgba[...,:3], -1)
    return gray


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.publisher_ = self.create_publisher(CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)
        
        self.cam_sub = self.create_subscription(
            Image, 
            '/carla/ego_vehicle/rgb_front/image',
            self.camera_callback,
            10)
            
        self.imu_sub = self.create_subscription(
	        Imu, 
	        '/carla/ego_vehicle/imu',
	        self.imu_callback,
	        10)
	    
	                
        self.depth_sub = self.create_subscription(
	        Image, 
	        '/carla/ego_vehicle/depth_front/image',
	        self.depth_callback,
	        10)
	    
        self.speedometer_sub = self.create_subscription(
	        Float32, 
	        '/carla/ego_vehicle/speedometer',
	        self.speedometer_callback,
	        10)



        self.image = None
        self.depth = None
        self.imu = None
        self.speed = None
        
        self.model = Model()
        #self.model = torch.load("/home/pei/carla-ros-bridge/src/py_pubsub/py_pubsub/model_whole2.pt")
        self.model.eval()
     
    def provide_output(self):
        if self.image is None or self.depth is None or self.imu is None or self.speed is None :
            print("skip")
            return
        i = torch.Tensor(self.image).permute(2,0,1)
        print(i.shape)
        data = [torch.Tensor(self.image).permute(2, 0, 1)[None, :], torch.Tensor(self.depth)[None, None, :], torch.Tensor(self.imu)[None, :], torch.Tensor([self.speed])[None, :], torch.Tensor([0])[None, :]]
        output = self.model.forward(data)
        print(output)
        self.publisher_.publish(CarlaEgoVehicleControl(throttle=float(output[0,0]), steer=float(output[0,2]), brake=float(0.0)))    

	    
	    
    def camera_callback(self, msg):
        img = np.reshape(msg.data, (msg.height, msg.width, 4))
        img = rgba2rgb(img)
        img = resize(img, rgb_cropsize)
        #img = np.expand_dims(img, axis=0)
        self.image = img
        print(img.shape)
        print("get camera")
        
        self.provide_output()
        
    def depth_callback(self, msg):
        img = np.reshape(msg.data, (msg.height, msg.width, 4))
        img = rgba2gray(img)
        img = resize(img, depth_cropsize)
        #img = np.expand_dims(img, axis=0)
        self.depth = img
        print("get depth")
        
    def imu_callback(self, msg):
        imu = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        #imu = np.expand_dims(imu, axis=0)
        self.imu = imu
        print("get imu")
                
    def speedometer_callback(self, msg):
        speed = msg.data
        #speed = np.expand_dims(speed, axis=0)
        self.speed = speed
        print("get speed")
        



def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    print(sys.version)
    main()
