# APP-RAS
Applications of Robotics and Autonomous Systems (Imitation Learning for small-scale car)

Guide of files:
Dataloader.py :  Creates the data loader class for training and main script will test that it works
Helper.py :  Helper methods for data augmentation and normalization
Network.py :  Creates the network model class and main script will start the training (there are parameter inside for batch_size, cuda, etc)
Calc_norms : This notebook will calculate the mean from the dataset so we can normalize later (the means are hardcoded in helper.py)
Inference : This noteboook shows how to use the trained network for inferencing

Notes:
There is one small note to take into account. It seems that the use of the model is sensitive to the version of torchvision that was used creating the model.
