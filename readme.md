# Train CNN

This repo is my CNN learning example.

## Introduction

When planning unknown world, the robot (orange arrow) uniformly samples next poses (green arrows) around, computes the gain of each sample, and then chooses the best one to move to. 

The gain function g(\*, \*) is based on two aspects, one is the voxels the robot can newly explore when it is in the pose of the sample, and the other is the cost of the robot to go. The gain computing is expensive. So the idea is to train a CNN-based model predicting the new voxels and the gain of each sample.
![](https://github.com/yuliangzhong/trainCNN/blob/main/img/data.png)
**observed map | sample and its local map | 25\*25 grid map input**

## Dataset
For each sample, we collect the 25*25 local 0-1-2 grid sub-map around it. Besides, we represent sample pose by a vector [dx, dy, cos(phi), sin(phi)], where phi is the robot orientation. For each sample, we want to predict the new voxels (y1) and the gain (y2).

So the data goes like this: [y1, y2, dx, dy, cosphi, sinphi, 625-flatten-grid-map]

[Download data here (about 1GB)](https://drive.google.com/drive/folders/1hUYjd82v9BCHl-uHP1ADMj1uuOm3VWTp?usp=sharing)
 
## Network structure

We want to train a network for new-voxel and gain prediction. The structure and summary go like this:

|:-:|:-:|
|![]()|![]()|
|model structure|model summary|

## Result

For 30000 pieces of test data, the prediction performance and MSE error are shown below:

![]()

To verify the performance, we test the model in the planning simulator. We compare 15 different samples, predicting the new-voxel-to-see and gain of each sample. The results go as following:

|:-:|:-:|:-:|
|![]()|![]()|![]()
|test1|test2|test3


