# Train CNN

This repo is my CNN learning example.

## [Update]

Torch version uploaded.

## Introduction

When planning unknown world, the robot (orange arrow) uniformly samples next poses (green arrows) around, computes the gain of each sample, and then chooses the best one to move to. 

The gain function g(\*, \*) is based on two aspects. One is the voxels the robot can newly explore when it is in the pose of the sample, and the other is the cost of the robot to go. The gain computing is expensive. 

**So the idea is to train a CNN-based model predicting the new voxels and the gain of each sample.**
![](https://github.com/yuliangzhong/trainCNN/blob/main/img/data.png)
**observed map | sample and its local map | 25\*25 grid map input**

## Dataset
For each sample, we collect the 25*25 local 0-1-2 grid sub-map around it. Besides, we represent sample pose by a vector [dx, dy, cos(phi), sin(phi)], where phi is the robot orientation. For each sample, we want to predict the new voxels (y1) and the gain (y2).

So the data goes like this: [y1, y2, dx, dy, cosphi, sinphi, 625-flatten-grid-map] \* 200000

[Download data here (about 1GB)](https://drive.google.com/drive/folders/1hUYjd82v9BCHl-uHP1ADMj1uuOm3VWTp?usp=sharing)
 
## Network structure

We want to train a network for new-voxel and gain prediction. The model structure and summary go like this:

|model structure|model summary|
|:-:|:-:|
|![](https://github.com/yuliangzhong/trainCNN/blob/main/modelFig/model1.png)|![](https://github.com/yuliangzhong/trainCNN/blob/main/modelFig/modelSummary.png)|

## Result

For 30000 pieces of test data, the prediction performance and Mean Square Error are shown below.
- Each blue dot is a pair of (predict data, true data);
- Red line is "y=x";
- Closer to the read line, better performance.

![](https://github.com/yuliangzhong/trainCNN/blob/main/img/result4.png)

To verify the gain prediction, we tested the CNN model in the planning simulator. 
- For each case, we compare 15 different samples;
- We predict the new-voxel-to-see (red pair) and gain (blue pair) of each sample;
- The first elements of pairs are true, the second are predicted;
- The uniform planner chooses the best true gain sample (red arrow);
- The gain prediction based planner chooses the best predicted gain sample (purple arrow).
### Result: The gain predictor always give right choices

- Test1
![](https://github.com/yuliangzhong/trainCNN/blob/main/img/test1.png)
- Test2
![](https://github.com/yuliangzhong/trainCNN/blob/main/img/test2.png)
- Test3
![](https://github.com/yuliangzhong/trainCNN/blob/main/img/test3.png)


