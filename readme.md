# Train CNN

This repo is my CNN learning example.

## Introduction

When planning unknown world, the robot (orange arrow) uniformly samples next poses (green arrows) around, computes the gain of each sample, and then chooses the best one to move to. The gain computing is expensive. So the idea is to train a CNN-based model predicting the gain of each sample.

## Dataset
For each sample, we collect the 25*25 local 0-1-2 grid sub-map around it. Besides, we represent sample pose by a vector [dx, dy, cos(phi), sin(phi)], where phi is the robot orientation. For each sample, we want to predict the new voxels (y1) and the gain (y2).

So the data goes like this: [y1, y2, dx, dy, cosphi, sinphi, 625-flatten-grid-map]
 
## Network structure

We want to train 2 networks for new-voxel and gain prediction.

### new-voxel network

### gain network

## Result


