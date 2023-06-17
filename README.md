# Fundamentals-of-Artificial-Intelligence-and-Deep-Learning-Frameworks

This repository contains the assignments for the course ENPM 809K - Fundamentals of Artificial Intelligence and Deep Learning Frameworks, which is an adaptation of the course CS231n - Deep Learning for Computer Vision. The original course had 3 main assignments, which had multiple submodules that were supposed to be submitted separately.

## Module 1 - Image Classification

In this module we perform Image Classification using various algorithms and test their performances.

### kNN

We use the CIFAR10 dataset and classify images by using a simple k Nearest Neighbors classifier to generalize masks for image classes.

### SVM

Multi-class Support Vector Machines are implemented. The resultant generalizations are shown below:

![svm](https://github.com/Kelebrimbor97/Fundamentals-of-Artificial-Intelligence-and-Deep-Learning-Frameworks/assets/35636842/bd4fa62b-3fcd-4331-90e0-976905e2b183)

### Softmax

The softmax classifier is implemented here with the following generalizations:

![softmax](https://github.com/Kelebrimbor97/Fundamentals-of-Artificial-Intelligence-and-Deep-Learning-Frameworks/assets/35636842/a010d9dc-17f6-4ef6-a495-fea2f4463106)

### Two layer net

We implement a fully connected 2 layer Neural Network with ReLU activation and Softmax/SVM classifier for the Loss function.

### Feature based classifier

Finally, we extract features from the images, specifically the Histogram Of Gradients (HOG) of the hue channel from HSV color space to perform classification. These features are then used to train the Neural Network implemented in the last exercise.

## Module 2 - Fully Connected Networs

### Batch Normalization
