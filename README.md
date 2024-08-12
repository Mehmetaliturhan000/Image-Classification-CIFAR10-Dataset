# Image-Classification-CIFAR10-Dataset
Custom CNN and RESNET50 Supported CNN Implementations on CIFAR10 Dataset

This project implements deep learning models to classify images in the CIFAR-10 dataset using Convolutional Neural Networks (CNN) and ResNet50.

Introduction
This project aims to classify images into one of ten categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using deep learning models. We use both a custom CNN and a pre-trained ResNet50 model to achieve this task.

# Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

# Installation
To run this project, you need to have Python and the following libraries installed:

TensorFlow
Keras
NumPy
Matplotlib You can install these dependencies using pip: pip install tensorflow keras numpy matplotlib
Model Architecture
CNN Model
Convolutional layers with ReLU activation
ResNet50 Model
Fine-tuned ResNet50 pre-trained on ImageNet
Global average pooling
Results
The models were evaluated using accuracy and confusion matrices. The CNN model achieved an accuracy of [36%], while the ResNet50 model achieved an accuracy of [94%].
