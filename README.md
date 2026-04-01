# CIFAR-10 CNN Classifier

A deep learning project by **Yassin** 🧑‍💻

This repository contains a convolutional neural network (CNN) built with PyTorch to classify images from the CIFAR-10 dataset. The model uses batch normalization, dropout, and data augmentation to improve performance.

---

## 🚀 Features
- Custom CNN architecture with 3 convolutional layers
- Batch normalization and dropout for regularization
- Data augmentation (random crop, horizontal flip)
- Training with Adam optimizer and learning rate scheduler
- Saves trained model as `cifar10_cnn.pth`
- Plots training loss and validation accuracy

---

## 📦 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Yassin/cifar10-cnn.git
cd cifar10-cnn
pip install -r requirements.txt