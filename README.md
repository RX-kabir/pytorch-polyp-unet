# 🩺 PyTorch U-NET for Gastrointestinal Polyp Segmentation

This repository contains the implementation of a U-NET convolutional neural network in PyTorch for the task of **gastrointestinal polyp segmentation**. The goal is to accurately predict the binary mask of polyps from colonoscopy images, which is a critical step in automated medical image analysis.

The project uses a custom dataset structure for loading images and their corresponding ground truth masks, along with PyTorch DataLoaders and Albumentations for robust data augmentation.

## ✨ Features

* **U-NET Architecture:** Full implementation of the classic U-NET model for semantic segmentation.
* **Custom Dataset:** `PolypDataset` class to efficiently load and preprocess image/mask pairs.
* **Data Augmentation:** Utilizes the Albumentations library for on-the-fly transformations (e.g., resizing, normalization, and flips).
* **Metrics:** Implements the **Dice Coefficient** (F1-score) and pixel accuracy for evaluation.
* **Mixed Precision Training:** Uses `torch.cuda.amp` for faster training and reduced memory consumption on compatible GPUs.

