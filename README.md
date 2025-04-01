# BrainInjury_CTScan
# CT Image Analysis Project

## Overview
This project processes brain and bone CT images from the "Computed Tomography CT Images" dataset to perform feature extraction and classification. The dataset is organized under a `Patients_CT` folder with patient subfolders, each containing `bone` and `brain` subfolders with `.jpg` images. The code preprocesses these images, extracts statistical and texture features, and trains a Random Forest classifier with simulated labels (due to no ground truth provided).

## Steps to Download the Dataset
1. Visit the dataset page on Kaggle: [Computed Tomography CT Images](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images).
2. Sign in to your Kaggle account (or create one if needed).
3. Click the "Download" button to get the dataset as a zip file (e.g., `archive.zip`).
4. Unzip the dataset to a local directory or upload it to Google Drive if using Colab. Ensure the `Patients_CT` folder is accessible.

