# Orthodontic Image Classification (7 Classes)

## Overview
This project classifies facial/teeth related images i.e. extraoral and intraoral images into 7 categories using ResNet-18. The pipeline uses DVC for versioning and reproducibility.

## Features

- Dataset created from multiple sources
- Splitting into train/val/test via parepare_data.py script
- Training with ResNet-18
- Augmentation (train only)
- Classification Report, Confusion Matrix
- Grad-CAM visualizations
- Tracked with DVC

## Usage

### Setup

```bash
git clone <https://github.com/SandraJervas/orthodontic_image_classification.git>
cd orthodontic_image_classification
pip install -r requirements.txt
dvc pull
