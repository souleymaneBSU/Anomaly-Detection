# Image Poisoning Attack

This script demonstrates the implementation of various clean-label poisoning techniques, applied to the CIFAR-10 dataset. The goal is to create poisoned images that remain visually similar to their original class but subtly influence the model's feature representation.



## Overview
This code includes multiple poisoning techniques to generate adversarially poisoned images. The poisoned images are saved alongside real, unaltered images to facilitate analysis and training of robust machine learning models.

### Key Features:
- Multiple poisoning techniques: Gradient Matching, Poison Frogs, Metapoison, and more.
- Applied to the CIFAR-10 dataset.
- Generates poisoned and real images for evaluation.

## Setup Instructions

### Prerequisites
- Python 3.9+
- Required libraries:
  - `tensorflow`
  - `numpy`
  - `opencv-python`
  - `Pillow`

Install dependencies using:
```bash
pip install -r ../requirements.tx
```
## Running


```bash
python generate_posoined_img.py
```