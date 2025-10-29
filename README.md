# 🛰️ Satellite Image Segmentation

A deep learning project for semantic segmentation of satellite images using a DeepGlobe-inspired architecture based on ResNet34 encoder-decoder model.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Visualization](#visualization)
- [Model Files](#model-files)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a semantic segmentation model for satellite imagery, capable of identifying and segmenting different land cover classes. The model uses a ResNet34 backbone with a custom decoder to perform pixel-level classification on satellite images.

### Key Capabilities
- Semantic segmentation of satellite images into multiple classes
- Support for custom number of segmentation classes (default: 6)
- GPU acceleration support via CUDA
- Easy-to-use training and inference scripts
- Visual mask generation for results

## ✨ Features

- **Deep Learning Model**: ResNet34-based encoder-decoder architecture
- **Flexible Training**: Command-line interface or Jupyter notebook
- **GPU Support**: Automatic CUDA detection and usage
- **Custom Dataset Support**: Easy configuration for your own image-mask pairs
- **Visualization Tools**: Generate colorized segmentation masks
- **Model Persistence**: Save and load trained models

## 📁 Project Structure

```
SATELLITE-IMAGE-SEGMENTATION/
│
├── Satellite Image Segmentation/
│   ├── DeepGloob Model Training (1).ipynb  # Jupyter notebook for training
│   ├── train.py                            # Python training script
│   ├── visualize_mask.py                   # Mask visualization utility
│   └── demo_model.pth                      # Pre-trained model (Git LFS)
│
├── Training Images/                        # Training RGB images
├── Training Mask Images/                   # Training mask images (grayscale)
├── Validation Image/                       # Validation RGB images
├── Va;idation Mask Image/                  # Validation mask images
│
├── .gitattributes                          # Git LFS configuration
├── .gitignore                             # Git ignore rules
└── README.md                              # This file
```

## 📦 Requirements

### Python Packages
- `torch` >= 1.9.0
- `torchvision` >= 0.10.0
- `PIL` (Pillow)
- `numpy`
- `opencv-python` (cv2)
- `matplotlib` (for notebook)

### Hardware
- **CPU**: Any modern CPU
- **GPU**: NVIDIA GPU with CUDA support (optional, recommended for faster training)
- **RAM**: Minimum 8GB recommended
- **Storage**: Space for dataset and model files

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rehan9508/SATELLITE-IMAGE-SEGMENTATION.git
   cd SATELLITE-IMAGE-SEGMENTATION
   ```

2. **Install Python dependencies:**
   ```bash
   pip install torch torchvision pillow numpy opencv-python matplotlib
   ```

   Or install from requirements.txt (if available):
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Git LFS (for large model files):**
   ```bash
   git lfs install
   git lfs pull
   ```

## 📊 Dataset Preparation

### Directory Structure
Your dataset should be organized as follows:

```
Training Images/          # RGB satellite images
  ├── image1.jpg
  ├── image2.jpg
  └── ...

Training Mask Images/    # Grayscale masks (same filenames as images)
  ├── image1.jpg
  ├── image2.jpg
  └── ...

Validation Image/        # Validation RGB images
  ├── val1.jpg
  └── ...

Va;idation Mask Image/   # Validation masks
  ├── val1.jpg
  └── ...
```

### Image Requirements
- **Input Images**: RGB format (JPG, PNG)
- **Mask Images**: Grayscale format (each pixel value represents a class)
- **Image Size**: Any size (automatically resized to 256x256 during training)
- **Mask Values**: Integer values from 0 to (num_classes - 1)

### Notes
- Images and masks must have matching filenames
- Files are sorted alphabetically for pairing
- Mask pixel values should correspond to class indices (0, 1, 2, ..., num_classes-1)

## 💻 Usage

### Training via Python Script

**Basic training:**
```bash
python "Satellite Image Segmentation/train.py" \
    --train-image-dir "Training Images" \
    --train-mask-dir "Training Mask Images" \
    --num-classes 6 \
    --epochs 20 \
    --batch-size 8 \
    --lr 0.001 \
    --output "model.pth"
```

**With validation:**
```bash
python "Satellite Image Segmentation/train.py" \
    --train-image-dir "Training Images" \
    --train-mask-dir "Training Mask Images" \
    --val-image-dir "Validation Image" \
    --val-mask-dir "Va;idation Mask Image" \
    --num-classes 6 \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.001 \
    --output "model.pth"
```

**Dry run (test without data):**
```bash
python "Satellite Image Segmentation/train.py" --dry-run --epochs 1
```

**Force CPU usage:**
```bash
python "Satellite Image Segmentation/train.py" --cpu --train-image-dir "Training Images" ...
```

### Training via Jupyter Notebook

1. Open `Satellite Image Segmentation/DeepGloob Model Training (1).ipynb`
2. Update the dataset paths in the notebook:
   ```python
   train_image_dir = r"path\to\Training Images"
   train_mask_dir = r"path\to\Training Mask Images"
   val_image_dir = r"path\to\Validation Image"
   val_mask_dir = r"path\to\Va;idation Mask Image"
   ```
3. Run all cells to train the model

### Visualization

Colorize and visualize segmentation masks:

```bash
python "Satellite Image Segmentation/visualize_mask.py" \
    --image "path/to/image.jpg" \
    --mask "path/to/mask.jpg" \
    --output "segmented_output.png"
```

## 🏗️ Model Architecture

The model uses an encoder-decoder architecture:

### Encoder
- **Backbone**: ResNet34 (pre-trained on ImageNet)
- **Output**: 512-dimensional feature vector
- **Adaptation**: Classification layer removed, features extracted after average pooling

### Decoder
- **Architecture**: 8-layer transposed convolutional network
- **Upsampling**: Progressively upsamples from 1x1 to 256x256
- **Output**: `num_classes` channels (one per segmentation class)

### Architecture Details
```
Input (3, 256, 256)
  ↓
ResNet34 Encoder
  ↓
Features [B, 512]
  ↓
Reshape to [B, 512, 1, 1]
  ↓
Decoder (8 ConvTranspose2d layers)
  ↓
Output [B, num_classes, 256, 256]
```

## 🎓 Training

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-classes` | 6 | Number of segmentation classes |
| `--epochs` | 1 | Number of training epochs |
| `--batch-size` | 4 | Batch size for training |
| `--lr` | 0.001 | Learning rate for Adam optimizer |
| `--output` | deepglobe_model.pth | Output model file path |

### Loss Function
- **CrossEntropyLoss**: Standard multi-class segmentation loss
- Each pixel is classified independently

### Optimizer
- **Adam optimizer** with default beta parameters
- Learning rate: configurable (default: 0.001)

### Training Tips
- Start with a small number of epochs to verify the pipeline
- Use a batch size that fits in your GPU memory
- Monitor training/validation loss to prevent overfitting
- Use data augmentation for better generalization
- Adjust learning rate if loss plateaus

## 🎨 Visualization

The visualization script generates colorized segmentation masks:

- **Input**: RGB image and grayscale mask
- **Output**: Colorized PNG mask where each class has a unique random color
- **Usage**: Helpful for visual inspection of segmentation results

## 💾 Model Files

- **File Format**: PyTorch state dictionary (`.pth`)
- **Storage**: Large model files (>100MB) are stored via Git LFS
- **Loading**: Use `torch.load()` with `map_location` for CPU/GPU compatibility

Example loading:
```python
import torch
model = DeepGlobeModel(num_classes=6)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- DeepGlobe Challenge for inspiration
- PyTorch team for the deep learning framework
- ResNet architecture (He et al., 2016)

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.

---

**Happy Segmenting! 🛰️✨**
