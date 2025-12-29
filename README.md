# 🧠 Neural Movement Decoder

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

**Predicting movement directions from multi-channel neural recordings using Deep Learning**

[Overview](#-overview) • [Results](#-results) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage)

</div>

---

## 📖 Overview

This project develops a **neural decoder** that predicts mouse movement directions (Right, Left, Up, Down, No Movement) from 32-channel brain recordings. Using a Multi-Layer Perceptron (MLP) architecture, the system achieves **82.39% validation accuracy** in classifying movement intent from neural signals.

The project demonstrates the practical application of deep learning in **Brain-Computer Interfaces (BCIs)** and neural signal processing.

### Key Achievements

- 🎯 **82.39% validation accuracy** on 5-class movement classification
- 📊 Processed **91,800 neural samples** across 32 channels
- 🔬 Compared hyperparameter tuning effects on model performance
- 📈 Comprehensive evaluation with confusion matrix analysis

## 🧬 The Problem

Decoding movement intention from neural signals is a fundamental challenge in:
- **Brain-Computer Interfaces** - enabling paralyzed patients to control devices
- **Neuroprosthetics** - developing mind-controlled prosthetic limbs
- **Neuroscience Research** - understanding motor cortex function

This project uses real neural recording data to classify 5 movement directions in real-time.

## 📊 Results

### Model Performance Comparison

| Model | Learning Rate | Validation Accuracy |
|-------|--------------|---------------------|
| **Model 1 (Baseline)** | 0.001 | **82.39%** |
| Model 2 (Tuned) | 0.0005 | 78.61% |

### Training Curves

The baseline model shows strong convergence with training and validation accuracy closely tracking each other, indicating good generalization without overfitting.

### Confusion Matrix

The confusion matrix reveals:
- Strong classification of **Up** and **Down** movements
- "No Movement" class shows some confusion with adjacent directions
- Overall balanced performance across all 5 classes

## 🏗️ Architecture

### Model Structure

```
Input Layer (32 neurons)     ← 32 neural channels
        ↓
Dense Layer (128 neurons)    ← ReLU activation
        ↓
Dense Layer (64 neurons)     ← ReLU activation
        ↓
Output Layer (5 neurons)     ← Softmax (5 movement classes)
```

### Data Pipeline

```
Raw Neural Data (.mat)
        ↓
    Normalization (0-1 scaling)
        ↓
    Movement Label Generation (ΔX, ΔY thresholding)
        ↓
    Train/Validation Split (80/20, stratified)
        ↓
    MLP Classification
        ↓
    Movement Prediction
```

## 📁 Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | MATLAB neural recording file |
| **Samples** | 91,800 time points |
| **Neural Channels** | 32 |
| **Position Data** | X, Y coordinates |
| **Classes** | 5 (No Move, Right, Left, Up, Down) |

### Class Distribution

| Class | Label | Samples |
|-------|-------|---------|
| No Movement | 0 | 15,649 |
| Right | 1 | 3,775 |
| Left | 2 | 3,832 |
| Up | 3 | 30,709 |
| Down | 4 | 37,835 |

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/sakshilathi1/neural-movement-decoder.git
cd neural-movement-decoder

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Running the Notebook

```bash
jupyter notebook AI_Project.ipynb
```

### Quick Start

```python
import scipy.io
import numpy as np
from tensorflow import keras

# Load data
data = scipy.io.loadmat('data/delta_reach_20080724-111450-001_Processed.mat')
neural_data = data['neural_data_pro']
mouse_position = data['mouse_position']

# Preprocess and train (see notebook for full implementation)
```

## 📁 Project Structure

```
neural-movement-decoder/
├── AI_Project.ipynb                                    # Main Jupyter notebook with full analysis
├── requirements.txt                                    # Python dependencies
├── README.md                                           # Project documentation
├── .gitignore                                          # Git ignore file
└── data/
    └── delta_reach_20080724-111450-001_Processed.mat   # Neural recording data
```

## 🔬 Methodology

### 1. Data Loading
Load MATLAB `.mat` file containing 32-channel neural recordings and corresponding mouse position data.

### 2. Preprocessing
- **Normalization**: Scale neural data to [0, 1] range
- **Label Generation**: Calculate movement direction from position differences (ΔX, ΔY)
- **Thresholding**: Apply 0.5 threshold to filter noise

### 3. Model Training
- **Architecture**: 2-layer MLP (128 → 64 neurons)
- **Optimizer**: Adam with learning rate 0.001
- **Loss**: Sparse categorical cross-entropy
- **Epochs**: 30 with batch size 64

### 4. Hyperparameter Tuning
- Reduced learning rate from 0.001 to 0.0005
- Compared validation accuracy between configurations

### 5. Evaluation
- Validation accuracy comparison
- Confusion matrix for class-wise analysis

## 📈 Key Findings

1. **Learning Rate Impact**: The default learning rate (0.001) outperformed the reduced rate (0.0005), suggesting the original rate was already well-suited for this dataset.

2. **Class Imbalance**: The dataset shows imbalanced classes, with "Up" and "Down" movements being more frequent. Despite this, the model maintains reasonable performance across all classes.

3. **Generalization**: Training and validation curves closely track each other, indicating the model generalizes well without significant overfitting.

## 🔮 Future Work

- **Advanced Architectures**: Implement RNNs (LSTM/GRU) to capture temporal dependencies
- **CNN Approach**: Use 1D CNNs to detect spatial patterns across neural channels
- **Real-time Decoding**: Optimize for low-latency inference
- **Transfer Learning**: Adapt to different subjects or recording setups
- **Data Augmentation**: Apply time-series augmentation techniques

## 📚 References

- Neural data processing techniques for BCIs
- TensorFlow/Keras documentation for MLP implementation
- scikit-learn for preprocessing and evaluation metrics

## 👤 Author

**Sakshi Lathi**  
Arizona State University  
BME 526: Introduction to Neural Engineering

---

<div align="center">

*This project demonstrates the application of deep learning to neural signal decoding for Brain-Computer Interface applications.*

</div>
