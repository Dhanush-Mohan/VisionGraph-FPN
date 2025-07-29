#  VisionGraph-FPN: Graph-Enhanced FPN for Pneumonia Detection

VisionGraph-FPN is an advanced deep learning framework designed for accurate pneumonia detection in chest X-ray (CXR) images. By integrating **Graph Neural Networks (GNNs)** with **Faster R-CNN** and **Feature Pyramid Networks (FPN)**, it enhances spatial and contextual feature representations, outperforming traditional convolutional neural network (CNN)-based object detectors. This project leverages graph-based feature extraction and multi-scale learning to achieve state-of-the-art performance on the RSNA Pneumonia Detection Challenge dataset.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Visualizations](#visualizations)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Training](#training)
- [Future Scope](#future-scope)
- [License](#license)
- [Citation](#citation)

---

## 🧩 Overview

Pneumonia is a critical respiratory condition contributing significantly to global morbidity and mortality. Manual interpretation of CXR images is time-consuming and subject to variability among radiologists. VisionGraph-FPN addresses these challenges by combining isotropic graph convolutions, pyramid feature hierarchies, and UNet-style skip connections to detect spatially scattered anomalies, such as pneumonia lesions, with high accuracy and robustness.

The framework processes CXR images by dividing them into patches, treating each as a node in a graph. Edges are constructed based on spatial proximity and feature similarity, enabling global contextual learning. A Deep Graph Convolutional Network (Deep-GCN) refines these representations through a Grapher module and Feedforward Network (FFN), which are then fed into a Faster R-CNN detection head with a custom FPN for precise localization and classification.

---

## ⚙️ Key Features

- **Graph-Based Learning**: Captures non-local dependencies using isotropic graph convolutions.
- **Custom Feature Pyramid Network**: Enhances multi-scale feature extraction for diverse pneumonia manifestations.
- **UNet-Inspired Skip Connections**: Improves gradient flow and feature integration via ConvTranspose2D upsampling.
- **Superior Performance**: Achieves higher detection accuracy and localization precision compared to CNN-based models like VGG16, ResNet50, and EfficientNet-B3.
- **Robust Generalization**: Performs well across diverse patient datasets, reducing false positives and negatives.

---

## 🧱 Model Architecture

The VisionGraph-FPN architecture comprises the following components:

- **Backbone**: Isotropic Vision Graph Neural Network (ViG) for graph-based feature extraction.
- **Neck**: Custom Feature Pyramid Network (FPN) with pyramid-style ViG and skip connections for multi-scale learning.
- **Head**: Faster R-CNN detection head for region proposal generation, classification, and bounding box refinement.
- **Dataset**: Trained and evaluated on the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset.

*Architecture Diagram*: <img width="4050" height="2570" alt="VisionGraphFPN_FinalDiagram" src="https://github.com/user-attachments/assets/99e4a9a8-2f13-47fd-8971-44b591137577" />


The workflow involves:
1. **Image Preprocessing**: Dividing CXR images into structured patches.
2. **Graph Construction**: Creating nodes (patches) and edges (spatial/feature similarity).
3. **Feature Extraction**: Deep-GCN with Grapher module and FFN for enhanced representations.
4. **Region Proposals**: Region Proposal Network (RPN) identifies potential pneumonia-affected areas.
5. **Classification & Localization**: Final classification and bounding box refinement for precise detection.

*Workflow Diagram*: <img width="3995" height="1810" alt="VisionGraphFPN_workflow drawio" src="https://github.com/user-attachments/assets/4f68bfc0-859e-47e3-8598-fcaef06f5320" />


---

## 📊 Results

### 📉 Training Loss Convergence

| Epochs | Training Loss |
|--------|---------------|
| 100    | 1.3523        |
| 200    | 1.0269        |
| 400    | **0.9575**    |

### 🆚 Baseline IoU-Accuracy Comparison

| Model              | IoU Accuracy |
|--------------------|--------------|
| VisionGraph-FPN    | **0.8571**   |
| VGG16              | 0.8000       |
| MobileNetV3-Large  | 0.6666       |
| ResNet50           | 0.5714       |
| EfficientNet-B3    | 0.5238       |
| DenseNet121        | 0.5000       |

VisionGraph-FPN outperforms baseline models due to its graph-enhanced feature modeling, leading to better differentiation between pneumonia-affected and normal lung regions.

---

## 🩻 Visualizations

- *Bounding Boxes*: <img width="992" height="465" alt="image" src="https://github.com/user-attachments/assets/d1ce4555-d4d6-4439-8f90-377fd5230b31" />
- *Anchor Points*: <img width="1015" height="473" alt="image" src="https://github.com/user-attachments/assets/36651d6e-e2ce-4bbc-a566-b77a255bc0ae" />
- *Anchor Boxes*: <img width="968" height="458" alt="image" src="https://github.com/user-attachments/assets/6f61701d-533d-42c0-aa10-faeaca10d2ba" />
- *Positive and Negative Anchor Boxes*: <img width="945" height="442" alt="image" src="https://github.com/user-attachments/assets/5d44445c-641e-4d81-bfe0-46d7596f2957" />
- *Sample Predictions*: <img width="1090" height="515" alt="image" src="https://github.com/user-attachments/assets/83ad8dc3-f05c-44a6-9a2d-cff1de13977d" />

---

## 📁 Folder Structure

```bash
VisionGraph-FPN/
│
├── models/                  # Model architecture definitions
│   ├── modelFRCNN.py        # Faster R-CNN implementation
│   ├── pyramid_vig.py       # Pyramid-style ViG module
│   ├── pyramid_vig_custom.py # Custom pyramid ViG with skip connections
│   ├── vig.py               # Isotropic Vision GNN (ViG) backbone
│   ├── vig_custom.py        # Custom ViG variants
│   ├── vig_with_skip.py     # ViG with UNet-style skip connections
│
├── utils/                   # Helper utilities
│   ├── utils.py             # General utility functions
│   ├── visualizer_utils.py  # Visualization tools
│
├── train.py                 # Training script
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── LICENSE                  # MIT License
```

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/VisionGraph-FPN.git
   cd VisionGraph-FPN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

*Note*: Ensure you have Python 3.8+ and a compatible GPU environment (e.g., CUDA) for optimal performance.

---

## 🏃‍♂️ Training

Run the training script with:
```bash
python train.py
```

- Modify hyperparameters (e.g., learning rate, batch size) and backbone variants in `train.py`.
- Training logs and checkpoints are saved in `outputs/` (created automatically).
- Refer to `notebooks/training_loss/` for epoch-wise loss analysis.

---

## 🔭 Future Scope

- **Model Optimization**: Reduce computational complexity for real-time inference in clinical settings.
- **Domain Generalization**: Extend VisionGraph-FPN to other imaging modalities (e.g., CT, MRI, histopathology).
- **Clinical Integration**: Validate performance in real-world clinical environments.
- **Expanded Applications**: Adapt the framework for other radiological detection tasks.

---

## 📄 License

MIT License © Dhanush Mohan

---

## 📚 Citation

If you use VisionGraph-FPN in your research, please cite:
```
@misc{visiongraphfpn2025,
  author = {Alan Jacob Anil and Pritika Kannapiran and Dhanush Mohanasundaram and Sakthivel Velusamy and Prakash P},
  title = {VisionGraph-FPN: Graph Neural Network-Based Feature Pyramid Network for Robust Pneumonia Detection using Chest X-Ray Medical Image Dataset},
  year = {2025},
  url = {https://github.com/Dhanush-Mohan/VisionGraphFPN}
}

```
