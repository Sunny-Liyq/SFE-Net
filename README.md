# SFE-Net:Harnessing Biological Principles of Differential Gene Expression for Improved Feature Selection in Deep Learning Networks

[![arXiv](https://img.shields.io/badge/arXiv-Paper-green)](https://arxiv.org/abs/2412.20799)

**SFE-Net** is a deep learning framework inspired by biological principles of differential gene expression, designed to improve feature selection and adaptability for robust **DeepFake detection** across various datasets.

---

## 📝 Abstract

DeepFake detection faces challenges from diverse synthesis methods like **Faceswap**, **Deepfakes**, **Face2Face**, and **NeuralTextures**, which traditional machine learning models struggle to generalize across. Inspired by **differential gene expression** in biological systems, **SFE-Net** introduces a novel mechanism to dynamically prioritize critical features while suppressing irrelevant cues. This results in enhanced generalization and performance across varied DeepFake generation techniques.

Key highlights:
- Bio-inspired **Selective Feature Activation** for dynamic response.
- Rigorous cross-dataset testing to ensure robustness.
- Superior performance in mitigating overfitting and improving generalizability.

---

## 📋 Methodology

The architecture of SFE-Net is divided into the following stages:
1. **Preprocessing**:
   - Input videos are converted into frames, masks, and landmarks for analysis.
2. **Feature Extraction**:
   - Extracts five key features:
     - **Lico**: Light consistency feature.
     - **Hifr**: High-frequency feature.
     - **Comr**: Compression-reconstruction feature.
     - **Moop**: Morphological operation feature.
     - **Text**: Texture feature.
3. **Selective Expression**:
   - Inspired by gene expression, dynamically adjusts feature prioritization.
4. **Classification**:
   - Final layer predicts the authenticity (`Real` or `Fake`) of the video.

---

## 📊 Experiment

### Datasets
The model was trained and evaluated on a variety of datasets:
- **FaceForensics++**
- **Celeb-DF-v1 & Celeb-DF-v2**
- **DFDC & DFDCP**
- **DeepFakeDetection**

### Evaluation Metrics
Performance metrics include:
- **Frame-level AUC**
- **Video-level AUC**
- **Average Precision (AP)**
- **Equal Error Rate (EER)**

### Ablation Studies
Ablation experiments demonstrate the contribution of each feature, with **SFE-Net** achieving an average AUC of **0.795** across all datasets.

---

## 📈 Performance

SFE-Net outperforms other state-of-the-art models across multiple datasets, achieving(Detailed comparison results can be found in the article):
- **CDF-v1**: 0.866 AUC
- **CDF-v2**: 0.798 AUC
- **DFD**: 0.840 AUC
- **DFDC**: 0.709 AUC
- **DFDCP**: 0.760 AUC

| Method            | CDF-v1 | CDF-v2 | DFD   | DFDC  | DFDCP | Avg  |
|--------------------|--------|--------|-------|-------|-------|------|
| **SFE-Net**       | **0.866** | 0.798  | 0.840 | 0.709 | 0.760 | **0.795** |

---

## 📦 Repository Structure
```
SFE-Net/ 
├── preprocess/ # Scripts for data preprocessing 
├── models/ # SFE-Net architecture implementation 
├── experiments/ # Scripts for training and evaluation 
├── datasets/ # Instructions for dataset setup 
├── results/ # Results and evaluation metrics 
└── README.md # Project documentation
```

---

## ⚙️ Usage

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SFE-Net.git
   cd SFE-Net
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
### Training
Train SFE-Net on your dataset:
```bash
python train.py --dataset [DATASET_PATH]
```

### Evaluation
Evaluate the trained model:
```bash
python evaluate.py --model [MODEL_PATH] --dataset [DATASET_PATH]
```




## 📚 Citation
If you use SFE-Net in your research, please cite:
```css
@article{sfe-net,
  title={SFE-Net: Harnessing Biological Principles of Differential Gene Expression for Improved Feature Selection in Deep Learning Networks},
  author={Yuqi Li, Yuanzhong Zheng, Yaoxuan Wang, Jianjun Yin, Haojun Fei},
  journal={arXiv preprint arXiv:2412.20799},
  year={2024}
}
```

