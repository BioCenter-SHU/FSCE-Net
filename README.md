# FSCE-Net: Flow-Enhanced Structural Context Encoding Network for Incomplete Conversational Emotion Recognition

This repository contains the official implementation of **FSCE-Net**, a novel framework designed to address the critical issue of random modality missingness in Multimodal Emotion Recognition in Conversation (MERC).

## 📂 Project Structure

```
.
├── AdaptiveConfidenceFusion.py   # Confidence fusion module
├── dataloader_iemocap.py         # IEMOCAP data loader
├── dataloader_meld.py            # MELD data loader
├── environment.yml               # Environment config
├── glow.py                       # Glow model (Flow module)
├── graph.py                      # Graph construction
├── loss.py                       # Loss functions
├── model.py                      # Main architecture
├── path.py                       # Path settings
├── rcan.py                       # Reconstruction network (RCAN)
├── train.py                      # Training script
└── utils.py                      # Utility functions
```

## 🛠️ Environment Setup

Please follow the steps below to set up the environment using Conda and Pip.

### 1. Create Conda Environment

```
conda env create -f environment.yml --force
conda activate fsce_net
```

### 2. Install PyTorch Dependencies

The model requires specific versions of PyTorch and PyTorch Geometric compatible with CUDA 11.8.

```
# Install PyTorch
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Scatter and Sparse
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install PyTorch Geometric
pip install torch-geometric==2.6.1
```

## 💾 Data Preparation

This project supports **IEMOCAP** and **MELD** datasets. Please download the pre-processed features from the sources below and organize them according to the settings in `path.py`.

### 1. IEMOCAP Dataset

The pre-processed features are provided by the **SDR-GNN** open-source repository.

- **Download Link:** [Baidu Netdisk](https://pan.baidu.com/s/1mts1_R8Lq2SZ-eUQChDCfg?pwd=sdr1)

### 2. MELD Dataset

The pre-processed features are provided by the **MMGCN** open-source repository.

- **Download Link:** [MMGCN/MELD_features at master · hujingwen6666/MMGCN](https://www.google.com/search?q=https://github.com/hujingwen6666/MMGCN/tree/master/MELD_features)

### 3. Path Configuration

After downloading, please ensure the file paths in `path.py` match your local directory structure.

**Example `path.py` configuration:**

```
DATA_DIR_Win = {
    'IEMOCAPFour': '/path/to/your/dataset/IEMOCAPFour',
    'IEMOCAPSix': '/path/to/your/dataset/IEMOCAP',
}
PATH_MELD = '/path/to/your/dataset/MELD_features/MELD_features_raw1.pkl'
```

## 🚀 How to Run

Use `train.py` to start training and evaluating the model.

### Basic Command

```
python -u train.py --mask-type='constant-0.0' --dataset='MELD'
```

### Arguments

- `--dataset`: Choose the dataset to use.
  - Options: `'MELD'`, `'IEMOCAPFour'`, `'IEMOCAPSix'`
- `--mask-type`: Simulate missing modalities by masking input features.
  - Format: `'constant-<rate>'`
  - Supported rates: `0.0`, `0.1`, `0.2`, `0.3`, `0.4`, `0.5`, `0.6`, `0.7` (Note: 0.7 equates to approx 0.67 in implementation due to retaining at least one view).
  - Example: `--mask-type='constant-0.3'` simulates 30% missing data.

## 📝 Citation & Acknowledgements

If you use the preprocessed data above and require details on the preprocessing steps, please refer to the methods in SDR-GNN and MMGCN.

**Acknowledgement:** The code in this repository references the work and implementations of **GCNet**, **SDR-GNN**, and **DiCMoR**.
