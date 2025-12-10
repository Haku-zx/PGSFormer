```markdown
# PGSFormer: Patch-based Graph-Spatial Transformer for Traffic Forecasting

PGSFormer is a spatial–temporal prediction model designed for multi-step traffic flow forecasting.  
It integrates **dilated temporal convolutions**, **graph convolution networks (GCN)**, and a **patch-based Transformer encoder**, enabling effective modeling of long-range temporal dependencies and dynamic spatial correlations.

This repository contains:
- The full implementation of **PGSFormer** (model, training engine, utilities)
- Scripts for **data preprocessing**, **training**, and **testing**
- Support for **PEMS-BAY**, **METR-LA**, and other graph-structured traffic datasets.

---

## 📌 1. Key Features

✔ **Patch-based Temporal Transformer**  
- Splits temporal signals into patches to capture long-range dependencies efficiently.  
- Learnable positional encoding and mask-based pretraining framework.

✔ **Graph Convolutional Module (GCN)**  
- Supports multiple adjacency matrix types (transition, double-transition, Laplacian, etc.).  
- Optional adaptive adjacency matrix (learnable).

✔ **Dilated Temporal Convolutions**  
- Multi-scale receptive field expansion for efficient time-series modeling.

✔ **Dynamic Adaptive Graph Learning**  
- Automatically generates node correlation graphs based on dynamic characteristics.

✔ **Efficient Training Framework**  
- Dataset loaders & normalization based on StandardScaler.  
- Built-in MAE / RMSE / MAPE metrics.

---

## 📂 2. Repository Structure

```

PGSFormer/
│── model.py                   # Main model implementation (PGSFormer)  
│── train.py                   # Training script                        
│── test.py                    # Evaluation script                      
│── util.py                    # Utilities: data loading, metrics, adj  
│── generate_training_data.py  # Generate seq2seq dataset               
│── data/
│    ├── PEMS_BAY/
│    │    ├── train.npz
│    │    ├── val.npz
│    │    └── test.npz
│    └── sensor_graph/
│         └── adj_mx_bay.pkl
│── README.md

```

---

## 🛠️ 3. Installation

### **Python environment**
```

python >= 3.8
torch >= 1.10
numpy
pandas
matplotlib
seaborn
tqdm
scipy

````

---

## 🗂️ 4. Dataset Preparation

PGSFormer uses `train.npz`, `val.npz`, and `test.npz` format as in typical graph forecasting benchmarks.

You can generate data using:

```bash
python generate_training_data.py \
    --traffic_df_filename data/pems.h5 \
    --output_dir data/PEMS_BAY \
    --seq_length_x 12 \
    --seq_length_y 12
```

This script will produce:

```
data/PEMS_BAY/
    train.npz
    val.npz
    test.npz
```

Adjacency matrix is expected at:

```
data/sensor_graph/adj_mx_bay.pkl
```

---

## 🚀 5. Train the Model

Example command:

```bash
python train.py \
    --device cuda:0 \
    --data data/PEMS_BAY \
    --adjdata data/sensor_graph/adj_mx_bay.pkl \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --dropout 0.3 \
    --save ./checkpoints/
```

Training outputs include:

* Best model checkpoint
* Training & validation logs
* Final evaluation results (MAE, RMSE, MAPE)

---

## 🧪 6. Test / Inference

After training:

```bash
python test.py \
    --device cuda:0 \
    --data data/PEMS_BAY \
    --adjdata data/sensor_graph/adj_mx_bay.pkl \
    --checkpoint ./checkpoints/best_model.pth
```

Example output:

```
Horizon 1: MAE=1.58 RMSE=3.21 MAPE=2.43%
...
On average over 12 horizons:
MAE=2.01 RMSE=4.45 MAPE=3.17%
```

---

## 🧩 7. Model Overview

### **Temporal Modeling**

* Uses gated dilated convolutions (filter + gate convs)
* Skip connections accumulate multi-resolution temporal features
* Patch-based Transformer (`TransformerLayers`, `InputEmbedding`) enhances long-term dependency modeling

### **Spatial Modeling**

* GCN module supports:

  * asymmetric adjacency
  * symmetric normalization
  * graph Laplacian
  * double-transition matrix
* Optional **adaptive graph matrix** learned by:

  ```
  adp = softmax(ReLU(x W xᵀ))
  ```

  (implemented in forward() of PGSFormer)  

### **Loss & Metrics**

Provided in `util.py`:  

* MAE
* RMSE
* MAPE
* Mask-aware versions for missing values

---

## 📖 8. Citation


```bibtex
@article{PGSFormer2025,
  title={PGSFormer: Patch-based Graph-Spatial Transformer for Traffic Forecasting},
  author={ },
  journal={ },
  year={2025}
}
```
---
