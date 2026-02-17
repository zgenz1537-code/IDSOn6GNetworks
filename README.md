# 🛡️ Intrusion Detection Framework on 6G Networks (ToN-IoT)

This repository implements a **Network Intrusion Detection System on 6G Networks** using:

* Data preprocessing & normalization
* Feature selection using **Improved Sparrow Search Optimizer (ISSOA)**
* Deep Learning model (**MIX_LSTM**)
* Custom **Rabbit Optimizer (ROA)** for training
* Performance evaluation with ROC, PR curves, and confusion matrix

The pipeline is designed for the **ToN-IoT Network Dataset**.

---

# 📂 Project Workflow

```
Raw Dataset
    ↓
Preprocessing & Encoding
    ↓
Feature Selection (ISSOA)
    ↓
Deep Learning Training (MIX_LSTM)
    ↓
Evaluation & Visualization
```

---

# ⚙️ Requirements

## Python Version

```
Python >= 3.11.9
```

## Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn prettytable
pip install tensorflow keras
pip install Py_FS
```

---

# 📁 Project Structure

```
Data/
 ├── TON_IoT_Train_Test_Network.csv
 ├── preprocessed.csv
 ├── features_selected.csv
 └── ss.pkl

model/
results/
utils.py
model.py
ISSOA.py
ROA.py
train.py
data_handler.py
feature_selection.py
reset_random.py
```

---

# 🚀 Execution Steps

Follow the steps **in order**.

---

## ✅ Step 1 — Dataset Preprocessing

This step:

* Loads ToN-IoT dataset
* Cleans missing values
* Encodes categorical columns
* Normalizes using StandardScaler
* Saves processed data

Run:

```bash
python data_handler.py
```

Output:

```
Data/preprocessed.csv
```

---

## ✅ Step 2 — Feature Selection using ISSOA

This stage selects optimal features using the Improved Sparrow Search Optimizer.

Run:

```bash
python feature_selection.py
```

Outputs:

```
Data/features_selected.csv
Data/selected_features.txt
Data/convergence_graph.jpg
```

---

## ✅ Step 3 — Model Training

Model Architecture:

* MIX LSTM
* Dense Softmax Classifier
* Rabbit Optimizer (ROA)

Run:

```bash
python train.py
```

Training performs:

* Standard scaling
* Train/Test split
* Model checkpointing
* Metrics plotting

Outputs:

```
model/model.h5
model/accuracy.png
model/loss.png

results/train/
results/test/
```

---

# 📊 Evaluation Outputs

Generated automatically:

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Class-wise Metrics CSV

Location:

```
results/train/
results/test/
```

---

# 🔁 Reproducibility

Random seed control is handled by:

```
reset_random.py
```

Ensures deterministic runs.

---

# 🧠 Model Summary

```
Input → MIX LSTM → Dense Softmax
```

Optimizer:

```
Rabbit Optimizer (ROA)
```

Feature Selection:

```
Improved Sparrow Search Optimizer (ISSOA)
```

---

# ⚠️ Notes

* Dataset file must be placed inside `Data/`.
* Training uses full dataset for validation as implemented.
* Large batch size (4096) requires sufficient GPU/CPU memory.
* TensorFlow eager execution is disabled for reproducibility.

---

# 📜 License

This project is intended for academic and research use.

---
