# 📌 Anomaly Detection in Twitter Mentions of Public Companies

### 🔎 Overview

This project implements an **anomaly detection pipeline** for time-series data on Twitter mentions of major publicly traded companies.
The workflow is fully reproducible with **DVC (Data Version Control)**, covering data collection, preprocessing, model training, and evaluation.

---

### 📂 Dataset

* Source: [Kaggle – Anomaly Detection in Time Series](https://www.kaggle.com/code/ahmadihossein/anomaly-detection-in-time-series/input)
* Description: Twitter mentions of large publicly traded companies (every 5 minutes).
* Companies included:

  * Apple
  * Amazon
  * Salesforce
  * CVS
  * Facebook
  * Google
  * IBM
  * Coca-Cola
  * Pfizer
  * UPS

The metric is the **number of mentions per ticker symbol per 5-minute interval**.

---

### ⚙️ Pipeline Stages

#### **Stage 1: Data Collection**

* Script: `src/data_collection.py`
* Output: `data/raw/`
* Collects raw Twitter mention data.

#### **Stage 2: Data Preparation**

* Script: `src/data_prep.py`
* Input: `data/raw/` → Output: `data/processed/`
* Cleans and preprocesses the raw data (scaling, feature engineering).

#### **Stage 3: Model Building & Training**

* Script: `src/model_building.py`
* Input: `data/processed/` → Output: `model.pkl`
* Labels anomalies using a **3-sigma statistical rule**.
* Trains a **RandomForestClassifier** to detect anomalies.
* Saves trained model (`model.pkl`).

#### **Stage 4: Model Testing**

* Script: `src/model_testing.py`
* Input: `model.pkl` → Output: `anomaly_results.json`
* Loads the trained model.
* Evaluates on test data using **Accuracy, Precision, Recall, F1-score**.
* Saves metrics in structured JSON.

---

### 📊 Tools & Libraries

* **DVC** → pipeline reproducibility and data versioning.
* **scikit-learn** → preprocessing, Random Forest, evaluation.
* **pandas, numpy** → data handling and feature engineering.
* **joblib** → saving and loading trained models.
* **JSON** → saving evaluation results.

---

### 🗂 Project Structure

```bash
├── data/
│   ├── raw/               # raw data (collected via Stage 1)
│   ├── processed/         # processed data (Stage 2)
│
├── src/
│   ├── data_collection.py # data collection logic
│   ├── data_prep.py       # preprocessing & feature engineering
│   ├── model_building.py  # anomaly labeling & model training
│   ├── model_testing.py   # model evaluation and metrics export
│
├── model.pkl              # trained RandomForestClassifier (output of Stage 3)
├── anomaly_results.json   # evaluation metrics (output of Stage 4)
├── dvc.yaml               # DVC pipeline definition
├── requirements.txt       # Python dependencies
└── README.md              # project documentation
```

---

### 🚀 How to Run

1. Clone the repo and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Initialize DVC (first time only):

   ```bash
   dvc init
   ```

3. Run the entire pipeline:

   ```bash
   dvc repro
   ```

4. Visualize the DAG of stages:

   ```bash
   dvc dag
   ```

---

### ✅ Outputs

* **`model.pkl`** → trained anomaly detection model.
* **`anomaly_results.json`** → evaluation metrics (accuracy, precision, recall, f1-score).
* **`dvc.yaml`** → reproducible ML pipeline.
