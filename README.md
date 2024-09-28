# Anti-Money Laundering Detection with Machine Learning and GNN Approaches
Students: Giannakos Konstantinos, Vrouvaki Evrydiki
This project aims to detect money laundering activities using a combination of classical machine learning techniques and advanced Graph Neural Networks (GNNs). The dataset used for this project is the [IBM Transactions for Anti-Money Laundering dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data), which contains synthetic transaction data modeling various financial activities.

## Dataset
The dataset simulates a virtual financial ecosystem containing both legitimate and illicit transactions. The data covers various laundering techniques and provides a comprehensive set of features and labels, making it ideal for training machine learning models to detect money laundering.

### Data Structure:
- **Group HI**: Datasets with higher illicit activity.
- **Group LI**: Datasets with lower illicit activity.
  
Each group is further divided into small, medium, and large datasets.

For each dataset, there are two files:
1. A CSV file containing the transactions.
2. A text file listing laundering patterns.

## Notebooks Overview

### 1. **EDA.ipynb**
This notebook performs an exploratory data analysis (EDA) on the dataset to understand the structure and distribution of transactions, as well as laundering patterns.

- **Goal**: To gain insights into the dataset, understand key trends, and prepare the data for modeling.
- **Main Steps**:
  - Data loading and basic statistics.
  - Visualization of distributions (e.g., transaction amounts, timestamps, laundering vs. non-laundering transactions).
  - Analysis of laundering patterns and feature correlations.

### 2. **XGBoost-LightGBM.ipynb**
This notebook focuses on implementing two classical machine learning models: XGBoost and LightGBM.

- **Goal**: To test these models on the imbalanced dataset and later on a balanced dataset using SMOTE (Synthetic Minority Over-sampling Technique).
- **Main Steps**:
  - Training and evaluation of XGBoost and LightGBM on the raw, imbalanced dataset.
  - Application of SMOTE to balance the training data.
  - Retraining and evaluation of the models with the balanced dataset.
  - Comparison of performance before and after applying SMOTE.

### 3. **GNN Approaches.ipynb**
This notebook implements three Graph Neural Network (GNN) models: GAT (Graph Attention Networks), GCN (Graph Convolutional Networks), and GIN (Graph Isomorphism Networks). 

- **Goal**: To leverage the graph structure of the dataset and explore how GNNs perform in detecting laundering transactions.
- **Main Steps**:
  1. **Hyperparameter Tuning for BCELoss**: 
     - Apply tuning for BCELoss, followed by training and evaluating each model.
  2. **SMOTE Application**: 
     - Apply SMOTE to the training data and re-run the tuning, training, and evaluation.
  3. **Focal Loss Tuning**: 
     - Perform hyperparameter tuning for Focal Loss, which is more suitable for imbalanced datasets, and repeat the training-evaluating process.
  4. **SMOTE with Focal Loss**: 
     - Re-apply SMOTE and re-tune, train, and evaluate using Focal Loss.
