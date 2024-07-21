# 2024_ia651_DineshPendyala_SrihariNagineni
# Heart Attack Prediction Project

This project aims to predict the occurrence of heart attacks using various machine learning models. The analysis is performed using PCA-transformed data and includes the evaluation of multiple classifiers such as Logistic Regression, SVC, Decision Tree, Random Forest, and XGBoost.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)

## Project Overview

The goal of this project is to build and evaluate machine learning models to predict the likelihood of a heart attack based on various patient features. The analysis involves:

1. Data preprocessing and PCA transformation.
2. Model training using Stratified K-Fold Cross-Validation.
3. Model evaluation using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
4. Visualization of results.
 ## Setup Instructions

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`

### Installation

1. Clone the repository or download the notebook file.
2. Ensure you have the required libraries installed. You can install them using:
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib
    ```

## Usage

1. Open the Jupyter Notebook `IA651_Project_HeartAttack.ipynb`.
2. Ensure the dataset is loaded correctly as `pca_df` and `y`.
3. Run the notebook cells sequentially to execute the analysis.

## Dataset

The dataset used in this project contains various features related to patient demographics and hematological information which are used to predict the occurrence of a heart attack. The dataset has been preprocessed and transformed using Principal Component Analysis (PCA) to reduce dimensionality and improve model performance.

### Features

The dataset includes the following features (columns):

![image](https://github.com/user-attachments/assets/778c493a-3fbb-48d6-8dcb-bb2052fbd836)



### Target Variable

The target variable (`y`) is: **sub-type** which contains 3 classes encodes as 0 - STEMI , 1-Non-STEMI and 2 - Control



### Data Preprocessing

Before feeding the data into the machine learning models, the following preprocessing steps were performed:

1. **Handling Missing Values**: Any missing values in the dataset were handled using appropriate imputation techniques.
2. **Scaling and Normalization**: The features were scaled and normalized to ensure all values are on a similar scale.
3. **Principal Component Analysis (PCA)**: PCA was applied to reduce the dimensionality of the dataset, retaining the components that explain the majority of the variance.

### Sample Data

Below is a sample of the preprocessed and PCA-transformed data:

| Principal Component 1 | Principal Component 2 | Principal Component 3 | ... | Principal Component N |
|-----------------------|-----------------------|-----------------------|-----|-----------------------|
| 0.23                  | -0.12                 | 0.45                  | ... | 0.67                  |
| -0.34                 | 0.56                  | -0.78                 | ... | -0.12                 |
| 0.45                  | -0.23                 | 0.12                  | ... | 0.34                  |
| ...                   | ...                   | ...                   | ... | ...                   |

### Data Source

The dataset was sourced from [source name], which provides comprehensive data on patient health metrics and heart attack occurrences. The data has been cleaned and preprocessed to ensure quality and consistency.

Feel free to explore the dataset further and tweak the models or parameters to improve performance or adapt it to different datasets.


## Results

The results include model evaluation metrics and plots of ROC curves for each model. Below is an example of the output for one of the models:

### SVC Classification Report:
