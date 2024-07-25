# 2024_ia651_DineshPendyala_SrihariNagineni
# Evaluating Hematological Predictors for Acute Myocardial Infarction (AMI) Forecasting


This project aims to predict the occurrence of heart attacks(AMI) by analysis the hematological predictors using various machine learning models. The analysis is performed using PCA-transformed data and includes the evaluation of multiple classifiers such as Logistic Regression, SVC, Decision Tree, Random Forest, and XGBoost.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#EDA)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)

## Project Overview

The goal of this project is to build and evaluate machine learning models to predict the likelihood of a heart attack types based on various Hematological parameters of patient features. The analysis involves:

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

The dataset was sourced from [\[Mendeley Data - Heart Disease\]](https://data.mendeley.com/datasets/m482gb564t/1) .Dataset used in this project contains various features related to patient demographics and hematological information which are used to predict the occurrence of a heart attack. The dataset has been preprocessed and transformed using Principal Component Analysis (PCA) to reduce dimensionality and improve model performance.

### Features

The dataset includes the following features (columns):

![image](https://github.com/user-attachments/assets/778c493a-3fbb-48d6-8dcb-bb2052fbd836)

### Target Variable

The target variable (`y`) is: **sub-type** which contains 3 classes encodes as 0 - STEMI , 1-Non-STEMI and 2 - Control

### Data Preprocessing

Before feeding the data into the machine learning models, the following preprocessing steps were performed:

1. **Handling Missing Values**: Any missing values in the dataset were handled using appropriate imputation techniques.
2. **Scaling and Normalization**: The features were scaled and normalized to ensure all values are on a similar scale.
3. **Principal Component Analysis (PCA)**: PCA was applied to reduce the dimensionality of the dataset, retaining the components that explain the majority(85%) of the variance.

### Sample Data

Below is a sample of the preprocessed and PCA-transformed data:

| Principal Component 1 | Principal Component 2 | Principal Component 3 | ... | Principal Component N |
|-----------------------|-----------------------|-----------------------|-----|-----------------------|
| 0.23                  | -0.12                 | 0.45                  | ... | 0.67                  |
| -0.34                 | 0.56                  | -0.78                 | ... | -0.12                 |
| 0.45                  | -0.23                 | 0.12                  | ... | 0.34                  |
| ...                   | ...                   | ...                   | ... | ...                   |

## Exploratory Data Analysis
### Key Visiualization
1. #### Histograms


2. #### Box plots

<div style="display: flex; flex-wrap: wrap;">

<div style="flex: 33%; padding: 10px; box-sizing: border-box;">
    <img src="/Plots/box_plot_Age.png" alt="Box Plot Age" style="width:100%">
    <p>Box Plot of Age</p>
</div>

<div style="flex: 33%; padding: 10px; box-sizing: border-box;">
    <img src="/Plots/box_plot_WBC.png" alt="Box Plot WBC" style="width:100%">
    <p>Box Plot of WBC</p>
</div>

<div style="flex: 33%; padding: 10px; box-sizing: border-box;">
    <img src="/Plots/box_plot_RBC.png" alt="Box Plot RBC" style="width:100%">
    <p>Box Plot of RBC</p>
</div>

<div style="flex: 33%; padding: 10px; box-sizing: border-box;">
    <img src="/Plots/box_plot_HGB.png" alt="Box Plot HGB" style="width:100%">
    <p>Box Plot of HGB</p>
</div>

<!-- Repeat this structure for other box plots -->

</div>




3. #### Correlation Matrixs

![Correlation Matrixs](/Plots/CorrelationMatrix.png)

### Models Used
The following machine learning models are used in this project:

1. Logistic Regression
2. Support Vector Classifier (SVC)
3. Decision Tree Classifier
4. Random Forest Classifier
5. XGBoost Classifier

## Cross-Validation

The models are evaluated using 5-fold cross-validation. The following metrics are computed:
- Accuracy
- ROC-AUC

## Results

The results include model evaluation metrics and plots of ROC curves for each model. Below is an example of the output for one of the models:

### Final Evaluation

The models are trained on the entire dataset and evaluated using:
- Classification Report
- Confusion Matrix
- ROC-AUC Curves

### SVC Classification Report:


### ExploratoryDataAnalysis(EDA)