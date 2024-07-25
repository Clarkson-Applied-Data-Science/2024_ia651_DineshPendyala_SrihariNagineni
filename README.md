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

![SCREE Plot](/Plots/SCREE.png)
![Varianceplot](/Plots/Varianceplot.png)
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
<div style="display: flex; flex-wrap: wrap;">

<div style="flex: 33%; padding: 10px; box-sizing: border-box;">
<img src="/Plots/histogram_NEU.png" alt="Box Plot Age" style="width:100%">
<p>histogram_NEU</p>
</div>

 <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
        <img src="/Plots/histogram_MO.png" alt="Box Plot WBC" style="width:100%">
        <p>histogram_MO</p>
    </div>

   <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
<img src="/Plots/histogram_LY.png" alt="Box Plot Age" style="width:100%">
<p>histogram_LY</p>
</div>

   <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
<img src="/Plots/histogram_EO.png" alt="Box Plot Age" style="width:100%">
<p>histogram_EO</p>
</div>
        <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
<img src="/Plots/histogram_BA.png" alt="Box Plot Age" style="width:100%">
<p>histogram_BA</p>
</div>
    <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
<img src="/Plots/histogram_HGB.png" alt="Box Plot Age" style="width:100%">
<p>histogram_HGB</p>
</div>
    <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
<img src="/Plots/histogram_RBC.png" alt="Box Plot Age" style="width:100%">
<p>histogram_RBC</p>
</div>
    <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
<img src="/Plots/histogram_WBC.png" alt="Box Plot Age" style="width:100%">
<p>histogram_WBC</p>
</div>
   <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
<img src="/Plots/histogram_Gender.png" alt="Box Plot Age" style="width:100%">
<p>histogram_Gender</p>
</div>
   <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
<img src="/Plots/histogram_Age.png" alt="Box Plot Age" style="width:100%">
<p>histogram_Age</p>
</div>
</div>



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
        <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
        <img src="/Plots/box_plot_LY.png" alt="Box Plot HGB" style="width:100%">
        <p>Box Plot of LY</p>
    </div>
    <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
        <img src="/Plots/box_plot_MO.png" alt="Box Plot HGB" style="width:100%">
        <p>Box Plot of MO</p>
    </div>
    <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
        <img src="/Plots/box_plot_NEU.png" alt="Box Plot HGB" style="width:100%">
        <p>Box Plot of NEU</p>
    </div>
    <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
        <img src="/Plots/box_plot_BA.png" alt="Box Plot HGB" style="width:100%">
        <p>Box Plot of BA</p>
    </div>
    <div style="flex: 33%; padding: 10px; box-sizing: border-box;">
        <img src="/Plots/box_plot_EO.png" alt="Box Plot HGB" style="width:100%">
        <p>Box Plot of EO</p>
    </div>
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
- ### Accuracies
- Average Logistic Regression Accuracy: 0.6778514451465866
- Average SVC Accuracy: 0.678861493836113
- Average Decision Tree Accuracy: 0.6268672951414069
- Average Random Forest Accuracy: 0.700269346317207
- Average XGBoost Accuracy: 0.7186418729928519

- ### Confusion Matrix
![Confusion Matrix MLR](/Plots/ConfusionMatrix_MLR.png)
![Confusion Matrix SVC](/Plots/ConfusionMatrix_SVC.png)
![Confusion Matrix DTC](/Plots/ConfusionMatrix_DTC.png)
![Confusion Matrix RFC](/Plots/ConfusionMatrix_RFC.png)
![Confusion Matrix XGB](/Plots/ConfusionMatrix_XGB.png)
- ### ROC-AUC Curves
![ROC - AUC - Curves](/Plots/ROC-AUC.png)



### ExploratoryDataAnalysis(EDA)