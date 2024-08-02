# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV,KFold
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from IPython.display import display, HTML
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from itertools import cycle



# %%
df = pd.read_csv('HeartAttack.csv')
# Removing column 'group' as it is not required
df = df.drop(columns=['group'])
# Removing rows with any NaN values
df_cleaned = df.dropna()
print(df_cleaned.head(10))

# %%
df_cleaned.info()

# %% [markdown]
# # creating the correlation matrix and implementing the PCA in order to reduce the features count

# %%

matrix = df_cleaned.corr() * 100

plt.figure(figsize=(25,15))
sns.heatmap(matrix, annot=True, linewidth=.5, vmin=0, vmax=100,
            fmt=".1f", cmap=sns.color_palette("flare", as_cmap=True))
plt.show()

# %%
copy_columns = df_cleaned.columns[1:]
X = df_cleaned[copy_columns].copy()
y = df_cleaned['sub-type']


# %%
# Scale the data before PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)
scaled_data_df = pd.DataFrame(scaled_data, columns=X.columns)
scaled_data_df

# %%
# n_components can also be a percentage indicating the percentage of explained variance you want to capture
pca1 = PCA(n_components=0.85,random_state=42)
pca_data = pca1.fit_transform(scaled_data)
pca_data

# %%
pca1.explained_variance_ratio_

# %%
# Multiply explained by 100 and round
per_var = np.round(pca1.explained_variance_ratio_ * 100, decimals=1)
# Create labels for barplot
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]


plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)

plt.ylabel('Percentage')

plt.xlabel('Principal comp')

plt.title('Scree Plot')

plt.show()

# %%
sum(pca1.explained_variance_ratio_)

# %%
from pca import pca
#scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns)
###########################################################
# COMPUTE AND VISUALIZE PCA
###########################################################
# Initialize the PCA
model = pca(n_components=0.85)

# Fit and transform
results = model.fit_transform(X=scaled_data_df)

# Plot the explained variance
fig, ax = model.plot()

# Scatter the first two PCs
fig, ax = model.scatter()

# Create a biplot
fig, ax = model.biplot()

# %%
pca_df = pd.DataFrame(pca_data, columns=labels)
pca_df

# %% [markdown]
# # Simple Multinomial Logistic Regression model using PCA data

# %%
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Splitting the data into features and target
X_PCA = pca_df.copy()


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_PCA, y, test_size=0.2, random_state=42)

# Creating the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Fitting the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy*100)
print("Classification Report:\n", report)

# plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix , annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Multinomial Logistic Regression')
#plt.suptitle()
plt.show()


# %% [markdown]
# # Using the GridSearch to find best combination of hyperparameter of the model

# %%
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define the parameter grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'sag'],
    'max_iter': [100, 200, 300]
}

# Initialize the Logistic Regression model
logreg = LogisticRegression(multi_class='multinomial', random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Get the best model
best_logreg_model = grid_search.best_estimator_

# Make predictions
y_pred = best_logreg_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix_best = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy*100)
print("Classification Report:\n", report)
#print("Confusion Matrix:\n", conf_matrix)

# plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_best , annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Best Performing Multinomial Logistic Regression')
#plt.suptitle()
plt.show()


# %% [markdown]
# ### There is no change in the results when we use the GridSearchCV techinique to find the best hyperparameters for Multinomial Logistic Regression model.

# %% [markdown]
# # Implementing the Multinomial Logistic regression,SVC,Decision Tree,Random Forest and XGBOOST with optimization techinique cross validation. Stratified K fold cross validation strategy is used to assess the performance of the machine learning models and ensure its generalizability to unseen data.

# %%

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

X_pca = pca_data.copy()
# Define the stratified k-fold cross-validation strategy
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Initialize the models with hyperparameters
logistic_model = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)  # L2 regularization with C parameter
svc_model = SVC(kernel='linear', probability=True, C=1.0)  # Regularization with C parameter
tree_model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=10, min_samples_leaf=5)  # Regularization with tree para-meters
forest_model = RandomForestClassifier(random_state=42, max_depth=10, min_samples_split=10, min_samples_leaf=5)  # Regularization with tree parameters
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, reg_alpha=0.1, reg_lambda=1.0)  # L1 and L2 regularization

# Lists to store evaluation metrics for each fold
logistic_accuracies = []
svc_accuracies = []
tree_accuracies = []
forest_accuracies = []
xgb_accuracies = []

logistic_aucs = []
svc_aucs = []
tree_aucs = []
forest_aucs = []
xgb_aucs = []

# Perform k-fold cross-validation
for train_index, test_index in skf.split(X_pca, y):
    X_train, X_test = X_pca[train_index], X_pca[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Logistic Regression
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)
    y_prob_logistic = logistic_model.predict_proba(X_test)
    logistic_accuracies.append(accuracy_score(y_test, y_pred_logistic))
    logistic_aucs.append(roc_auc_score(y_test, y_prob_logistic, multi_class='ovr'))
    
    # SVC
    svc_model.fit(X_train, y_train)
    y_pred_svc = svc_model.predict(X_test)
    y_prob_svc = svc_model.predict_proba(X_test)
    svc_accuracies.append(accuracy_score(y_test, y_pred_svc))
    svc_aucs.append(roc_auc_score(y_test, y_prob_svc, multi_class='ovr'))
    
    # Decision Tree
    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)
    y_prob_tree = tree_model.predict_proba(X_test)
    tree_accuracies.append(accuracy_score(y_test, y_pred_tree))
    tree_aucs.append(roc_auc_score(y_test, y_prob_tree, multi_class='ovr'))
    
    # Random Forest
    forest_model.fit(X_train, y_train)
    y_pred_forest = forest_model.predict(X_test)
    y_prob_forest = forest_model.predict_proba(X_test)
    forest_accuracies.append(accuracy_score(y_test, y_pred_forest))
    forest_aucs.append(roc_auc_score(y_test, y_prob_forest, multi_class='ovr'))
    
    # XGBoost
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)
    xgb_accuracies.append(accuracy_score(y_test, y_pred_xgb))
    xgb_aucs.append(roc_auc_score(y_test, y_prob_xgb, multi_class='ovr'))

# Average accuracy and AUC across all folds
avg_logistic_accuracy = np.mean(logistic_accuracies)
avg_svc_accuracy = np.mean(svc_accuracies)
avg_tree_accuracy = np.mean(tree_accuracies)
avg_forest_accuracy = np.mean(forest_accuracies)
avg_xgb_accuracy = np.mean(xgb_accuracies)

avg_logistic_auc = np.mean(logistic_aucs)
avg_svc_auc = np.mean(svc_aucs)
avg_tree_auc = np.mean(tree_aucs)
avg_forest_auc = np.mean(forest_aucs)
avg_xgb_auc = np.mean(xgb_aucs)

display(HTML(f"<h2>Average Accuracies of Models on Test Data </h2>"))

print(f"Average Logistic Regression Accuracy: {avg_logistic_accuracy}")
print(f"Average SVC Accuracy: {avg_svc_accuracy}")
print(f"Average Decision Tree Accuracy: {avg_tree_accuracy}")
print(f"Average Random Forest Accuracy: {avg_forest_accuracy}")
print(f"Average XGBoost Accuracy: {avg_xgb_accuracy}")

display(HTML(f"<h2>Average AUCs of Models on Test Data </h2>"))
print(f"Average Logistic Regression AUC: {avg_logistic_auc}")
print(f"Average SVC AUC: {avg_svc_auc}")
print(f"Average Decision Tree AUC: {avg_tree_auc}")
print(f"Average Random Forest AUC: {avg_forest_auc}")
print(f"Average XGBoost AUC: {avg_xgb_auc}")

# Fit the models on the entire dataset and get final evaluation metrics
#display(HTML(f"<h2>Fit the models on the entire dataset and display final evaluation metrics</h2>"))
#logistic_model.fit(X_pca, y)
#svc_model.fit(X_pca, y)
#tree_model.fit(X_pca, y)
#forest_model.fit(X_pca, y)
#xgb_model.fit(X_pca, y)

y_pred_logistic = logistic_model.predict(X_pca)
y_prob_logistic = logistic_model.predict_proba(X_pca)
y_pred_svc = svc_model.predict(X_pca)
y_prob_svc = svc_model.predict_proba(X_pca)
y_pred_tree = tree_model.predict(X_pca)
y_prob_tree = tree_model.predict_proba(X_pca)
y_pred_forest = forest_model.predict(X_pca)
y_prob_forest = forest_model.predict_proba(X_pca)
y_pred_xgb = xgb_model.predict(X_pca)
y_prob_xgb = xgb_model.predict_proba(X_pca)

# Final evaluation
display(HTML(f"<h3><i>Average Accuracies ,Confusion matrix,Hyperparameters of Models on Entire Data </i></h3>"))

display(HTML(f"<h2>Average Accuracies of Models on Entire Data </h2>"))
print("Logistic Regression Accuracy Score:", accuracy_score(y, y_pred_logistic))
print("SVC Accuracy Score:", accuracy_score(y, y_pred_svc))
print("Decision Tree Classifier Accuracy Score:", accuracy_score(y, y_pred_tree))
print("Random Forest Classifier Accuracy Score:", accuracy_score(y, y_pred_forest))
print("XGBoost Accuracy Score:", accuracy_score(y, y_pred_xgb))

display(HTML(f"<h2>Balanced Accuracies of Models on Entire Data </h2>"))
print("Logistic Regression Accuracy Score:", balanced_accuracy_score(y, y_pred_logistic))
print("SVC Accuracy Score:", balanced_accuracy_score(y, y_pred_svc))
print("Decision Tree Classifier Accuracy Score:", balanced_accuracy_score(y, y_pred_tree))
print("Random Forest Classifier Accuracy Score:",balanced_accuracy_score(y, y_pred_forest))
print("XGBoost Accuracy Score:", balanced_accuracy_score(y, y_pred_xgb))

print("\nLogistic Regression Classification Report:\n", classification_report(y, y_pred_logistic))
print("Logistic Regression Hyperparameters:", logistic_model.get_params())

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y, y_pred_logistic), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Multinomial Logistic Regression')
plt.show()

print("SVC Classification Report:\n", classification_report(y, y_pred_svc))
print("SVC Hyperparameters:", svc_model.get_params())

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y, y_pred_svc), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Support Vector Classifier')
plt.show()

print("Decision Tree Classification Report:\n", classification_report(y, y_pred_tree))
print("Decision Tree Hyperparameters:", tree_model.get_params())

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y, y_pred_tree), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.show()

print("Random Forest Classification Report:\n", classification_report(y, y_pred_forest))
print("Random Forest Hyperparameters:", forest_model.get_params())

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y, y_pred_forest), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest Classifier')
plt.show()

print("XGBoost Classification Report:\n", classification_report(y, y_pred_xgb))
print("XGBoost Hyperparameters:", xgb_model.get_params())

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y, y_pred_xgb), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Extreme Gradient Booster Classifier')
plt.show()

display(HTML(f"<h3><i>Average AUCs of Models on Entire Data </i></h3>"))

# Plot ROC-AUC curves for all models
def plot_roc_curve(models, X, y):
    plt.figure(figsize=(14, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])

    for model, color in zip(models, colors):
        y_prob = model.predict_proba(X)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y.shape[1])]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(y.shape[1]):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= y.shape[1]
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.plot(fpr["macro"], tpr["macro"], color=color,
                 label=f'{model.__class__.__name__} (area = {roc_auc["macro"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - AUC')
    plt.legend(loc="lower right")
    plt.show()

# One-hot encode the labels
y_one_hot = pd.get_dummies(y).values

# Plot ROC curves for all models
plot_roc_curve([logistic_model, svc_model, tree_model, forest_model, xgb_model], X_pca, y_one_hot)


# %%
import numpy as np

# Assuming X_pca is your PCA-transformed feature set and xgb_model is your trained XGBoost model
# Randomly select 4 samples from X_pca
random_indices = np.random.choice(X_pca.shape[0], 4, replace=False)
random_samples = X_pca[random_indices]

# Make predictions using the xgb_model
predictions = xgb_model.predict(random_samples)

# Capture and display the details
print("Randomly selected samples from X_pca and their predictions:")
for i, index in enumerate(random_indices):
    print(f"\nSample {i + 1} (Index {index}):")
    print("PCA Features:", random_samples[i])
    print("Prediction:", predictions[i])
    print("Prediction from y_pred_xgb", y_pred_xgb[index])
    



# %%
print({y_pred_xgb[365]},{y_pred_xgb[345]},{y_pred_xgb[582]},{y_pred_xgb[176]})


# %%
import numpy as np

# Assuming X_pca is your PCA-transformed feature set
# Calculate the mean and covariance of X_pca
mean_X_pca = np.mean(X_pca, axis=0)
cov_X_pca = np.cov(X_pca, rowvar=False)

# Generate synthetic data similar to X_pca
num_samples = 4  # Number of synthetic samples to generate
synthetic_data = np.random.multivariate_normal(mean_X_pca, cov_X_pca, num_samples)

# Make predictions using the xgb_model on the synthetic data
predictions_synthetic = xgb_model.predict(synthetic_data)

# Capture and display the details
print("Synthetic samples similar to X_pca and their predictions:")
for i, sample in enumerate(synthetic_data):
    print(f"\nSynthetic Sample {i + 1}:")
    print("PCA Features:", sample)
    print("Prediction:", predictions_synthetic[i])



