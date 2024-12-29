# Gaussian Process Regressor

Purpose: Implements Gaussian Process Regression (GPR) on the diabetes dataset using a custom regression class framework.

Workflow:

Data Preparation:
Loads the diabetes dataset, reduces its dimensions to 1 feature using PCA, and splits the data into training and testing sets.

Model Training:
A Gaussian Process Regressor with an RBF kernel is trained on the reduced training data.
The model’s R2-score is computed and saved as a .pkl file.

Visualization:
The model predicts mean values and uncertainties on the data.

A scatter plot shows:
Observed data, predictions, and a 95% confidence interval.

Key Highlights:

PCA reduces data complexity for easier visualization.
Gaussian Process Regression models non-linear relationships.
The trained model is saved for reuse, and its predictions are visualized.

<img width="1264" alt="image" src="https://github.com/user-attachments/assets/05e3027a-49a4-4a42-bdf1-ad1b250ad3dc" />





