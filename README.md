# Sentiment-Prediction-on-Movie-Reviews

This project involves predicting the sentiment of movie reviews (positive or negative) using machine learning techniques. The dataset consists of movie reviews along with their respective sentiment labels (1 for positive and 0 for negative). The goal is to predict the sentiment of a movie review based on its content, along with additional features such as audience score and runtime.

# Project Overview

In this project:

### Preprocess the data:
Perform data scaling on numerical features (audienceScore, runtimeMinutes) using Min-Max scaling.
Clean and preprocess the text data (reviewText) using TF-IDF Vectorization.

### Split the data:
The dataset is split into training and testing sets, ensuring that the sentiment distribution is maintained in both sets (stratified sampling).

### Feature Extraction:
The text data is transformed into numerical features using TF-IDF vectorization.
Numeric columns like audienceScore and runtimeMinutes are scaled using the Min-Max scaler.

### Model Building:
A Logistic Regression model is used to predict the sentiment of the movie reviews.
The model is evaluated using metrics such as accuracy, precision, recall, f1-score, and confusion matrix.

### Model Evaluation:
A detailed classification report is provided along with evaluation metrics.
Precision-Recall curves and ROC curves are plotted to visually assess the model performance.

### Performance:
The model achieves an accuracy of approximately 84% on the training set and 80% on the test set.
F1-micro score for training and test data is also provided to evaluate model performance in class imbalance scenarios.
