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
<img width="722" alt="Screenshot 2025-01-24 at 2 08 02 PM" src="https://github.com/user-attachments/assets/c0780911-ba7c-4107-b14f-af60f7ca8add" />


### Model Building:
A Logistic Regression model is used to predict the sentiment of the movie reviews.
The model is evaluated using metrics such as accuracy, precision, recall, f1-score, and confusion matrix.

### Model Evaluation:
A detailed classification report is provided along with evaluation metrics.
Precision-Recall curves and ROC curves are plotted to visually assess the model performance.

### Performance:
The model achieves an accuracy of approximately 84% on the training set and 80% on the test set.
F1-micro score for training and test data is also provided to evaluate model performance in class imbalance scenarios.

# Requirements
To run this project, you need the following Python libraries:
- **pandas**
- **scikit-learn**
- **numpy**
- **matplotlib**
- **seaborn**
- **nltk**
- **scipy**

# Evaluation Metrics
- **Accuracy:** Proportion of correct predictions.
- **Precision:** The proportion of positive reviews correctly predicted among all predictions of positive sentiment.
- **Recall:** The proportion of actual positive reviews correctly predicted among all actual positive reviews.
- **F1-Score:** The harmonic mean of precision and recall.
- **Confusion Matrix:** A summary of prediction results, showing true positive, true negative, false positive, and false negative counts.
- **ROC-AUC:** Receiver Operating Characteristic and Area Under Curve, measuring the model's ability to distinguish between classes.

# Results
- The model achieves an accuracy of 84% on the training data and 80% on the test data.
- The model performs well on the positive sentiment class, with high recall, while the negative class has slightly lower recall, indicating the presence of class imbalance in the data.

  <img width="496" alt="Screenshot 2025-01-24 at 2 09 12 PM" src="https://github.com/user-attachments/assets/99639a40-bd9c-4729-88b6-76d0d67a7493" />


# Future Work
- Experiment with more advanced models such as Random Forest, XGBoost, or Neural Networks to potentially improve performance.
- Tune hyperparameters using GridSearchCV or RandomizedSearchCV for better results.
- Explore other feature extraction techniques, such as Word2Vec or BERT embeddings.

# Contributing
If you'd like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. Please ensure that your code adheres to the existing structure and is well-documented.
