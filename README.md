# Credit Card Fraud Detection App

### Overview

This project is a machine learning-powered web application for detecting fraudulent credit card transactions. The app uses a Logistic Regression model to classify transactions as either legitimate or fraudulent. The web interface is built with Streamlit, a Python library for creating data apps.

The model was trained on a highly imbalanced dataset where features were anonymized to protect user privacy.

### Machine Learning Workflow

The following key machine learning functions were performed to build the fraud detection model:

* **Data Preprocessing**: The dataset was loaded and checked for missing values, of which none were found. The `Time` and `Amount` features were then scaled using `StandardScaler` to ensure they have a consistent scale for the model.

* **Handling Class Imbalance**: The original dataset was highly imbalanced, with only a small number of fraudulent transactions. To address this, an **Under-Sampling** technique was used to create a new, balanced dataset. This new dataset contains an equal number of legitimate and fraudulent transactions, ensuring the model does not become biased towards the majority class.

* **Model Training**: A **Logistic Regression** model was chosen for its interpretability and efficiency. The model was trained on the balanced dataset, learning to identify the patterns that differentiate fraudulent from legitimate transactions.

* **Model Evaluation**: The model's performance was evaluated on a test set to ensure its reliability. The following metrics were used:
    * **Accuracy:** 96.45%
    * **Precision:** 97.89%
    * **Recall:** 94.89%
    * **F1-Score:** 96.37%

### Project Files

* `app.py`: The main Streamlit application code.
* `trained_model.sav`: The saved Logistic Regression model.
* `scaler.pkl`: The saved `StandardScaler` object.
* `Fraud Detection in Credit Card Transactions.ipynb`: The Jupyter notebook detailing the full project workflow from data exploration to model saving.
* `README.md`: This file.
* `requirements.txt`: A list of all Python libraries needed for the project.

### Contact

For any questions, feel free to contact me at [Your Email Address].
