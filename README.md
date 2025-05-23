# Teachable-Machine
# Flask ML Model Trainer and Predictor

A Flask-based web application that allows users to upload CSV datasets, select features and target columns, train a machine learning model (Random Forest for classification or regression), and make predictions through a simple web interface or API.

---

Features

- Upload CSV datasets directly from the browser
- Select features and target column dynamically
- Choose task type: Classification or Regression
- Train a Random Forest model on the uploaded dataset
- View model performance metrics (accuracy for classification, mean absolute error for regression)
- Make predictions with the trained model via web form or API
- Data preprocessing including handling missing values and feature scaling
- Label encoding for classification targets

---

Technologies Used

- Python 3.x
- Flask
- Pandas
- NumPy
- scikit-learn
- Werkzeug (for secure file upload)

---

Installation and Setup

1. Clone the repository

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
