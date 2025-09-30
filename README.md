# Titanic Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Overview

This project is an end-to-end data science implementation focused on predicting passenger survival on the RMS Titanic. The goal was to build a machine learning model that accurately predicts whether a passenger survived based on features like age, gender, and passenger class.

The project demonstrates a complete workflow, including:
* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Comparative model building
* Hyperparameter tuning
* Deployment of the final model as an interactive web application.

The final **Tuned Random Forest model achieved 83.2% accuracy** on unseen data.

---

## 2. Project Workflow

The project followed a structured data science pipeline:

1.  **Data Loading & Initial Analysis:** Loaded the dataset and performed an initial assessment of its structure, data types, and missing values.
2.  **Preprocessing & Feature Engineering:** Cleaned the data by imputing missing values (Age, Embarked), encoding categorical variables (Sex, Embarked) into a numerical format, and standardizing numerical features (Age, Fare).
3.  **Exploratory Data Analysis (EDA):** Created visualizations (histograms, count plots, heatmaps) to identify key patterns and correlations, such as the significant impact of gender and passenger class on survival.
4.  **Model Building & Evaluation:** Trained two different baseline models (Logistic Regression and Random Forest) to compare their performance using metrics like accuracy, precision, recall, and F1-score.
5.  **Hyperparameter Tuning:** Optimized the best-performing model (Random Forest) using `GridSearchCV` to find the optimal combination of parameters and improve its predictive power.
6.  **Deployment:** Saved the final trained model and the feature scaler using `joblib` and built an interactive web application with **Streamlit** to serve predictions to users.

---

## 3. How to Run This Project

To run this project on your local machine, please follow the steps below.

### **Prerequisites**
* Python 3.9 or higher
* pip package manager

### **Step 1: Download requirement.txt File**


### **Step 2: Clone the Repository & Install Dependencies**

```bash
# Clone this repository
git clone https://github.com/dheerajjvermaa/Titanic-survival-prediction.git
cd Titanic-survival-prediction

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirement.txt
````

### **Step 3: Run the Jupyter Notebook**

The `Titanic_survival.ipynb` file contains all the data analysis, model training, and evaluation steps.

```bash
# Launch Jupyter Notebook
jupyter notebook
```

### **Step 4: Launch the Streamlit Web App**

The `app.py` file deploys the trained model as an interactive web app.

```bash
# Run the Streamlit app from the terminal
streamlit run app.py
```

After running the command, a new tab will open in your web browser with the application.

-----

## 4\. Model Performance

The Random Forest model was the top performer. After hyperparameter tuning, the final model achieved the following results on the test set:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **83.2%** |
| Precision (for "Survived") | 0.82 |
| Recall (for "Survived") | 0.76 |
| F1-Score (for "Survived")| 0.79 |

The tuning process successfully increased the model's precision, making it more reliable when predicting a positive "survived" outcome.

## 5\. File Structure

```
.
├── Titanic_survival.ipynb    # Jupyter Notebook with all analysis and modeling
├── app.py                    # Streamlit app script for deployment
├── titanic_model.pkl         # Saved/trained Random Forest model
├── scaler.pkl                # Saved feature scaler
├── requirements.txt          # List of Python dependencies
└── README.md                 # Project documentation
```

```
```
