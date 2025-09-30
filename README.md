# **Titanic Survival Prediction: A Machine Learning Project Report**

### **Project by: Dheeraj Verma**

## **1. Project Overview**

This project demonstrates an end-to-end machine learning workflow to predict passenger survival on the RMS Titanic. Using the public Titanic dataset from Kaggle, the process involved comprehensive data cleaning, exploratory data analysis (EDA) to uncover key insights, and the development of multiple classification models. The final model, a tuned **Random Forest Classifier**, achieved an **accuracy of 83.2%** on the unseen test set. To showcase practical application, the final model was deployed as an interactive web application using Streamlit.

## **2. Data Preprocessing & Cleaning**

A clean and well-structured dataset is crucial for effective modeling. The following preprocessing steps were performed:

* **Handling Missing Values:**
    * Missing `Age` values were imputed using the dataset's **median age (28)** to avoid distortion from outliers.
    * The two missing `Embarked` values were filled with the **mode ('S')**, representing the most common port of embarkation.
    * The `Cabin` column was dropped due to having a high percentage of missing data.

* **Feature Engineering & Encoding:**
    * Categorical features like `Sex` and `Embarked` were converted into a numerical format using **one-hot encoding** to be compatible with machine learning models.

* **Normalization:**
    * Numerical features with different scales (`Age` and `Fare`) were standardized using **`StandardScaler`**. This ensures that no single feature disproportionately influences the model's performance.

## **3. Exploratory Data Analysis (EDA) - Key Findings**

Visual analysis of the dataset revealed several strong predictors of survival:

* **Finding 1: Gender was a Primary Factor**
    * A count plot visualization clearly showed that female passengers had a significantly higher chance of survival (~74%) compared to male passengers (~19%). This was the single most influential feature.

* **Finding 2: Socio-Economic Status Mattered**
    * A correlation heatmap and class-based analysis confirmed that passengers in **1st Class had a much higher survival rate** than those in 2nd or 3rd Class. Furthermore, the `Fare` paid for the ticket showed a positive correlation with survival.

* **Finding 3: Age Distribution**
    * A histogram of passenger ages showed a distribution centered around the late 20s. While not as strong a predictor as gender or class, age played a role in the model's predictions.

## **4. Model Development & Evaluation**

Two distinct classification models were trained to establish a performance baseline and identify the most effective approach. The dataset was split into 80% for training and 20% for testing.

| Model | Accuracy | Precision (Survived) | Recall (Survived) | F1-Score (Survived) |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 81.0% | 0.78 | 0.78 | 0.78 |
| **Random Forest** | **83.2%** | **0.79** | **0.81** | **0.80** |

The **Random Forest Classifier** was selected as the superior model due to its higher overall accuracy and better balance of precision and recall.

## **5. Model Optimization with GridSearchCV**

To enhance the performance of the Random Forest model, its hyperparameters were fine-tuned using `GridSearchCV`.

* The search optimized for parameters like `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
* While the overall accuracy remained stable at **83.2%**, the tuned model showed a notable improvement in **precision (from 0.79 to 0.82)**. This demonstrates an understanding of the precision-recall tradeoff, allowing the model to be tailored for specific objectives where minimizing false positives is critical.

## **6. Deployment (Bonus)**

To complete the project lifecycle, the final tuned model was saved using `joblib` and deployed as a simple, user-friendly web application with **Streamlit**. This interactive app allows a user to input hypothetical passenger details (Class, Sex, Age, Fare, etc.) and receive a real-time survival prediction, demonstrating the ability to move a machine learning model from research to a practical, usable product.

## **7. Conclusion**

This project successfully built and deployed a machine learning model capable of predicting Titanic survival with 83.2% accuracy. It showcased a comprehensive skill set, including data cleaning, insightful EDA, comparative model building, hyperparameter tuning, and web deployment.

**Potential next steps** could include exploring more advanced feature engineering (e.g., extracting titles from names) or implementing gradient boosting models (like XGBoost or LightGBM) to potentially achieve even higher accuracy.
