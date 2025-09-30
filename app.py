import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('titanic_survival_model.pkl', 'rb'))

st.title('Titanic Survival Predictor')
st.header('Enter Passenger Information:')

pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0.1, 80.0, 30.0)
sibsp = st.slider('SibSp', 0, 8, 0)
parch = st.slider('Parch', 0, 6, 0)
fare = st.slider('Fare', 0.0, 500.0, 50.0)
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

input_data = {
    'Pclass': pclass,
    'Age': age,
    'Fare': fare,
    'FamilySize': family_size,
    'IsAlone': is_alone,
    'Sex_male': 1 if sex == 'male' else 0,
    'Title_Col': 0,
    'Title_Countess': 0,
    'Title_Don': 0,
    'Title_Dr': 0,
    'Title_Jonkheer': 0,
    'Title_Lady': 0,
    'Title_Major': 0,
    'Title_Master': 0,
    'Title_Miss': 0,
    'Title_Mlle': 0,
    'Title_Mme': 0,
    'Title_Mr': 1 if sex == 'male' else 0,
    'Title_Mrs': 1 if sex == 'female' else 0,
    'Title_Ms': 0,
    'Title_Rev': 0,
    'Title_Sir': 0,
    'Embarked_C': 1 if embarked == 'C' else 0,
    'Embarked_Q': 1 if embarked == 'Q' else 0,
    'Embarked_S': 1 if embarked == 'S' else 0,
}

input_df = pd.DataFrame([input_data])

train_columns = ['Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Sex_male',
       'Title_Col', 'Title_Countess', 'Title_Don', 'Title_Dr', 'Title_Jonkheer',
       'Title_Lady', 'Title_Major', 'Title_Master', 'Title_Miss', 'Title_Mlle',
       'Title_Mme', 'Title_Mr', 'Title_Mrs', 'Title_Ms', 'Title_Rev',
       'Title_Sir', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

input_df = input_df[train_columns]

scaler_age_mean = 29.69911764705882
scaler_age_std = 13.002015248063374
scaler_fare_mean = 32.2042079685746
scaler_fare_std = 49.69342859445439

input_df['Age'] = (input_df['Age'] - scaler_age_mean) / scaler_age_std
input_df['Fare'] = (input_df['Fare'] - scaler_fare_mean) / scaler_fare_std

if st.button('Predict Survival'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.success(f'Survived! (Probability: {prediction_proba[0]:.2f})')
    else:
        st.error(f'Did not survive (Probability: {prediction_proba[0]:.2f})')
