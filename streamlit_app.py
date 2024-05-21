import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import numpy as np
import pandas as pd
from math import ceil

def main():
    st.title('Predict Student Scores')

    gender = st.selectbox('Gender', ['Select Gender','male', 'female'])
    ethnicity = st.selectbox('Race/Ethnicity', ['Select Ethnicity','group A', 'group B', 'group C', 'group D', 'group E'])
    parental_education = st.selectbox('Parental Education', ['Select Parental Education Level','some high school', 'high school', 'some college', 'associate\'s degree', 'bachelor\'s degree', 'master\'s degree'])
    lunch = st.selectbox('Lunch', ['Select Lunch Plan','standard', 'free/reduced'])
    test_preparation_course = st.selectbox('Test Preparation Course', ['Select Test Preparation Course', 'none', 'completed'])
    reading_score = st.slider('Reading Score', min_value=0, max_value=100, step=1)
    writing_score = st.slider('Writing Score', min_value=0, max_value=100, step=1)

    if (st.button('Predict')):
        if (gender=='Select Gender' or ethnicity=='Select Ethnicity' or parental_education=='Select Parental Education Level' or lunch=='Select Lunch Plan' or test_preparation_course=='Select Test Preparation Course'):
            st.error('Please select all the fields')
        else:
            data = CustomData(
                gender=gender,
                race_ethnicity=ethnicity,
                parental_level_of_education=parental_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            pred_df = data.get_data_as_dataframe()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            st.success(f'The predicted maths score is {ceil(results[0])}')



if __name__=="__main__":
    main()