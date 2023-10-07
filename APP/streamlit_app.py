import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sklearn

model = joblib.load('APP/iris_log_model.joblib')

st.title('APP/Iris Flower Classifier')
image = st.image('flowers.png', use_column_width=True)

sepal_length = st.slider('Sepal Length (cm)', 0.0, 10.0)
sepal_width = st.slider('Sepal Width (cm)', 0.0, 10.0)
petal_length = st.slider('Petal Length (cm)', 0.0, 10.0)
petal_width = st.slider('Petal Width (cm)', 0.0, 10.0)

if st.button('Predict the Species'):
    input_data = pd.DataFrame({
    'SepalLengthCm': [sepal_length],
    'SepalWidthCm': [sepal_width],
    'PetalLengthCm': [petal_length],
    'PetalWidthCm': [petal_width]
})


    prediction = model.predict(input_data)

    st.write(f'Predicted Iris Species: {prediction[0]}')

st.markdown("""
    <div style="text-align:center;">
        Made with ❤️ by <a href="https://www.linkedin.com/in/sai-manoj-yarlagadda-a6159b225/">Sai Manoj Yarlagadda</a>
    </div>
    """, unsafe_allow_html=True)