import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error
import streamlit as st

tab1, tab2 = st.tabs(["Predict", "Data"])

with tab1:
    st.title("Brandon's Horsepower Predicter")
    st.subheader("Correlation graphs")

    data = pd.read_csv("mpg.csv")
    data = data.dropna()

    x = data.drop(["horsepower", "name", "origin", "model_year"], axis=1)
    y = data["horsepower"]

    linModel = LinearRegression()
    linModel.fit(x, y)

    predictedy = linModel.predict(x)

    mse = mean_squared_error(y, predictedy)
    print(f"Mean Squared Error on Test Set: {mse}")

    plt.subplot(2, 2, 1)
    plt.scatter(x["mpg"], predictedy)
    plt.xlabel("Miles per Gallon")
    plt.ylabel("Horsepower")

    plt.subplot(2, 2, 2)
    plt.scatter(x["weight"], predictedy)
    plt.xlabel("Weight")
    plt.ylabel("Horsepower")

    plt.subplot(2, 2, 3)
    plt.scatter(x["displacement"], predictedy)
    plt.xlabel("Displacement")
    plt.ylabel("Horsepower")

    plt.subplot(2, 2, 4)
    plt.scatter(x["weight"], predictedy)
    plt.xlabel("Weight")
    plt.ylabel("Horsepower")

    st.pyplot(plt.gcf())

    st.subheader("Horsepower Prediction")
    mpg = st.number_input("Miles per Gallon")
    cylinder = st.number_input("Cylinders")
    displacement = st.number_input("Displacement")
    weight = st.number_input("Weight")
    acceleration = st.number_input("Acceleration")

    predicted_horsepower = st.session_state.get("predicted_horsepower", 0)
    prediction_result = st.empty()

    def predict():
        global predicted_horsepower
        st.session_state.clicked = True
        new_data_point = {
            "mpg": mpg,
            "cylinders": cylinder,
            "displacement": displacement,
            "weight": weight,
            "acceleration": acceleration,
        }
        query_data = pd.DataFrame([new_data_point])
        predicted_horsepower = linModel.predict(query_data)
        st.session_state["predicted_horsepower"] = predicted_horsepower
        prediction_result.text(f"Predicted value: {predicted_horsepower[0]}")

    st.button("Predict", on_click=predict)

    if st.session_state.get("clicked"):
        prediction_result.text(f"Predicted value: {predicted_horsepower[0]}")

with tab2:
    st.subheader("Horespower Data")
    st.dataframe(data)
