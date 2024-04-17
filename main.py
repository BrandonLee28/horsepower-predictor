import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, input, output):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input, output)

    def forward(self, x):
        return self.lin(x)


# files
data = pd.read_csv("salary.csv")
data = data.dropna()

x = data["YearsExperience"].values.reshape(-1, 1)  # Reshape to a column vector
y = data["Salary"].values.reshape(-1, 1)  # Reshape to a column vector


x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

n_samples, n_features = x.shape

model = torch.load("salarypredict.pt")
model.eval()


def predict(hours):
    global salary_generated
    global model
    x_test = torch.tensor([[hours]], dtype=torch.float32)
    salary_generated = model(x_test).item()


salary_generated = None

st.header("Brandon's Salary Predicter")
st.subheader("Predict")
hours = st.number_input("Enter Years of Experience", min_value=0)
st.button("Generate", on_click=predict(hours))
if salary_generated:
    st.write(f"Predicted Salary: {salary_generated}")
st.subheader("Data")
st.dataframe(data)
