import torch
import pandas as pd
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, input, output):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input, output)

    def forward(self, x):
        return self.lin(x)


data = pd.read_csv("salary.csv")
data = data.dropna()
x = data["YearsExperience"].values.reshape(-1, 1)  # Reshape to a column vector
y = data["Salary"].values.reshape(-1, 1)  # Reshape to a column vector


x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

n_samples, n_features = x.shape

input_size, output_size = n_features, n_features


model = LinearRegression(input_size, output_size)
x_test = torch.tensor([[10.5]], dtype=torch.float32)

learning_rate = 0.01
epochs = 10000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    y_pred = model(x)
    l = loss(y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 10 == 0:
        w, b = model.parameters()  # unpack parameters
        print("epoch ", epoch + 1, ": w = ", w[0][0].item(), " loss = ", l.item())

print(f"Prediction after training: f({x_test.item()}) = {model(x_test).item():.3f}")
torch.save(model, "salarypredict.pt")
