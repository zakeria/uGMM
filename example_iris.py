import torch
from test_architectures import *
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np

import random

torch.set_printoptions(sci_mode=False)

def predictNLLMPE(spn, data, label, device, epoch):
    mpe = spn.infer_mpe(data, mpe_vars=[4], mpe_states=[0.,1.,2.])
    predictions = mpe.argmax(dim=0).squeeze()
    accuracy = (predictions == label).sum() / len(label)
    print(f'epoch: {epoch}%, MPE Accuracy: {accuracy}%')

def Classify_iris_nll():
      device = "cpu"
      # device = "cuda"
      random_seed = 0
      torch.manual_seed(random_seed)
      features, label = load_iris(return_X_y=True)
      scaler = StandardScaler()
      features = scaler.fit_transform(features)
      features = torch.tensor(features, dtype=torch.float32).to(device)
      label = torch.tensor(label, dtype=torch.int).to(device)
      data = torch.cat([features, label.unsqueeze(1).int()], dim=1).to(device)

      spn = iris_nll_gmm(device)
      optimizer = torch.optim.Adam(spn.parameters(), lr=0.001)

      test = True
      for i in range(1000000):
            optimizer.zero_grad()
            output = spn.infer(data, training=True)
            loss = -1 * output.mean()
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                  print("log-likelihood: {output.sum().item():10.4f}")

                  if test:
                        predictNLLMPE(spn, data, label, device, i)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def Classify_iris_mlp():
      iris = load_iris()
      X = iris.data
      y = iris.target

      scaler = StandardScaler()
      X = scaler.fit_transform(X)

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
      y_train_tensor = torch.tensor(y_train, dtype=torch.long)
      X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
      y_test_tensor = torch.tensor(y_test, dtype=torch.long)

      train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
      train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

      input_size = X_train.shape[1]
      hidden_size = 32  
      output_size = len(torch.unique(y_train_tensor))

      model = MLP(input_size, hidden_size, output_size)
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

      for epoch in range(100):
            for batch_X, batch_y in train_loader:
                  optimizer.zero_grad()
                  outputs = model(batch_X)
                  loss = criterion(outputs, batch_y)
                  loss.backward()
                  optimizer.step()

            with torch.no_grad():
                  model.eval()
                  outputs = model(X_test_tensor)
                  _, predicted = torch.max(outputs, 1)
                  accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
                  print(f'Test Accuracy: {accuracy * 100:.2f}%')
                  print(f'Epoch {epoch + 1}/{100}, Loss: {loss.item()}')


Classify_iris_nll()
# Classify_iris_mlp()