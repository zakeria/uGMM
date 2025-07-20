from test_architectures import *

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import MNIST

def testAccuracyFF(model, test_loader, batch_size, device):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            img = inputs.to(device)
            label = labels.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    model.train()

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def mnistFF():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])    

    batch_size = 256

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda"

    model = MLP()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()  
        for batch_index, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad() 
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward() 
            optimizer.step()  

        model.eval() 
        testAccuracyFF(model, test_loader, batch_size, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
        model.train() 

def mnistCrossEntropy():    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])    
    torch.manual_seed(0)
    batch_size = 256

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda"

    model = mnist_spn_cross_entropy(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_index, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()  
            batch_size = inputs.shape[0]
            data = inputs.reshape(batch_size, 28*28)
            data = data.to(device)
            output = model.infer(data, training=True)
            loss = criterion(output, labels.to(device))         
            
            loss.backward()  
            optimizer.step()            
            
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
        testAccuracyCrossEntropy(model, test_loader, device)


def testAccuracyCrossEntropy(spn, test_loader, device):
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (test_batch_data, test_batch_labels) in enumerate(test_loader):
                batch_size = test_batch_data.shape[0]
                data = test_batch_data.reshape(batch_size, 28*28)
                data = data.to(device)
                mpe = spn.infer(data)

                predictions = mpe.argmax(dim=1)
                labels = test_batch_labels.to(device)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
    
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

# mnistFF()
mnistCrossEntropy()