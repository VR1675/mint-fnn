#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.L1 = nn.Linear(28*28, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.L2 = nn.Linear(256, 128)
        self.L3 = nn.Linear(128, 10)
        self.dropoutLayer = nn.Dropout(0.5)
    
    def forward(self, data):
        data = data.view(-1, 28*28)
        data = F.relu(self.L1(data))
        data = self.batch_norm1(data)
        data = self.dropoutLayer(data)
        data = F.relu(self.L2(data))
        data = self.L3(data)
        return F.log_softmax(data, dim=1)

learning_rate = 0.001
batch_size = 64
num_epochs = 10

transform_pipeline = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, transform=transform_pipeline, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform_pipeline)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

fnn_model = FNN()
loss_function = nn.CrossEntropyLoss()
optimizer_fun = optim.Adam(fnn_model.parameters(), lr=learning_rate)

train_loss_his = []
test_loss_his = []

for epoch in range(num_epochs):
    fnn_model.train()
    epoch_train_loss = 0
    for images, labels in train_loader:
        optimizer_fun.zero_grad()
        predictions = fnn_model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer_fun.step()
        epoch_train_loss += loss.item()
    train_loss_his.append(epoch_train_loss / len(train_loader))
    
    fnn_model.eval()
    epoch_test_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = fnn_model(images)
            loss = loss_function(predictions, labels)
            epoch_test_loss += loss.item()
            predicted_labels = predictions.argmax(dim=1, keepdim=True)
            correct_predictions += predicted_labels.eq(labels.view_as(predicted_labels)).sum().item()
    
    test_loss_his.append(epoch_test_loss / len(test_loader))
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_his[-1]:.4f}, Test Loss: {test_loss_his[-1]:.4f}, Accuracy: {correct_predictions / len(test_data):.4f}')

plt.figure(figsize=(10,5))
plt.plot(train_loss_his, label='Training Loss')
plt.plot(test_loss_his, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss vs Epochs')
plt.show()


# In[ ]:




