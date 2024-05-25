import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define Prototypical Network
class ProtoNet(nn.Module):
    def _init_(self, num_classes, feature_dim):
        super(ProtoNet, self)._init_()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),  # Increased dropout rate
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),  # Increased dropout rate
        )
        self.fc = nn.Linear(64*5*5, feature_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 64*5*5)
        x = self.fc(x)
        return x


# Define few-shot learning setup
def compute_prototypes(support_set):
    return torch.mean(support_set, dim=1)

def compute_distances(query_set, prototypes):
    return torch.cdist(query_set, prototypes)

def classify_prototypes(distances):
    return torch.argmin(distances, dim=1)

# Initialize model, loss function, and optimizer
model = ProtoNet(num_classes=10, feature_dim=64).to(device)  # Changed num_classes to 10 for MNIST
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay

# Training loop
def train_prototypes(model, train_loader, criterion, optimizer, epochs=10):  # Increased epochs
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

# Testing loop
def test_prototypes(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'Accuracy on test set: {(correct / total) * 100:.2f}%')

# Train the model
train_prototypes(model, train_loader, criterion, optimizer)

# Test the model
test_prototypes(model, test_loader)
