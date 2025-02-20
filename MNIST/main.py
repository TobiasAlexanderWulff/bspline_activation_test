import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import mplcursors
import time
import numpy as np


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def __str__(self):
        return (
            f"SimpleCNN({self.fc1.in_features}, {self.fc1.out_features}, {self.fc2.out_features}) "
            f"with {self.conv1.kernel_size[0]}x{self.conv1.kernel_size[1]} kernel size "
            f"and {self.pool.kernel_size} pooling kernel size"
        )
  
  
# Listen zum Speichern der Metriken
train_losses = []
test_losses = []
test_accuracies = []
train_times = []
count_dead_neurons = []


def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 10

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="MNIST/data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root="MNIST/data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model
    model = SimpleCNN().to(device)
    print(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        end = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train(train_loader, model, criterion, optimizer, device, epoch)
        test(test_loader, model, criterion, device)
        end = time.time() - end
        train_times.append(np.round(end, 2))
        print(f"Time: {end:.2f} s")
    
    print(f"Finished training in {sum(train_times):.2f} s")
    
    # Save model
    torch.save(model.state_dict(), "MNIST/model.ckpt")
    
    # Plot metrics
    plot_metrics()


def plot_metrics():
    plt.figure(figsize=(12, 12))
    
    # Plot Training- and Test-Loss
    ax1 = plt.subplot(2, 2, 1)
    line1, = ax1.plot(range(len(train_losses)), train_losses, 'o-', label="Training Loss")
    line2, = ax1.plot(range(len(test_losses)), test_losses, 'o-', label="Test Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid()
    mplcursors.cursor([line1, line2], hover=True)
    
    # Plot Test-Accuracy
    ax2 = plt.subplot(2, 2, 2)
    line3, = ax2.plot(range(len(test_accuracies)), test_accuracies, 'o-', label="Test Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid()
    mplcursors.cursor([line3], hover=True)
    
    # Plot Train-Time over Epochs
    ax3 = plt.subplot(2, 2, 3)
    line4, = ax3.plot(range(len(train_times)), train_times, 'o-', label="Train Time")
    ax3.set_title(f"Train Time ({sum(train_times):.2f} s in total)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Time (s)")
    ax3.legend()
    ax3.grid()
    mplcursors.cursor([line4], hover=True)
    
    # Show plot
    plt.tight_layout()
    plt.show()

def train(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    size = len(train_loader.dataset)
    for batch, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Avg loss: {avg_loss:>8f}, Epoch: {epoch:>3d}") 


def test(test_loader, model, criterion, device):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
    test_loss /= num_batches
    accuracy = 100. * correct / size
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    main()
