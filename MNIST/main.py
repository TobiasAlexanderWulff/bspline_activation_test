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
from enum import Enum
from functools import partial
import json
import os
from bspline import BSpline, Operation





class AF(Enum):
    RELU = partial(F.relu)
    SIGMOID = partial(F.sigmoid)
    TANH = partial(F.tanh)
    LEAKY_RELU = partial(F.leaky_relu)
    B_SPLINE_X = partial(BSpline(Operation.X))
    B_SPLINE_Y = partial(BSpline(Operation.Y))
    B_SPLINE_SUM = partial(BSpline(Operation.SUM))
    B_SPLINE_DIF = partial(BSpline(Operation.DIF))
    B_SPLINE_MUL = partial(BSpline(Operation.MUL))
    B_SPLINE_MAX = partial(BSpline(Operation.MAX))
    B_SPLINE_MIN = partial(BSpline(Operation.MIN))
    
seeds = [2934, 1234, 9859283]


# Hyperparameters
batch_size = 128
learning_rate = 0.01
num_epochs = 5
  
# Listen zum Speichern der Metriken
train_losses = []
test_losses = []
test_accuracies = []
train_times = []
l2_norms = []
count_dead_neurons = []

# Hyperparameters as dictionary for saving to JSON
metrics = {
    "dataset": "MNIST",
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
}



class SimpleCNN(nn.Module):
    def __init__(self, activation, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = activation
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.activation.value(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def __str__(self):
        return (
            f"SimpleCNN({self.fc1.in_features}, {self.fc1.out_features}, {self.fc2.out_features}) "
            f"with {self.conv1.kernel_size[0]}x{self.conv1.kernel_size[1]} kernel size "
            f"and {self.pool.kernel_size} pooling kernel size"
        )


def main(activation, seed):
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(seed)

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="MNIST/data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_dataset = datasets.MNIST(root="MNIST/data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

    # Initialize model
    model = SimpleCNN(activation).to(device)
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
        print(f"Time: {sum(train_times):.2f}s (+{end:.2f}s)")
    
    print(f"Finished training in {sum(train_times):.2f}s\n")
    
    # Save model
    seed_idx = seeds.index(seed)
    if type(activation.value.func) == BSpline:
        path = f"MNIST/{activation.name}/model/model_{seed_idx:03d}_k{activation.value.func.k}_{num_epochs}.pt"
    else:
        path = f"MNIST/{activation.name}/model/model_{seed_idx:03d}_{num_epochs}.pt"
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
    # Plot metrics
    plot_metrics(activation, model)
    
    # Save hyperparameters
    if type(activation.value.func) == BSpline:
        path = f"MNIST/{activation.name}/metrics/metrics_{seed_idx:03d}_k{activation.value.func.k}_{num_epochs}.json"
    else:
        path = f"MNIST/{activation.name}/metrics/metrics_{seed_idx:03d}_{num_epochs}.json"
    metrics["activation"] = activation.name
    metrics["train_losses"] = train_losses
    metrics["test_losses"] = test_losses
    metrics["test_accuracies"] = test_accuracies
    metrics["train_times"] = train_times
    metrics["l2_norms"] = l2_norms
    metrics["count_dead_neurons"] = count_dead_neurons
    if activation.name.startswith("B_SPLINE"):
        metrics["k"] = activation.value.func.k
        metrics["operation"] = activation.value.func.operation.name
        metrics["control_points"] = activation.value.func.control_points.tolist()
        metrics["knots"] = activation.value.func.knots.tolist()
        metrics["min_knot"] = activation.value.func.min_knot
        metrics["max_knot"] = activation.value.func.max_knot
        metrics["cp_mode"] = activation.value.func.cp_mode
        metrics["cp_count"] = activation.value.func.cp_count
        metrics["repeated_start_knots"] = activation.value.func.repeated_start_knots
        metrics["repeated_end_knots"] = activation.value.func.repeated_end_knots
        metrics["num_knots"] = activation.value.func.num_knots
        metrics["free_knots"] = activation.value.func.free_knots
        metrics["n"] = activation.value.func.n
    with open(path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {path}")


def plot_metrics(activation, model):
    plt.figure(figsize=(12, 12))
    
    # Plot Training- and Test-Loss
    ax1 = plt.subplot(3, 2, 1)
    line1, = ax1.plot(range(1, len(train_losses) + 1), train_losses, 'o-', label="Training Loss")
    line2, = ax1.plot(range(1, len(test_losses) + 1), test_losses, 'o-', label="Test Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_xticks(range(1, len(train_losses) + 1))
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid()
    min_loss = min(test_losses)
    min_loss_idx = test_losses.index(min_loss)
    ax1.annotate(f"Min Loss: {min_loss:.2f}", (min_loss_idx, min_loss), textcoords="offset points", xytext=(0, 10), ha='center')
    mplcursors.cursor([line1, line2], hover=False)
    
    # Plot Test-Accuracy
    ax2 = plt.subplot(3, 2, 2)
    line3, = ax2.plot(range(1, len(test_accuracies) + 1), test_accuracies, 'o-', label="Test Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_xticks(range(1, len(test_accuracies) + 1))
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid()
    max_acc = max(test_accuracies)
    max_acc_idx = test_accuracies.index(max_acc)
    ax2.annotate(f"Max Accuracy: {max_acc:.2f}%", (max_acc_idx, max_acc), textcoords="offset points", xytext=(0, 10), ha='center')
    mplcursors.cursor([line3], hover=False)
    
    # Plot Train-Time over Epochs
    ax3 = plt.subplot(3, 2, 3)
    line4, = ax3.plot(range(1, len(train_times) + 1), train_times, 'o-', label="Train Time")
    ax3.set_title(f"Train Time ({sum(train_times):.2f}s in total)")
    ax3.set_xlabel("Epoch")
    ax3.set_xticks(range(1, len(train_times) + 1))
    ax3.set_ylabel("Time (s)")
    ax3.legend()
    ax3.grid()
    mplcursors.cursor([line4], hover=False)
    
    # Plot L2-Norm over Epochs
    ax4 = plt.subplot(3, 2, 4)
    line5, = ax4.plot(range(1, len(l2_norms) + 1), l2_norms, 'o-', label="L2-Norm")
    ax4.set_title("L2-Norm")
    ax4.set_xlabel("Epoch")
    ax4.set_xticks(range(1, len(l2_norms) + 1))
    ax4.set_ylabel("L2-Norm")
    ax4.legend()
    ax4.grid()
    mplcursors.cursor([line5], hover=False)
    
    # Plot Activation-Function
    ax5 = plt.subplot(3, 2, 6)
    ax5.set_title(f"Activation Function {activation.name}")
    t = torch.linspace(-5, 5, 100).to(device)
    t.requires_grad = True
    ax5.set_xlabel("t")
    ax5.set_ylabel("f(t)", rotation=0)
    x, y = t, model.activation.value(t)
    x, y = x.squeeze().cpu().detach().numpy(), y.squeeze().cpu().detach().numpy()
    ax5.plot(x, y)
    ax5.grid()
        
    # Save plot
    seed_idx = seeds.index(seed)
    if type(activation.value.func) == BSpline:
        path = f"MNIST/{activation.name}/metrics/metrics_{seed_idx:03d}_k{activation.value.func.k}_{num_epochs}.png"
    else:
        path = f"MNIST/{activation.name}/metrics/metrics_{seed_idx:03d}_{num_epochs}.png"
    plt.tight_layout()
    plt.savefig(path)
    print(f"Metrics saved to {path}\n")
    

    

def train(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    batch_l2_norms = []
    size = len(train_loader.dataset)
    for batch, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Berechne L2-Norm für diesen Batch
        batch_l2_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                batch_l2_norm += p.grad.data.norm(2).item() ** 2
        batch_l2_norm = batch_l2_norm ** 0.5
        batch_l2_norms.append(batch_l2_norm)
        
        optimizer.step()
        running_loss += loss.item()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    # Mittelwert der L2-Normen aller Batches in dieser Epoche
    avg_l2_norm = sum(batch_l2_norms) / len(batch_l2_norms)
    l2_norms.append(avg_l2_norm)
        
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Avg loss: {avg_loss:>8f}, Epoch: {epoch+1:>3d}") 


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


def reset_lists():
    train_losses.clear()
    test_losses.clear()
    test_accuracies.clear()
    train_times.clear()
    l2_norms.clear()
    count_dead_neurons.clear()


if __name__ == "__main__":
    ## Set seed for reproducibility
    #seed = seeds[0]
    #
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    for seed in seeds:
        reset_lists()
        torch.manual_seed(seed)
        np.random.seed(seed)
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        print(f"Seed: {seed}")
        metrics["seed"] = seed
                
        for activation in AF:
            
            # * Uncomment to skip all non B-Spline activations
            #if not activation.name.startswith("B_SPLINE"):
            #    continue
            
            # * Uncomment to skip all B-Spline activations
            #if activation.name.startswith("B_SPLINE"):
            #    continue
            
            reset_lists()
            os.makedirs(f"MNIST/{activation.name}", exist_ok=True)
            os.makedirs(f"MNIST/{activation.name}/model", exist_ok=True)
            os.makedirs(f"MNIST/{activation.name}/metrics", exist_ok=True)
            print(f"\nActivation: {activation.name}")
            if type(activation.value.func) == BSpline:
                for k in range(2, 4):
                    reset_lists()
                    print(f"K: {k}")
                    activation.value.func.set_k(k)
                    activation.value.func.set_seed(seed)
                    main(activation, seed)
            else:
                main(activation, seed)
