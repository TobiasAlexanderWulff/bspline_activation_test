import torch
import torch.nn as nn
from enum import Enum
from functools import partial
import matplotlib.pyplot as plt


class Operation(Enum):#
    XY = partial(lambda x, y: torch.column_stack((x, y)))
    X = partial(lambda x, y: x)
    Y = partial(lambda x, y: y)
    SUM = partial(torch.add)
    DIF = partial(torch.sub)
    MUL = partial(torch.mul)
    MAX = partial(torch.max)
    MIN = partial(torch.min)


bspline_id = 30

k = 3
control_points = torch.tensor([
    [0.0, 0.0],
    [0.2, 0.8],
    [1.0, 1.0],
])
control_points.requires_grad = True
n = len(control_points) - 1

#knots
min_knot = 0.0
max_knot = 1.0
repeated_start_knots = k # <= k
repeated_end_knots = k # <= k
num_knots = n + 1 + k
free_knots = num_knots - repeated_start_knots - repeated_end_knots
knots = torch.cat((
    torch.tensor([min_knot] * max(repeated_start_knots - 1, 0)),
    torch.linspace(
        min_knot, 
        max_knot, 
        free_knots + min(1, repeated_start_knots) + min(1, repeated_end_knots)
        ),
    torch.tensor([max_knot] * max(repeated_end_knots - 1, 0)),
    ), dim=0
)
knots.requires_grad = False


class BSpline(nn.Module):
    def __init__(self, operation):
        super(BSpline, self).__init__()
        self.operation = operation
        
    def forward(self, t):
        return self.bspline(t)
    
    def bspline(self, t):
        x = torch.zeros_like(t)
        y = torch.zeros_like(t)
        for i in range(n + 1):
            x += control_points[i][0] * self.bspline_basis(i, k, t)
            y += control_points[i][1] * self.bspline_basis(i, k, t)
        return self.operation.value(x, y)
    
    def bspline_basis(self, i, k, t):
        if k == 1:
            return torch.where((knots[i] <= t) & (t < knots[i + 1]), torch.tensor(1.0), torch.tensor(0.0))
        else:
            numerator1 = t - knots[i]
            numerator2 = knots[i + k] - t
            denominator1 = knots[i + k - 1] - knots[i]
            denominator2 = knots[i + k] - knots[i + 1]
            term1 = torch.tensor(0.0)
            term2 = torch.tensor(0.0)
            if denominator1 != 0:
                term1 = (numerator1 / denominator1) * self.bspline_basis(i, k - 1, t)
            if denominator2 != 0:
                term2 = (numerator2 / denominator2) * self.bspline_basis(i + 1, k - 1, t)
            return term1 + term2
            
            
           
        
    def __str__(self):
        return f"BSpline({k}, {n}) with control points {control_points}"
    
    def __repr__(self):
        return str(self)
    

def plot(ax, model):
    ax.set_title(f"{model.operation.name}")
    t = torch.linspace(knots[0], knots[-1], 100).to(device)
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0)
    f = model(t)
    if model.operation == Operation.XY:
        x, y = f.chunk(2, dim=1)
        ax.plot(control_points[:, 0].cpu().detach().numpy(), control_points[:, 1].cpu().detach().numpy(), 'ro')
        ax.legend(["Control Points"])
    else:
        x, y = t, f
        ax.set_xlabel("t")
        ax.set_ylabel("f(t)", rotation=0)
    x, y = x.squeeze().cpu().detach().numpy(), y.squeeze().cpu().detach().numpy()
    ax.plot(x, y)
    ax.legend(["Control Points", "BSpline"])
    ax.grid()
    return ax


def plot_derivative(ax, model):
    ax.set_title(f"{model.operation.name} Derivative")
    t = torch.linspace(knots[0], knots[-1], 100).to(device)
    t.requires_grad = True
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0)
    f = model(t)
    if model.operation == Operation.XY:
        x, y = f.chunk(2, dim=1)
        x = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
        y = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
        ax.plot(control_points[:, 0].cpu().detach().numpy(), control_points[:, 1].cpu().detach().numpy(), 'ro')
        ax.legend(["Control Points"])
    else:
        grad = torch.autograd.grad(f, t, torch.ones_like(f), create_graph=True)[0]
        x, y = t, grad
        ax.set_xlabel("t")
        ax.set_ylabel("f(t)", rotation=0)

    x, y = x.squeeze().cpu().detach().numpy(), y.squeeze().cpu().detach().numpy()
    ax.plot(x, y)
    ax.legend(["Control Points", "BSpline"])
    ax.grid()
    return ax


def plot_everything():
    
    # Plot all operations 
    
    plt.figure(figsize=(20, 12))
    plt.suptitle(f"BSpline #{bspline_id:03d} --- n = {n}, k = {k}, knots = {knots.tolist()}")
    for i, operation in enumerate(list(Operation)):
        model = BSpline(operation).to(device)
        ax = plt.subplot(4, 4, i + 1)
        plot(ax, model)
        ax = plt.subplot(4, 4, 8 + i + 1)
        plot_derivative(ax, model)
    

    plt.tight_layout()
    plt.savefig(f"bsplines/{bspline_id:03d}.png")
    
    plt.show()


    
if __name__ == "__main__":
    current_operation = Operation.XY
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BSpline(current_operation).to(device)
    t = torch.tensor([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]).to(device)
    print(f"\nOut: {model(t)}")
    plot_everything()
    


        