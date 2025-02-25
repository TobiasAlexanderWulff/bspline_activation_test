import torch
import torch.nn as nn
from enum import Enum
from functools import partial
import matplotlib.pyplot as plt
import mplcursors


class Operation(Enum):
    XY = partial(lambda x, y: torch.column_stack((x, y)))
    X = partial(lambda x, y: x)
    Y = partial(lambda x, y: y)
    SUM = partial(torch.add)
    DIF = partial(torch.sub)
    MUL = partial(torch.mul)
    MAX = partial(torch.max)
    MIN = partial(torch.min)


bspline_id = 50

# "normal" or "clamp" or "normalize" or "normalize knots"
t_mode = "normalize knots" 


# Beim Boor-Cox-Algorithmus kann es wohl zu Rundungsfehlern kommen, wodurch die Basisfunktion falsch ausgwertet werden kann.
# epsilon ist ein kleiner Wert, der auf die Min- und Max-Knoten addiert bzw. subtrahiert wird, um sicherzustellen, dass die
# Werte innerhalb der Knoten liegen.
# ? Quelle: leider nur chatGPT bisher ☹️, aber funktioniert
epsilon = 0.01


class BSpline(nn.Module):
    def __init__(
        self, 
        operation, 
        k=3,
        cp_mode="random", # "set" or "random"
        cp_count=4, # only used if cp_mode="random"
        seed=42,
        ):
        super(BSpline, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        self.operation = operation
        self.k = k # order of the B-Spline (not the degree! k = degree + 1)
        self.cp_mode = cp_mode
        self.cp_count = cp_count
        self._gen_control_points(self.cp_mode, self.cp_count)
        self.min_knot = 0.0
        self.max_knot = 1.0
        self._gen_knots()
        
    def _gen_control_points(self, cp_mode, cp_count):
        if cp_mode == "set":
            self.control_points = torch.tensor([
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 1.0],
                [1.5, 0.5]
                ])
        elif cp_mode == "random":
            self.control_points = torch.rand((cp_count, 2))
        else:
            raise ValueError(f"Unknown cp_mode: {cp_mode}")
        self.control_points.requires_grad = True
        self.n = len(self.control_points) - 1
    
    def _gen_knots(self):
        self.repeated_start_knots = self.k # <= k
        self.repeated_end_knots = self.k # <= k
        self.num_knots = self.n + 1 + self.k
        self.free_knots = self.num_knots - self.repeated_start_knots - self.repeated_end_knots
        self.knots = torch.cat((
            torch.tensor([self.min_knot] * max(self.repeated_start_knots - 1, 0)),
            torch.linspace(
                self.min_knot, 
                self.max_knot, 
                self.free_knots + min(1, self.repeated_start_knots) + min(1, self.repeated_end_knots)
                ),
            torch.tensor([self.max_knot] * max(self.repeated_end_knots - 1, 0)),
            ), dim=0,
        ).to(self.device)
        self.knots.requires_grad = False
        
    def forward(self, t):
        match(t_mode):
            case "normal":
                pass
            case "clamp":
                t = torch.clamp(t, self.knots[self.k-1], self.knots[self.n+1])
            case "normalize":
                t_min = torch.min(t)
                t_max = torch.max(t)
                t = (t - t_min) / (t_max - t_min)
                t = t * ((self.max_knot - epsilon) - (self.min_knot + epsilon)) + self.min_knot + epsilon
            case "normalize knots":
                self.min_knot = torch.min(t).item() - epsilon
                self.max_knot = torch.max(t).item() + epsilon
                self._gen_knots()
            case _:
                raise ValueError(f"Unknown t_mode: {t_mode}")
        
        
        return self.bspline(t)
    
    def bspline(self, t):
        x = torch.zeros_like(t)
        y = torch.zeros_like(t)
        for i in range(self.n + 1):
            x += self.control_points[i][0] * self.bspline_basis(i, self.k, t) if not self.operation == Operation.Y else torch.tensor(0.0)
            y += self.control_points[i][1] * self.bspline_basis(i, self.k, t) if not self.operation == Operation.X else torch.tensor(0.0)
        return self.operation.value(x, y)
    
    def bspline_basis(self, i, k, t):
        if k == 1:
            return torch.where((self.knots[i] <= t) & (t < self.knots[i + 1]), torch.tensor(1.0), torch.tensor(0.0))
        elif k >= 2:
            numerator1 = t - self.knots[i]
            numerator2 = self.knots[i + k] - t
            denominator1 = self.knots[i + k - 1] - self.knots[i]
            denominator2 = self.knots[i + k] - self.knots[i + 1]
            term1 = torch.tensor(0.0)
            term2 = torch.tensor(0.0)
            if denominator1 != 0:
                term1 = (numerator1 / denominator1) * self.bspline_basis(i, k - 1, t)
            if denominator2 != 0:
                term2 = (numerator2 / denominator2) * self.bspline_basis(i + 1, k - 1, t)
            return term1 + term2
        else:
            raise ValueError(f"Unknown k: {k}")
    
    
    def set_seed(self, seed):
        torch.manual_seed(seed)  
        self._gen_control_points(self.cp_mode, self.cp_count)
        self._gen_knots()      
    
    def set_k(self, k):
        self.k = k
        self._gen_knots()
    
    
        
    def __str__(self):
        return f"BSpline({self.k}, {self.n}) with control points {self.control_points}"
    
    def __repr__(self):
        return str(self)
    


def plot(ax, model):
    ax.set_title(f"{model.operation.name}")
    t = torch.linspace(-5, 5, 100).to(device)
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0)
    f = model(t)
    if model.operation == Operation.XY:
        x, y = f.chunk(2, dim=1)
        ax.plot(model.control_points[:, 0].cpu().detach().numpy(), model.control_points[:, 1].cpu().detach().numpy(), 'ro')
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
    t = torch.linspace(-5, 5, 100).to(device)
    t.requires_grad = True
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0)
    f = model(t)
    if model.operation == Operation.XY:
        x, y = f.chunk(2, dim=1)
        x = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
        y = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
        ax.plot(model.control_points[:, 0].cpu().detach().numpy(), model.control_points[:, 1].cpu().detach().numpy(), 'ro')
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
    model = BSpline(Operation.XY).to(device)
    model(torch.linspace(-5, 5, 100).to(device)) # to set the min_knot and max_knot
    plt.suptitle(
        f"BSpline #{bspline_id:03d} --- n = {model.n}, "
        f"k = {model.k}, knots = {model.knots.tolist()}, "
        f"CPs = {torch.round(model.control_points, decimals=2).tolist()}"
        )
    for i, operation in enumerate(list(Operation)):
        model = BSpline(operation).to(device)
        ax = plt.subplot(4, 4, i + 1)
        plot(ax, model)
        ax = plt.subplot(4, 4, 8 + i + 1)
        plot_derivative(ax, model)
    

    plt.tight_layout()
    plt.savefig(f"bsplines/{bspline_id:03d}.png")
    mplcursors.cursor(hover=True)
    plt.show()


    
if __name__ == "__main__":
    current_operation = Operation.XY
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BSpline(current_operation).to(device)
    t = torch.tensor([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]).to(device)
    print(f"\nOut: {model(t)}")
    plot_everything()
    


        