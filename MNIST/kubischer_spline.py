import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

    


class CubicSpline(nn.Module):
    def __init__(self, x, y, f):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        x = torch.tensor(x, device=self.device)
        y = torch.tensor(y, device=self.device)
        self.func = f
        
        super(CubicSpline, self).__init__()
        
        self._x1 = -2
        self._x2 = 2
        self._y1 = f(torch.tensor(self._x1)).item()
        self._y2 = f(torch.tensor(self._x2)).item()
        
        
        self._x = torch.tensor(x)
        self._y = y
                
        # Calc gradient of f at x[0] and x[-1]:
        self._x.requires_grad = True
        ysgrad = torch.autograd.grad(f(self._x), self._x, torch.ones_like(self._x), create_graph=True)[0]        
        self._x.requires_grad = False
        
        # Randbedingungen (erste Ableitungen an den Enden der Funktion) + Skalierung auf x-Abstand
        self._y0s = (ysgrad[0] * (x[1] - x[0])).to(self.device)
        self._yns = (ysgrad[-1] * (x[-1] - x[-2])).to(self.device)
        
        self._ys = self._solve_linear_system()


    def forward(self, t):
        # Berechne die Werte der Spline-Funktion
        # self._spline(t), wenn t in [x1, x2] liegt
        # self._y1 + t - self._x1, wenn t < x1
        # self._y2 + t - self._x2, wenn t > x2
        return torch.where(
            (t < self._x1),
            self._y1 + t - self._x1,
            torch.where(
                (t > self._x2),
                self._y2 + t - self._x2,
                self._spline(t)
            )
        )
    
    
    def _spline(self, t):
        n = len(self._x) - 1
        s = torch.zeros_like(t)
        
        for i in range(n):
            s = torch.where(
                (self._x[i] <= t) & (t <= self._x[i+1]),
                self._hermite(self._reparametrize(t, i), self._y[i], self._y[i+1], self._ys[i], self._ys[i+1]),
                s
            )
        return s
        
        
    def _reparametrize(self, t, i):
        t = (t - self._x[i]) / (self._x[i+1] - self._x[i])
        return t
    
    
    def _solve_linear_system(self):
        n = len(self._y)
        
        # Matrix A mit eingebauten Randbedingungen:
        # [4, 1, 0, 0, ..., 0]
        # [1, 4, 1, 0, ..., 0]
        # [0, 1, 4, 1, ..., 0]
        # ...
        # [0, 0, ..., 0, 1, 4]
        A = (
            4 * torch.eye(n-2) 
            + torch.diag(torch.ones(n-3), 1) 
            + torch.diag(torch.ones(n-3), -1)
            ).to(self.device)
        
        # Vektor b mit eingebauten Randbedingungen:
        # [3 * (y_1 - y_0)]
        # [3 * (y_2 - y_0)]
        # ...
        # [3 * (y_n - y_{n-2})]
        b = 3 * (self._y[2:] - self._y[:-2])
        b[0] -= self._y0s
        b[n-3] -= self._yns
        b = torch.tensor(b)
        
        # Löse das lineare Gleichungssystem
        c = torch.linalg.solve(A, b)
        
        
        # Füge die Randbedingungen wieder ein
        return torch.cat([torch.tensor([self._y0s], device=self.device), c, torch.tensor([self._yns], device=self.device)])
    
    def _hermite(self, t, y0, y1, y0s, y1s):
        term1 = torch.pow(t, 3) * (2*y0 - 2*y1 + y0s + y1s)
        term2 = torch.pow(t, 2) * (-3*y0 + 3*y1 - 2*y0s - y1s)
        term3 = t * y0s
        term4 = y0
        return term1 + term2 + term3 + term4


    def __str__(self):
        return f"CubicSpline mimicing {self.func.__name__} with {len(self._x)} points"
    
    
    def __repr__(self):
        return str(self)

    
if __name__ == "__main__":
    torch.set_default_device(device)
    n = 7 # Funktionsabschnitte
    s = F.sigmoid
    f_name = s.__name__
    x = torch.linspace(-5.5, 5.5, n+1)
    y = s(x)
    print(y)
    
    x.requires_grad = True
    ys = torch.autograd.grad(s(x), x, torch.ones_like(x), create_graph=True)[0]
    x.requires_grad = False
    
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ys = ys.detach().cpu().numpy()
    
    plt.figure()
    plt.subplots_adjust(hspace=0.5)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.set_title(f"{f_name} vs Cubic Spline")
    ax2.set_title("Derivatives")
    
    ax1.plot(x, y, "*g")
    ax2.plot(x, ys, "*g")
    
    
    t = torch.linspace(-15.5, 15.5, 100)
    yf = s(t)
    f = CubicSpline(x, y, s)
    yp = f(t)
    
    t.requires_grad = True
    yfs = torch.autograd.grad(s(t), t, torch.ones_like(t), create_graph=True)[0]
    yps = torch.autograd.grad(f(t), t, torch.ones_like(t), create_graph=True)[0]
    t.requires_grad = False

    

    t = t.detach().cpu().numpy()
    yf = yf.detach().cpu().numpy()
    yp = yp.detach().cpu().numpy()
    yfs = yfs.detach().cpu().numpy()
    yps = yps.detach().cpu().numpy()
    
    

    ax1.plot(t, yf, "-r", label=f"{f_name}")
    ax1.plot(t, yp, "-g", label="Spline")
    
    ax2.plot(t, yfs, "-r", label=f"{f_name}")
    ax2.plot(t, yps, "-g", label="Spline")
    
    plt.legend()
    plt.show()
