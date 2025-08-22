import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader

# Import gamma calculation functions
import math
from functools import lru_cache
from typing import Callable, Tuple, Dict, Any, Iterable
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

# ----------------------------
# GPU DataLoader Implementation
# ----------------------------

class GPUDataLoader:
    """Drop-in replacement for DataLoader that preloads data to GPU"""
    
    def __init__(self, dataset, batch_size=1, shuffle=False, device='cpu', max_samples=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.max_samples = max_samples
        
        # Preprocess and load data to GPU
        self.data, self.targets = self._preload_data(dataset)
        self.n_samples = len(self.data)
        
    def _preload_data(self, dataset):
        """Preprocess entire dataset and load to GPU"""
        print(f"Preprocessing and loading data to {self.device}...")
        
        # Determine how many samples to use
        n_total = len(dataset)
        n_use = min(self.max_samples or n_total, n_total)
        
        if n_use < n_total:
            print(f"Using {n_use}/{n_total} samples")
            indices = torch.randperm(n_total)[:n_use]
            subset = torch.utils.data.Subset(dataset, indices)
            temp_loader = DataLoader(subset, batch_size=n_use, shuffle=False)
        else:
            temp_loader = DataLoader(dataset, batch_size=n_total, shuffle=False)
        
        try:
            # Load and preprocess all data at once
            data, targets = next(iter(temp_loader))
            
            # Move to device
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            print(f"Successfully loaded {len(data)} samples to {self.device}")
            return data, targets
            
        except torch.cuda.OutOfMemoryError:
            print("GPU OOM! Trying with fewer samples...")
            fallback_samples = min(10000, n_use // 2)
            return self._preload_data_with_limit(dataset, fallback_samples)
    
    def _preload_data_with_limit(self, dataset, max_samples):
        """Fallback method with limited samples"""
        indices = torch.randperm(len(dataset))[:max_samples]
        subset = torch.utils.data.Subset(dataset, indices)
        temp_loader = DataLoader(subset, batch_size=max_samples, shuffle=False)
        data, targets = next(iter(temp_loader))
        return data.to(self.device), targets.to(self.device)
    
    def __iter__(self):
        """Iterator that yields batches like standard DataLoader"""
        if self.shuffle:
            indices = torch.randperm(self.n_samples, device=self.device)
        else:
            indices = torch.arange(self.n_samples, device=self.device)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.data[batch_indices], self.targets[batch_indices]
    
    def __len__(self):
        """Number of batches"""
        return (self.n_samples + self.batch_size - 1) // self.batch_size

def get_dataloader(dataset, config, is_train=True):
    """Factory function to create appropriate dataloader"""
    if config.get('use_gpu_loader', False) and is_train:
        # Use GPU loader for training with memory management
        return GPUDataLoader(dataset, 
                           batch_size=config['batch_size'], 
                           shuffle=True,
                           device=config['device'],
                           max_samples=config.get('max_gpu_samples', None))
    else:
        # Use standard DataLoader
        return DataLoader(dataset, 
                         batch_size=config['batch_size'], 
                         shuffle=is_train,
                         num_workers=config.get('num_workers', 2))

# ----------------------------
# Gamma calculation utilities
# ----------------------------

SQRT2   = math.sqrt(2.0)
SQRTPI  = math.sqrt(math.pi)
SQRT2PI = math.sqrt(2.0 * math.pi)
INV_SQRT2PI = 1.0 / SQRT2PI
LOG_SQRT2PI = 0.5 * math.log(2.0 * math.pi)

def phi(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x * x) * INV_SQRT2PI

def Phi(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / SQRT2))

def log_Phi(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    lo = x < -10.0
    hi = x >  10.0
    md = ~(lo | hi)
    if np.any(lo):
        xm = x[lo]
        corr = 1.0 - 1.0 / (xm * xm)
        corr = np.maximum(corr, 1e-12)
        out[lo] = (-0.5 * xm * xm) - LOG_SQRT2PI - np.log(np.abs(xm)) + np.log(corr)
    if np.any(md):
        xm = x[md]
        out[md] = np.log(np.clip(Phi(xm), 1e-300, 1.0))
    if np.any(hi):
        out[hi] = 0.0
    return float(out) if out.shape == () else out

@lru_cache(maxsize=None)
def _leg_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x, w = leggauss(n)
    return x.astype(np.float64), w.astype(np.float64)

def _gl_chunk_expectation(fn_z: Callable[[np.ndarray], np.ndarray], L: float, n: int) -> float:
    x, w = _leg_nodes(n)
    z = L * x
    vals = fn_z(z).astype(np.float64) * phi(z)
    return float(L * (w @ vals))

def normal_expectation_integral(fn_z: Callable[[np.ndarray], np.ndarray],
                                L_init: float = 10.0, n_init: int = 128,
                                tol_rel: float = 1e-9, tol_abs: float = 1e-12,
                                max_L_steps: int = 5, max_n_steps: int = 6) -> float:
    L = float(L_init)
    target = None
    eps = 1e-16
    I_prev = None

    for _ in range(max_L_steps):
        n = int(n_init)
        I_prev = _gl_chunk_expectation(fn_z, L=L, n=n)
        converged_n = False
        for _ in range(max_n_steps):
            n *= 2
            I_curr = _gl_chunk_expectation(fn_z, L=L, n=n)
            if abs(I_curr - I_prev) <= max(tol_abs, tol_rel * (abs(I_curr) + eps)):
                target = I_curr
                converged_n = True
                break
            I_prev = I_curr
        if converged_n:
            I_Lplus = _gl_chunk_expectation(fn_z, L=L+2.0, n=max(n_init, n//2))
            if abs(I_Lplus - target) <= max(tol_abs, tol_rel * (abs(I_Lplus) + eps)):
                return float(I_Lplus)
        L += 2.0
    return float(I_prev if I_prev is not None else 0.0)

# Activation functions and derivatives
def relu(u: np.ndarray) -> np.ndarray:
    return np.maximum(u, 0.0)

def relu_prime(u: np.ndarray) -> np.ndarray:
    return (u > 0.0).astype(np.float64)

def gelu(u: np.ndarray) -> np.ndarray:
    return u * Phi(u)

def gelu_prime(u: np.ndarray) -> np.ndarray:
    return Phi(u) + u * phi(u)

def tanh_fn(u: np.ndarray) -> np.ndarray:
    return np.tanh(u)

def tanh_prime(u: np.ndarray) -> np.ndarray:
    t = np.tanh(u)
    return 1.0 - t * t

def sigmoid(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    out = np.empty_like(u)
    pos = u >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    e = np.exp(u[neg])
    out[neg] = e / (1.0 + e)
    return out

def sigmoid_prime(u: np.ndarray) -> np.ndarray:
    s = sigmoid(u)
    return s * (1.0 - s)

def swish(u: np.ndarray) -> np.ndarray:
    s = sigmoid(u)
    return u * s

def swish_prime(u: np.ndarray) -> np.ndarray:
    s = sigmoid(u)
    return s + u * s * (1.0 - s)

def leaky_relu(u: np.ndarray, alpha_neg: float = 0.01) -> np.ndarray:
    return np.where(u > 0.0, u, alpha_neg * u)

def leaky_relu_prime(u: np.ndarray, alpha_neg: float = 0.01) -> np.ndarray:
    return np.where(u > 0.0, 1.0, alpha_neg)

def elu(u: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(u > 0.0, u, alpha * (np.exp(u) - 1.0))

def elu_prime(u: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(u > 0.0, 1.0, alpha * np.exp(u))

def selu(u: np.ndarray) -> np.ndarray:
    SELU_ALPHA = 1.6732632423543772
    SELU_LAMBDA = 1.0507009873554805
    return SELU_LAMBDA * elu(u, alpha=SELU_ALPHA)

def selu_prime(u: np.ndarray) -> np.ndarray:
    SELU_ALPHA = 1.6732632423543772
    SELU_LAMBDA = 1.0507009873554805
    return SELU_LAMBDA * elu_prime(u, alpha=SELU_ALPHA)

def mish(u: np.ndarray) -> np.ndarray:
    sp = np.log1p(np.exp(-np.abs(u))) + np.maximum(u, 0.0)  # softplus
    return u * np.tanh(sp)

def mish_prime(u: np.ndarray) -> np.ndarray:
    sp = np.log1p(np.exp(-np.abs(u))) + np.maximum(u, 0.0)  # softplus
    t = np.tanh(sp)
    return t + u * sigmoid(u) * (1.0 - t * t)

def moments_via_integral(m: float, s: float, act: Callable[..., np.ndarray], 
                         dact: Callable[..., np.ndarray], integral_opts: Dict[str, Any] | None = None,
                         **act_kwargs) -> Tuple[float, float, float, float]:
    integral_opts = integral_opts or {}
    def fz(z: np.ndarray) -> np.ndarray:
        u = m + s * z
        return act(u, **act_kwargs)
    def fpz(z: np.ndarray) -> np.ndarray:
        u = m + s * z
        return dact(u, **act_kwargs)
    mu  = normal_expectation_integral(fz,  **integral_opts)
    Ef2 = normal_expectation_integral(lambda z: fz(z) ** 2, **integral_opts)
    Efp = normal_expectation_integral(fpz, **integral_opts)
    Efp2= normal_expectation_integral(lambda z: fpz(z) ** 2, **integral_opts)
    return mu, Ef2, Efp, Efp2

def gammas_from_moments(a: float, mu: float, Ef2: float, Efp: float, Efp2: float,
                        eps_var: float = 1e-12) -> Tuple[float, float, float]:
    var_f = max(Ef2 - mu * mu, eps_var)
    sigma = math.sqrt(var_f)
    scale = (a / sigma) ** 2
    g2 = Efp2
    g1 = scale * g2
    g0 = scale * (Efp ** 2)
    return g0, g1, g2

# Closed-form moments for surrogate calculations
def relu_moments_closed(m: float, s: float) -> Tuple[float, float, float, float]:
    t = m / s
    Ph = Phi(t); ph = phi(t)
    mu  = s * ph + m * Ph
    Ef2 = (s * s + m * m) * Ph + m * s * ph
    Efp = Ph
    Efp2= Ph
    return float(mu), float(Ef2), float(Efp), float(Efp2)

@lru_cache(maxsize=None)
def _herm_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x, w = hermgauss(n)
    return x.astype(np.float64), w.astype(np.float64)

def gh_expectation(fn: Callable[[np.ndarray], np.ndarray], n_nodes: int = 128) -> float:
    """E[fn(Z)] with Z~N(0,1) via Gauss–Hermite"""
    x, w = _herm_nodes(n_nodes)
    z = math.sqrt(2.0) * x
    vals = fn(z).astype(np.float64)
    return float((w @ vals) / SQRTPI)

def get_exact_gamma_function(activation: str):
    """Get exact gamma calculation function for given activation"""
    activation_map = {
        'relu': (relu, relu_prime, {}),
        'gelu': (gelu, gelu_prime, {}),
        'tanh': (tanh_fn, tanh_prime, {}),
        'sigmoid': (sigmoid, sigmoid_prime, {}),
        'swish': (swish, swish_prime, {}),
        'silu': (swish, swish_prime, {}),
        'leaky_relu': (leaky_relu, leaky_relu_prime, {'alpha_neg': 0.01}),
        'elu': (elu, elu_prime, {'alpha': 1.0}),
        'selu': (selu, selu_prime, {}),
        'mish': (mish, mish_prime, {})
    }
    
    if activation not in activation_map:
        raise ValueError(f"Activation {activation} not supported for gamma calculation")
    
    act_func, act_prime_func, default_params = activation_map[activation]
    
    def gamma_calculator(a: float, b: float) -> Tuple[float, float, float]:
        if a <= 1e-8:  # Avoid numerical issues
            return 0.0, 0.0, 0.0
        
        m, s = a * b, a
        integral_opts = {'L_init': 10.0, 'n_init': 192, 'tol_rel': 1e-9, 'tol_abs': 1e-12}
        mu, Ef2, Efp, Efp2 = moments_via_integral(m, s, act_func, act_prime_func, 
                                                   integral_opts=integral_opts, **default_params)
        return gammas_from_moments(a, mu, Ef2, Efp, Efp2)
    
    return gamma_calculator

def get_surrogate_gamma_function(activation: str, gh_nodes: int = 160):
    """Get surrogate gamma calculation function for given activation"""
    activation_lower = activation.lower()
    
    def gamma_calculator(a: float, b: float) -> Tuple[float, float, float]:
        if a <= 1e-8:  # Avoid numerical issues
            return 0.0, 0.0, 0.0
        
        m, s = a * b, a
        
        # Use closed-form for ReLU, Gauss-Hermite for others
        if activation_lower == 'relu':
            mu, Ef2, Efp, Efp2 = relu_moments_closed(m, s)
        elif activation_lower == 'gelu':
            mu  = gh_expectation(lambda z: gelu(m + s*z), n_nodes=gh_nodes)
            Ef2 = gh_expectation(lambda z: gelu(m + s*z)**2, n_nodes=gh_nodes)
            Efp = gh_expectation(lambda z: gelu_prime(m + s*z), n_nodes=gh_nodes)
            Efp2= gh_expectation(lambda z: gelu_prime(m + s*z)**2, n_nodes=gh_nodes)
        elif activation_lower == 'tanh':
            mu  = gh_expectation(lambda z: tanh_fn(m + s*z), n_nodes=gh_nodes)
            Ef2 = gh_expectation(lambda z: tanh_fn(m + s*z)**2, n_nodes=gh_nodes)
            Efp = gh_expectation(lambda z: tanh_prime(m + s*z), n_nodes=gh_nodes)
            Efp2= gh_expectation(lambda z: tanh_prime(m + s*z)**2, n_nodes=gh_nodes)
        elif activation_lower == 'sigmoid':
            mu  = gh_expectation(lambda z: sigmoid(m + s*z), n_nodes=gh_nodes)
            Ef2 = gh_expectation(lambda z: sigmoid(m + s*z)**2, n_nodes=gh_nodes)
            Efp = gh_expectation(lambda z: sigmoid_prime(m + s*z), n_nodes=gh_nodes)
            Efp2= gh_expectation(lambda z: sigmoid_prime(m + s*z)**2, n_nodes=gh_nodes)
        elif activation_lower in ['swish', 'silu']:
            mu  = gh_expectation(lambda z: swish(m + s*z), n_nodes=gh_nodes)
            Ef2 = gh_expectation(lambda z: swish(m + s*z)**2, n_nodes=gh_nodes)
            Efp = gh_expectation(lambda z: swish_prime(m + s*z), n_nodes=gh_nodes)
            Efp2= gh_expectation(lambda z: swish_prime(m + s*z)**2, n_nodes=gh_nodes)
        elif activation_lower == 'leaky_relu':
            alpha_neg = 0.01
            mu  = gh_expectation(lambda z: leaky_relu(m + s*z, alpha_neg=alpha_neg), n_nodes=gh_nodes)
            Ef2 = gh_expectation(lambda z: leaky_relu(m + s*z, alpha_neg=alpha_neg)**2, n_nodes=gh_nodes)
            Efp = gh_expectation(lambda z: leaky_relu_prime(m + s*z, alpha_neg=alpha_neg), n_nodes=gh_nodes)
            Efp2= gh_expectation(lambda z: leaky_relu_prime(m + s*z, alpha_neg=alpha_neg)**2, n_nodes=gh_nodes)
        elif activation_lower == 'elu':
            alpha = 1.0
            mu  = gh_expectation(lambda z: elu(m + s*z, alpha=alpha), n_nodes=gh_nodes)
            Ef2 = gh_expectation(lambda z: elu(m + s*z, alpha=alpha)**2, n_nodes=gh_nodes)
            Efp = gh_expectation(lambda z: elu_prime(m + s*z, alpha=alpha), n_nodes=gh_nodes)
            Efp2= gh_expectation(lambda z: elu_prime(m + s*z, alpha=alpha)**2, n_nodes=gh_nodes)
        elif activation_lower == 'selu':
            mu  = gh_expectation(lambda z: selu(m + s*z), n_nodes=gh_nodes)
            Ef2 = gh_expectation(lambda z: selu(m + s*z)**2, n_nodes=gh_nodes)
            Efp = gh_expectation(lambda z: selu_prime(m + s*z), n_nodes=gh_nodes)
            Efp2= gh_expectation(lambda z: selu_prime(m + s*z)**2, n_nodes=gh_nodes)
        elif activation_lower == 'mish':
            mu  = gh_expectation(lambda z: mish(m + s*z), n_nodes=gh_nodes)
            Ef2 = gh_expectation(lambda z: mish(m + s*z)**2, n_nodes=gh_nodes)
            Efp = gh_expectation(lambda z: mish_prime(m + s*z), n_nodes=gh_nodes)
            Efp2= gh_expectation(lambda z: mish_prime(m + s*z)**2, n_nodes=gh_nodes)
        else:
            raise ValueError(f"Activation {activation} not supported for surrogate gamma calculation")
        
        return gammas_from_moments(a, mu, Ef2, Efp, Efp2)
    
    return gamma_calculator

class MLP(nn.Module):
    def __init__(self, input_size=3072, hidden_sizes=[512, 256], num_classes=10, activation='relu'):
        super().__init__()
        self.activations, self.pre_activations = {}, {}
        
        # Activation functions
        act_map = {
            'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(), 'swish': nn.SiLU(), 'silu': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.01), 'elu': nn.ELU(), 'selu': nn.SELU(),
            'mish': nn.Mish(), 'prelu': nn.PReLU()
        }
        self.activation_fn = act_map[activation]
        self.activation_name = activation
        
        # Build layers
        sizes = [input_size] + hidden_sizes + [num_classes]
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        self.activations, self.pre_activations = {}, {}
        
        for i, layer in enumerate(self.layers[:-1]):
            pre_act = layer(x)
            self.pre_activations[i] = pre_act.detach()
            x = self.activation_fn(pre_act)
            self.activations[i] = x.detach()
        
        return self.layers[-1](x)
    
    def activation_derivative_squared(self, z):
        """Compute f'(z)^2 for various activation functions"""
        derivatives = {
            'relu': lambda z: (z > 0).float(),
            'tanh': lambda z: 1 - torch.tanh(z)**2,
            'sigmoid': lambda z: torch.sigmoid(z) * (1 - torch.sigmoid(z)),
            'gelu': lambda z: 0.5 * (1 + torch.tanh((2/torch.pi)**0.5 * (z + 0.044715 * z**3))) + 
                             z * (1 - torch.tanh((2/torch.pi)**0.5 * (z + 0.044715 * z**3))**2) * 
                             (2/torch.pi)**0.5 * (1 + 3 * 0.044715 * z**2),
            'swish': lambda z: torch.sigmoid(z) + z * torch.sigmoid(z) * (1 - torch.sigmoid(z)),
            'silu': lambda z: torch.sigmoid(z) + z * torch.sigmoid(z) * (1 - torch.sigmoid(z)),
            'selu': lambda z: torch.where(z > 0, torch.ones_like(z) * (1.0507**2), (1.0507 * 1.6733 * torch.exp(z))**2),
            'leaky_relu': lambda z: torch.where(z > 0, torch.ones_like(z), 0.01 * torch.ones_like(z)),
            'elu': lambda z: torch.where(z > 0, torch.ones_like(z), torch.exp(z)),
            'mish': lambda z: torch.sigmoid(z) * torch.tanh(torch.nn.functional.softplus(z)),
            'prelu': lambda z: torch.where(z > 0, torch.ones_like(z), 0.25 * torch.ones_like(z))
        }
        return derivatives.get(self.activation_name, derivatives['relu'])(z)**2

def compute_max_correlation(arr):
    """Compute mean of max correlations per row from upper triangular matrix"""
    corr = np.corrcoef(arr.T)
    upper_tri = np.triu(corr, k=1)
    max_corrs = [np.nanmax(upper_tri[i, i+1:]) for i in range(len(upper_tri)-1)]
    return np.nanmean(max_corrs) if max_corrs else 0.0

def compute_duplicated_units_fraction(arr, threshold=0.95):
    """Compute fraction of units that have high correlation (>threshold) with other units"""
    corr = np.corrcoef(arr.T)  # arr.T so we get correlation between columns (units)
    n_units = corr.shape[0]
    if n_units <= 1:
        return 0.0
    
    duplicated_count = 0
    for i in range(n_units):
        # Look at row i, upper triangular part (j > i)
        upper_tri_row = corr[i, i+1:]
        if len(upper_tri_row) > 0 and np.any(np.abs(upper_tri_row) > threshold):
            duplicated_count += 1
    
    return duplicated_count / n_units

def calculate_metrics(model, trainloader, device, gamma_mode='exact', gh_nodes=160):
    """Calculate all metrics including gamma values for each layer"""
    model.eval()
    
    # Get gamma calculator for this activation
    try:
        if gamma_mode == 'exact':
            gamma_calc = get_exact_gamma_function(model.activation_name)
        elif gamma_mode == 'surrogate':
            gamma_calc = get_surrogate_gamma_function(model.activation_name, gh_nodes=gh_nodes)
        else:
            raise ValueError(f"Invalid gamma_mode: {gamma_mode}. Use 'exact' or 'surrogate'")
    except ValueError as e:
        print(f"Warning: {e}")
        gamma_calc = None
    
    with torch.no_grad():
        images, _ = next(iter(trainloader))
        
        # Handle GPU dataloader case where data is already on device
        if not isinstance(trainloader, GPUDataLoader):
            images = images.to(device)
        
        _ = model(images)
        
        metrics = {}
        for layer_idx in model.pre_activations:
            pre_act = model.pre_activations[layer_idx]
            post_act = model.activations[layer_idx]
            
            # Convert to numpy
            pre_np, post_np = pre_act.cpu().numpy(), post_act.cpu().numpy()
            
            # Calculate derivative squared
            deriv_sq = model.activation_derivative_squared(pre_act).cpu().numpy()
            
            # Calculate normalized means (Sharpe-like ratios): mean/std per channel
            def calc_normalized_stats(arr):
                channel_means = np.mean(arr, axis=0)  # Mean per channel
                channel_stds = np.std(arr, axis=0)    # Std per channel
                # Avoid division by zero: set ratio to 0 where std is very small
                normalized = np.where(channel_stds > 1e-8, channel_means / channel_stds, 0)
                return np.mean(normalized), np.std(normalized)
            
            pre_norm_mean, pre_norm_std = calc_normalized_stats(pre_np)
            
            # Calculate gamma values if available
            gamma0_mean = gamma1_mean = gamma2_mean = 0.0
            if gamma_calc is not None:
                # For each channel, compute gamma using std as 'a' and mean/std as 'b'
                channel_means = np.mean(pre_np, axis=0)
                channel_stds = np.std(pre_np, axis=0)
                
                gamma0_vals, gamma1_vals, gamma2_vals = [], [], []
                for i in range(len(channel_stds)):
                    std_val = float(channel_stds[i])
                    mean_std_ratio = float(channel_means[i] / channel_stds[i]) if channel_stds[i] > 1e-8 else 0.0
                    
                    if std_val > 1e-8:  # Only compute if std is reasonable
                        g0, g1, g2 = gamma_calc(std_val, mean_std_ratio)
                        gamma0_vals.append(g0)
                        gamma1_vals.append(g1)
                        gamma2_vals.append(g2)
                
                # Take mean of gamma values across channels
                gamma0_mean = np.mean(gamma0_vals) if gamma0_vals else 0.0
                gamma1_mean = np.mean(gamma1_vals) if gamma1_vals else 0.0
                gamma2_mean = np.mean(gamma2_vals) if gamma2_vals else 0.0
            
            metrics[layer_idx] = {
                'pre_corr': compute_max_correlation(pre_np),
                'post_corr': compute_max_correlation(post_np),
                'pre_norm_mean': pre_norm_mean,
                'pre_norm_std': pre_norm_std,
                'pre_std': np.mean(np.std(pre_np, axis=0)),
                'post_std': np.mean(np.std(post_np, axis=0)),
                'deriv_mean': np.mean(deriv_sq),
                'deriv_std': np.std(np.mean(deriv_sq, axis=0)),
                'gamma0_mean': gamma0_mean,
                'gamma1_mean': gamma1_mean,
                'gamma2_mean': gamma2_mean,
                'duplicated_fraction': compute_duplicated_units_fraction(pre_np, threshold=0.95)
            }
    return metrics

def plot_metrics(metrics_log, config):
    """Create 8-panel plot with separate gamma plots and duplicated units"""
    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    axes = axes.flatten()
    
    # Extract data
    iterations = [m[0] for m in metrics_log]
    num_layers = len(metrics_log[0][1])
    # Progression color scheme: cool (blue) to warm (red) through the network layers
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, num_layers))
    
    # Plot configurations: (metric_pairs, title, ylabel, log_scale)
    plots = [
        (('pre_corr', 'post_corr'), 'Correlations', 'Max Correlation', False),
        (('pre_norm_mean', 'pre_norm_std'), 'Normalized Means (Mean/Std)', 'Mean/Std Ratio', False),
        (('pre_std', 'post_std'), 'Channel Stds', 'Std', True),
        (('deriv_mean', 'deriv_std'), f"f'(z)² - {config['activation'].upper()}", "f'(z)²", True),
        (('gamma0_mean',), 'Gamma 0 (γ₀)', 'γ₀ Value', False),
        (('gamma1_mean',), 'Gamma 1 (γ₁)', 'γ₁ Value', False),
        (('gamma2_mean',), 'Gamma 2 (γ₂)', 'γ₂ Value', False),
        (('duplicated_fraction',), 'Duplicated Units Fraction', 'Fraction', False)
    ]
    
    for ax_idx, (metrics_pair, title, ylabel, log_scale) in enumerate(plots):
        ax = axes[ax_idx]
        
        for layer_idx in range(num_layers):
            color = colors[layer_idx]
            
            # Extract metric data for this layer
            metric1_data = [metrics_log[i][1][layer_idx][metrics_pair[0]] for i in range(len(metrics_log))]
            
            # Plot based on panel type
            if ax_idx == 0:  # Correlations: pre/post activation
                metric2_data = [metrics_log[i][1][layer_idx][metrics_pair[1]] for i in range(len(metrics_log))]
                ax.plot(iterations, metric1_data, color=color, linestyle='-', alpha=0.8, linewidth=2, label=f'z{layer_idx+1}')
                ax.plot(iterations, metric2_data, color=color, linestyle='--', alpha=0.8, linewidth=2, label=f'a{layer_idx+1}')
            elif ax_idx == 1:  # Normalized means: mean/std of ratios  
                metric2_data = [metrics_log[i][1][layer_idx][metrics_pair[1]] for i in range(len(metrics_log))]
                ax.plot(iterations, metric1_data, color=color, linestyle='-', alpha=0.8, linewidth=2, label=f'Mean z{layer_idx+1}')
            elif ax_idx == 2:  # Channel stds: pre/post activation
                metric2_data = [metrics_log[i][1][layer_idx][metrics_pair[1]] for i in range(len(metrics_log))]
                ax.plot(iterations, metric1_data, color=color, linestyle='-', alpha=0.8, linewidth=2, label=f'z{layer_idx+1}')
                ax.plot(iterations, metric2_data, color=color, linestyle='--', alpha=0.8, linewidth=2, label=f'a{layer_idx+1}')
            elif ax_idx == 3:  # Derivatives: mean/std
                metric2_data = [metrics_log[i][1][layer_idx][metrics_pair[1]] for i in range(len(metrics_log))]
                ax.plot(iterations, metric1_data, color=color, linestyle='-', alpha=0.8, linewidth=2, label=f'Mean z{layer_idx+1}')
            elif ax_idx in [4, 5, 6]:  # Gamma 0, 1, 2
                ax.plot(iterations, metric1_data, color=color, linestyle='-', alpha=0.8, linewidth=2, label=f'L{layer_idx+1}')
            elif ax_idx == 7:  # Duplicated units fraction
                ax.plot(iterations, metric1_data, color=color, linestyle='-', alpha=0.8, linewidth=2, label=f'L{layer_idx+1}')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if ax_idx > 0:
            ax.set_xscale('log')
        
        if log_scale:
            ax.set_yscale('log')
        
        # Add legend to each plot for clarity
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(config['save_path'], dpi=config['dpi'], bbox_inches='tight')
    plt.show()

def train_model_unified(model, dataloader, criterion, optimizer, config):
    """Unified training function that works with both standard and GPU loaders"""
    model.train()
    
    start_time = time.time()
    iteration_count = 0
    losses = []
    metrics_log = [(0, calculate_metrics(model, dataloader, config['device'], 
                                        gamma_mode=config.get('gamma_mode', 'exact'),
                                        gh_nodes=config.get('gh_nodes', 160)))]  # Initial metrics
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_losses = []
        
        for inputs, labels in dataloader:
            # Move to device only if using standard DataLoader
            if not config.get('use_gpu_loader', False):
                inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            losses.append(loss.item())
            iteration_count += 1
            
            # Log metrics periodically
            if iteration_count % config['calc_every_n_iterations'] == 0:
                metrics = calculate_metrics(model, dataloader, config['device'],
                                          gamma_mode=config.get('gamma_mode', 'exact'),
                                          gh_nodes=config.get('gh_nodes', 160))
                metrics_log.append((iteration_count, metrics))
                print(f"Epoch {epoch+1}/{config['epochs']}, Iter {iteration_count}, Loss: {loss:.4f}")
        
        # End of epoch metrics and logging
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        final_metrics = calculate_metrics(model, dataloader, config['device'],
                                        gamma_mode=config.get('gamma_mode', 'exact'),
                                        gh_nodes=config.get('gh_nodes', 160))
        metrics_log.append((iteration_count, final_metrics))
        print(f"Epoch {epoch+1} completed, Avg Loss: {avg_epoch_loss:.4f}")
    
    train_time = time.time() - start_time
    return train_time, metrics_log

def train_and_analyze(config):
    # Setup
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    gamma_mode = config.get('gamma_mode', 'exact')
    use_gpu_loader = config.get('use_gpu_loader', False)
    
    loader_type = "GPU DataLoader" if use_gpu_loader else "Standard DataLoader"
    print(f"Training {config['activation']} network on {config['dataset'].upper()} using {device}")
    print(f"Using {gamma_mode} gamma calculation method")
    print(f"Using {loader_type}")
    
    # Dataset-specific configuration
    if config['dataset'] == 'mnist':
        input_size = 784  # 28*28*1
        dataset_class = torchvision.datasets.MNIST
        normalize_mean = (0.1307,)
        normalize_std = (0.3081,)
    elif config['dataset'] == 'cifar10':
        input_size = 3072  # 32*32*3
        dataset_class = torchvision.datasets.CIFAR10
        normalize_mean = (0.5, 0.5, 0.5)
        normalize_std = (0.5, 0.5, 0.5)
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    trainset = dataset_class(root='./data', train=True, download=True, transform=transform)
    
    # Use the dataloader factory function
    trainloader = get_dataloader(trainset, config, is_train=True)
    
    # Model setup
    model = MLP(input_size=input_size, hidden_sizes=config['hidden_sizes'], activation=config['activation']).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                               lr=config['learning_rate'],
                               weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                              lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'], 
                              momentum=config['momentum'])
    else:
        raise ValueError(f"optimizer {config['optimizer']} not found")
    
    # Training
    train_time, metrics_log = train_model_unified(model, trainloader, criterion, optimizer, config)
    
    # Plotting and final summary
    plot_metrics(metrics_log, config)
    
    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"Samples used: {len(trainloader.data) if use_gpu_loader else len(trainset)}")
    print(f"\nFinal metrics for {len(config['hidden_sizes'])}-layer {config['activation']} network on {config['dataset'].upper()}:")
    
    final = metrics_log[-1][1]
    for i in range(len(config['hidden_sizes'])):
        m = final[i]
        print(f"Layer {i+1}: Corr(z/a)={m['pre_corr']:.3f}/{m['post_corr']:.3f}, "
              f"Norm(z)={m['pre_norm_mean']:.3f}±{m['pre_norm_std']:.3f}, "
              f"f'(z)²={m['deriv_mean']:.3f}±{m['deriv_std']:.3f}, "
              f"γ=(γ0:{m['gamma0_mean']:.3f}, γ1:{m['gamma1_mean']:.3f}, γ2:{m['gamma2_mean']:.3f}), "
              f"Dup={m['duplicated_fraction']:.3f}")

if __name__ == "__main__":
    config = {
        'dataset': 'mnist',  # Switch between 'mnist' and 'cifar10'
        'hidden_sizes': [1024, ]*4,
        'activation': 'gelu',
        'epochs': 100,
        'batch_size': 512,
        'optimizer': 'sgd',
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.001,
        'calc_every_n_iterations': 100,
        'device': 'cuda:3',
        'figsize': (24, 10),
        'dpi': 150,
        'save_path': 'correlation_analysis_with_gamma.png',
        'gamma_mode': 'surrogate',  # 'exact' or 'surrogate'
        'gh_nodes': 160,  # Number of Gauss-Hermite nodes for surrogate mode
        
        # GPU DataLoader specific settings
        'use_gpu_loader': True,  # Set to True to use GPU preloaded data loader
        'max_gpu_samples': None,  # Limit samples if GPU memory is limited (None = use all)
        'num_workers': 2  # Only used for standard DataLoader
    }
    
    # Run training and analysis
    train_and_analyze(config)
    
    # Optional: Clear GPU memory between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()