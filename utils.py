import torch
import numpy as np
import matplotlib.pyplot as plt

def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("----- Device Set to CUDA -----")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("----- Device Set to MPS -----")
    else:
        device = torch.device('cpu')
        print("----- Device Set to CPU -----")
    return device

def compute_Psi(Z, L, sigma2, n, device='cpu', dtype=torch.float32):
  return Z @ (L @ L.T) @ Z.T + sigma2 * torch.eye(n, device=device, dtype=dtype)

def negative_log_likelihood(Z, L, sigma2, n, y, F, device='cpu', dtype=torch.float32):
    Psi = compute_Psi(Z, L, sigma2, n, device, dtype)
    Psi_inv = torch.inverse(Psi)
    Psi_logdet = torch.logdet(Psi)
    residual = y - F
    return 0.5 * (n * torch.log(2 * torch.tensor(np.pi, dtype=dtype, device=device)) + Psi_logdet + residual.T @ Psi_inv @ residual)

def plot(solution_path):
    name_map = {
        "nll": "Negative Log Likelihood", 
        "sigma2": "Error Variance", 
        "test_mse": "RMSE over Test Set"
    }
    num_metrics = len(solution_path)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5), sharey=False)

    if num_metrics == 1:
        axes = [axes]

    for i, (metric, values) in enumerate(solution_path.items()):
        if not isinstance(values, list) or not all(isinstance(v, (int, float)) for v in values):
            print(f"Skipping metric '{metric}' as it does not contain a valid list of floats.")
            continue

        ax = axes[i]
        ax.plot(values, label=metric, marker='o' if len(values) < 20 else None)
        ax.set_title(metric)
        ax.set_xlabel("Iteration")
        ax.grid(True)

        min_value = min(values)
        max_value = max(values)
        padding = (max_value - min_value) * 0.1  # add 10% padding to the y-axis
        ax.set_ylim(min_value - padding, max_value + padding)

    fig.text(0.04, 0, "Metric Value", va="center", rotation="vertical")
    plt.tight_layout()
   