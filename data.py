import torch 
import math

def generate_fixed_effects(n, p, fixed_type='linear', device='cpu', dtype=torch.bfloat16):
    if fixed_type == 'linear':
        X = torch.rand(n, p, device=device, dtype=dtype)
        F = 1 + torch.sum(X, dim=1, dtype=dtype)
    elif fixed_type == 'friedman3':
        X1 = torch.rand(n, device=device, dtype=dtype)
        X2 = torch.rand(n, device=device, dtype=dtype)
        X3 = torch.rand(n, device=device, dtype=dtype)
        X4 = torch.rand(n, device=device, dtype=dtype)
        X = torch.stack((X1, X2, X3, X4), dim=1)
        F = torch.atan((X2 * X3 - 1 / (X2 * X4)) / X1) # friedman3 function
    elif fixed_type == 'hajjem':
        X = torch.randn(n, 9, device=device, dtype=dtype)
        F = 2 * X[:, 0] + X[:, 1]**2 + 4 * (X[:, 2] > 0).to(dtype) + 2 * torch.log(torch.abs(X[:, 0]))  # hajjem function
    else:
        raise ValueError("Unsupported fixed effect type. Choose from 'linear', 'friedman3', or 'hajjem'.")
    F = (1 / torch.std(F)) * F  # variance of F is approximately 1
    return X, F

def generate_random_effects(n, m, random_type='grouped', gp_params={}, device='cpu', dtype=torch.bfloat16):
    if random_type == 'grouped':
        # grouped random effects
        Z = torch.zeros(n, m, device=device, dtype=dtype)
        group_indices = torch.randint(0, m, (n,), device=device, dtype=torch.int32)
        Z[torch.arange(n, device=device), group_indices] = 1
        b = torch.randn(m, device=device, dtype=dtype)
    elif random_type == 'spatial':
        # spatial Gaussian process random effects
        sigma2 = gp_params.get('sigma2', 1)
        rho = gp_params.get('rho', 0.1)
        locations = torch.rand(m, 2, device=device, dtype=dtype)  # 2D locations for spatial data
        distance_matrix = torch.cdist(locations, locations)
        covariance_matrix = sigma2**2 * torch.exp(-distance_matrix / rho)
        b = torch.distributions.MultivariateNormal(torch.zeros(m, device=device, dtype=dtype), covariance_matrix).sample()
        Z = torch.eye(m, device=device, dtype=dtype)[torch.randint(0, m, (n,), device=device, dtype=torch.int32)]
    else:
        raise ValueError("Unsupported random effect type. Choose from 'grouped' or 'spatial'.")

    return Z, b

def generate_data(mode='train', fixed_type='linear', random_type='grouped', gp_params={}, device='cpu', dtype=torch.float32):
    if random_type == 'grouped':
        if mode == 'train':
            n = 1000
            m = 100
        else:
            n = 500
            m = 50
    else:
        n = 500
        m = 50
    
    if fixed_type == 'linear':
        p = 2
    elif fixed_type == 'friedman3':
        p = 4
    elif fixed_type == 'hajjem':
        p = 9
            
    # fixed effects
    X, F = generate_fixed_effects(n, p, fixed_type, device=device, dtype=dtype)

    # random effects
    m = n // 10
    Z, b = generate_random_effects(n, m, random_type, gp_params, device=device, dtype=dtype)

    # noise
    epsilon = torch.randn(n, device=device, dtype=dtype)

    # response
    y = F + Z @ b + epsilon

    return X, F, Z, b, y, m