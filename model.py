import torch
import numpy as np

class Node:
    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None, device='cpu', dtype=torch.float32):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.device = device
        self.dtype = dtype 

class RegressionTree:
    def __init__(self, max_depth=2, min_samples_leaf=1, device='cpu', dtype=torch.float32):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.device = device
        self.dtype = dtype
        self.root = None

    def fit(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=self.device, dtype=self.dtype)
        else:
            X = X.to(device=self.device, dtype=self.dtype)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device, dtype=self.dtype)
        else:
            y = y.to(device=self.device, dtype=self.dtype)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Stopping criteria
        if depth >= self.max_depth or len(y) < self.min_samples_leaf:
            leaf = Node(value=y.mean(), device=self.device, dtype=self.dtype)
            return leaf

        # Find the best split
        best_feature, best_threshold, best_loss = None, None, float('inf')
        n_features = X.size(1)

        for feature in range(n_features):
            # Get unique sorted split points
            split_points = torch.unique(X[:, feature])
            # Consider splits between samples
            split_points = (split_points[:-1] + split_points[1:]) / 2
            for threshold in split_points:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                # Avoid splits that leave empty branches
                if torch.sum(left_indices) < self.min_samples_leaf or torch.sum(right_indices) < self.min_samples_leaf:
                    continue

                # Calculate loss
                y_left = y[left_indices]
                y_right = y[right_indices]
                current_loss = (torch.sum((y_left - y_left.mean())**2) + torch.sum((y_right - y_right.mean())**2))

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_feature = feature
                    best_threshold = threshold

        # If no split found, make it a leaf node
        if best_feature is None:
            leaf = Node(value=y.mean(), device=self.device, dtype=self.dtype)
            return leaf

        # Build left and right branches
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_branch = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_branch = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Create internal node
        node = Node(value=None, 
                    feature=best_feature, 
                    threshold=best_threshold, 
                    left=left_branch, 
                    right=right_branch, 
                    device=self.device, 
                    dtype=self.dtype)
        return node

    def predict(self, X):
        X = X.to(device=self.device, dtype=self.dtype)
        predictions = torch.zeros(X.shape[0], device=self.device, dtype=self.dtype)
        for i in range(X.shape[0]):
            sample = X[i]
            node = self.root
            while node.feature is not None:
                if sample[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions[i] = node.value
        return predictions

class MetaLearner:
    def __init__(self, y, X, Psi_inv, lr, device='cpu', dtype=torch.float32):
        print("----- Initialize Meta Learner -----")
        self.y = y
        self.X = X
        self.lr = lr
        self.device = device
        self.dtype = dtype
        one = torch.ones_like(y, device=device, dtype=dtype)
        c = (one.T @ Psi_inv @ y) / (one.T @ Psi_inv @ one) 
        self.c = c * torch.ones(X.shape[-1], dtype=dtype, device=device)
        self.F = self.X @ self.c
        self.base_learners = []

    def update(self, f):
        self.base_learners.append(f)
        self.F += f.predict(self.X)
        return self.F
    
    def predict(self, X_new):
        y_pred = X_new @ self.c
        for f in self.base_learners:
            y_pred += self.lr * f.predict(X_new)
        return y_pred

def Boosting(y, F, Psi_inv, X, device='cpu', dtype=torch.float32):
    F_gradient = Psi_inv @ (y - F)
    f = RegressionTree(max_depth=2, min_samples_leaf=1, device=device, dtype=dtype)
    f.fit(X, F_gradient)
    return f
