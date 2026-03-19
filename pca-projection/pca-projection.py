import numpy as np

def pca_projection(X, k):
    X = np.array(X, dtype=float)
    n, d = X.shape
    
    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Step 2: Compute sample covariance matrix
    C = (X_centered.T @ X_centered) / (n - 1)
    
    # Step 3: Eigen-decomposition of symmetric matrix
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Step 4: Sort by descending eigenvalue, take top-k
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    W = eigenvectors[:, :k]  # shape: (d, k)
    
    # Step 5: Sign canonicalization for determinism
    for i in range(k):
        max_idx = np.argmax(np.abs(W[:, i]))
        if W[max_idx, i] < 0:
            W[:, i] = -W[:, i]
    
    # Step 6: Project centered data
    X_proj = X_centered @ W
    return X_proj.tolist()