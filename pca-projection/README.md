# 📐 PCA Projection

> Principal Component Analysis from scratch — implemented with numerically stable eigendecomposition.

---

## 📋 Table of Contents

- [Overview](#overview)
- [How PCA Works](#how-pca-works)
- [Algorithm](#algorithm)
- [Mathematical Background](#mathematical-background)
- [Implementation](#implementation)
- [Why `eigh` Instead of Power Iteration](#why-eigh-instead-of-power-iteration)
- [Examples](#examples)
- [Edge Cases](#edge-cases)
- [Complexity](#complexity)

---

## Overview

Principal Component Analysis (PCA) finds the directions of **maximum variance** in high-dimensional data and projects it onto a lower-dimensional subspace. Given a data matrix `X` of shape `(n, d)` and a target number of components `k`, this implementation returns the `(n, k)` projection onto the top-k principal components.

```python
X = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
k = 1

result = pca_projection(X, k)
# → [[-2.828], [-1.414], [0.0], [1.414], [2.828]]
```

---

## How PCA Works

```
Original Data (d dimensions)
        │
        ▼
  ┌─────────────┐
  │  1. Center  │  Subtract column means → X_centered
  └─────────────┘
        │
        ▼
  ┌──────────────────────┐
  │  2. Covariance Matrix │  C = Xc.T @ Xc / (n-1)
  └──────────────────────┘
        │
        ▼
  ┌──────────────────────────┐
  │  3. Eigendecomposition   │  C = V Λ V.T
  └──────────────────────────┘
        │
        ▼
  ┌─────────────────────────────┐
  │  4. Select Top-k Eigenvectors│  W = V[:, :k]
  └─────────────────────────────┘
        │
        ▼
  ┌──────────────────────────┐
  │  5. Project              │  X_proj = X_centered @ W
  └──────────────────────────┘
        │
        ▼
  Projected Data (k dimensions)
```

---

## Algorithm

### Step 1 — Center the Data

Subtract the mean of each feature column so the data is zero-centered:

```
X_centered = X - mean(X, axis=0)
```

This is required because PCA measures variance *around the mean*, not around the origin.

### Step 2 — Compute the Covariance Matrix

Use the **sample covariance** (divide by `n-1`, not `n`):

```
C = (X_centered.T @ X_centered) / (n - 1)
```

`C` is a `(d, d)` symmetric positive semi-definite matrix. Each entry `C[i, j]` represents how features `i` and `j` co-vary across samples.

### Step 3 — Eigendecomposition

Decompose `C` into its eigenvectors and eigenvalues:

```
C · v = λ · v
```

Each eigenvector `v` is a **principal component direction**. Its corresponding eigenvalue `λ` tells you how much variance lies along that direction.

Sort eigenvectors by their eigenvalues in **descending order** — the first eigenvector captures the most variance.

### Step 4 — Select Top-k Eigenvectors

Form the projection matrix `W` by taking the first `k` eigenvectors as columns:

```
W  →  shape (d, k)
```

### Step 5 — Project the Data

```
X_proj = X_centered @ W   →  shape (n, k)
```

---

## Mathematical Background

### Covariance Matrix

$$C = \frac{1}{n-1} X_c^T X_c$$

### Eigenvalue Equation

$$C \mathbf{v} = \lambda \mathbf{v}$$

### Projection

$$X_{\text{proj}} = X_c \cdot W$$

where $W \in \mathbb{R}^{d \times k}$ whose columns are the top-$k$ eigenvectors.

### Variance Retained

The fraction of total variance retained by the top-k components is:

$$\text{Variance Retained} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

---

## Implementation

```python
import numpy as np

def pca_projection(X, k):
    X = np.array(X, dtype=float)
    n, d = X.shape

    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Compute sample covariance matrix (d x d)
    C = (X_centered.T @ X_centered) / (n - 1)

    # Step 3: Eigendecomposition of symmetric matrix
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Step 4: Sort by descending eigenvalue, take top-k columns
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    W = eigenvectors[:, :k]   # shape: (d, k)

    # Sign canonicalization for determinism:
    # flip each eigenvector so its largest-magnitude component is positive
    for i in range(k):
        max_idx = np.argmax(np.abs(W[:, i]))
        if W[max_idx, i] < 0:
            W[:, i] = -W[:, i]

    # Step 5: Project centered data onto top-k principal components
    X_proj = X_centered @ W   # shape: (n, k)
    return X_proj.tolist()
```

---

## Why `eigh` Instead of Power Iteration

The natural first approach to finding eigenvectors is **power iteration with deflation**. Here's why it fails for this problem and why `numpy.linalg.eigh` is the correct tool.

### The Problem with Power Iteration + Deflation

Power iteration works by repeatedly multiplying a random vector by the matrix until it converges to the dominant eigenvector. After finding one component, **deflation** removes its contribution:

```
C_temp = C - λ * outer(v, v)
```

Then repeat for the next component. This breaks down in two ways:

**1. Degenerate / rank-deficient covariance matrices**

When data lies in a subspace (e.g., all points on a line in 2D), the covariance matrix has **zero eigenvalues**. After deflating the dominant component, `C_temp` should be exactly zero — but floating point errors leave numerical noise. Power iteration then "converges" to a random garbage direction instead of the correct zero vector.

```
Input: [[1,2],[3,4],[5,6],[7,8]]  (all points on y = x + 1)

Power iteration output:  [[-4.24, -4.24], [-1.41, -1.41], ...]  ❌
Correct output:          [[-4.24,  0.00], [-1.41,  0.00], ...]  ✓
```

**2. Sign ambiguity accumulates across components**

Power iteration converges to `±v` depending on the random seed. Even with sign canonicalization, the deflation step `C -= λ * outer(v, v)` must use exactly the same vector that was just found. Any accumulated sign flip cascades into incorrect projections for later components.

### Why `eigh` is the Right Choice

`numpy.linalg.eigh` is designed specifically for **symmetric/Hermitian matrices** (which covariance matrices always are). It uses LAPACK's divide-and-conquer algorithm, which:

| Property | Power Iteration + Deflation | `np.linalg.eigh` |
|---|---|---|
| Zero eigenvalues | ❌ Deflated residual becomes noise | ✅ Returns exact zero |
| Numerical stability | ❌ Errors accumulate per component | ✅ Stable (LAPACK `dsyevd`) |
| All components at once | ❌ Sequential, errors compound | ✅ One-shot decomposition |
| Symmetric guarantee used | ❌ No | ✅ Yes — enforces real eigenvalues |
| Time complexity | O(k · d² · iters) | O(d³) one-shot |

The covariance matrix is symmetric by construction (`C = C.T`), so `eigh` always applies and is strictly superior to general-purpose eigensolvers (`eig`) or iterative methods for this use case.

---

## Examples

### Example 1 — All Variance in One Dimension

```python
X = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
k = 1

pca_projection(X, k)
# → [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
```

All variance is in the first feature. The second feature is constant (zero variance). PC1 = `[1, 0]`. After centering (mean = `[3, 0]`), projecting onto PC1 gives the centered first coordinates.

---

### Example 2 — Data Along the Diagonal

```python
X = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
k = 1

pca_projection(X, k)
# → [[-2.828], [-1.414], [0.0], [1.414], [2.828]]
```

The data lies along `y = x`. The principal component is `[1/√2, 1/√2]`. Projecting the centered data onto this direction scales distances by `√2`.

---

### Example 3 — Rank-Deficient Matrix (Critical Edge Case)

```python
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
k = 2

pca_projection(X, k)
# → [[-4.2426, 0.0], [-1.4142, 0.0], [1.4142, 0.0], [4.2426, 0.0]]
```

All points lie on `y = x + 1` — a 1D subspace embedded in 2D. The covariance matrix has rank 1, so PC2 has eigenvalue ≈ 0 and the second projection column is all zeros. Power iteration fails here; `eigh` handles it correctly.

---

## Edge Cases

| Case | Behaviour |
|---|---|
| All variance in one feature | Remaining PCs correctly project to zero |
| Data on a line (rank-1 covariance) | Zero eigenvalue → zero projection column |
| `k = d` | Full reconstruction (no dimensionality reduction) |
| `k = 1` | Single principal component returned |
| Perfectly correlated features | Handled via zero eigenvalues |

---

## Complexity

| Step | Time | Space |
|---|---|---|
| Center data | O(n · d) | O(n · d) |
| Covariance matrix | O(n · d²) | O(d²) |
| Eigendecomposition (`eigh`) | O(d³) | O(d²) |
| Projection | O(n · d · k) | O(n · k) |
| **Total** | **O(n · d² + d³)** | **O(n · d)** |

For typical ML use cases where `n >> d`, the dominant cost is computing the covariance matrix at O(n · d²).

---

## Constraints

- `X` has at least 2 rows
- `k ≤ d` (cannot project to more dimensions than the input)
- The top-k eigenvalues are distinct (no ties)
- Returns an `n × k` list of floats
- Time limit: 300 ms
