# 📐 PCA Full Pipeline — Step-by-Step Math

> A complete walkthrough of Principal Component Analysis: from raw data to final projection, with every calculation shown explicitly.

---

## 📋 Table of Contents

- [Key Insight](#key-insight)
- [Step 0 — Given Data Matrix](#step-0--given-data-matrix)
- [Step 1 — Compute Column Means](#step-1--compute-column-means)
- [Step 2 — Center the Data](#step-2--center-the-data)
- [Step 3 — Covariance Matrix](#step-3--covariance-matrix)
- [Step 4 — Eigenvalues (Characteristic Equation)](#step-4--eigenvalues-characteristic-equation)
- [Step 5 — Eigenvector for Largest Eigenvalue](#step-5--eigenvector-for-largest-eigenvalue-λ--45)
- [Step 6 — Normalize the Eigenvector](#step-6--normalize-the-eigenvector)
- [Step 7 — Project the Data](#step-7--project-the-data)
- [Final Summary](#final-summary)
- [Full Implementation](#full-implementation)

---

## 💡 Key Insight

```
PCA = find the best direction → project data onto it
```

> **Best direction** = the axis along which the data varies the most.
> **Project** = compress each point down to its coordinate along that axis.

---

## Step 0 — Given Data Matrix

We start with a matrix $X$ of $n = 5$ samples and $d = 2$ features:

$$X = \begin{bmatrix} 2 & 0 \\ 
0 & 1 \\ 
3 & 1 \\ 
4 & 2 \\ 
5 & 3 \\
\end{bmatrix} \quad (n=5,\ d=2)$$

| Sample | Feature 1 | Feature 2 |
|--------|-----------|-----------|
| 1      | 2         | 0         |
| 2      | 0         | 1         |
| 3      | 3         | 1         |
| 4      | 4         | 2         |
| 5      | 5         | 3         |

---

## Step 1 — Compute Column Means

Average each feature column independently:

$$\mu = \left[\frac{2+0+3+4+5}{5},\ \frac{0+1+1+2+3}{5}\right]$$

$$\boxed{\mu = [2.8,\ 1.4]}$$

---

## Step 2 — Center the Data

Subtract the mean from every row so the data is zero-centered around the origin:

$$X_c = X - \mu$$

$$X_c = \begin{bmatrix} 2-2.8 & 0-1.4 \\ 
0-2.8 & 1-1.4 \\ 
3-2.8 & 1-1.4 \\ 
4-2.8 & 2-1.4 \\ 
5-2.8 & 3-1.4 \end{bmatrix} = \begin{bmatrix} -0.8 & -1.4 \\ 
-2.8 & -0.4 \\ 
0.2 & -0.4 \\ 
1.2 & 0.6 \\ 
2.2 & 1.6 \end{bmatrix}$$

> **Why center?** PCA measures variance *around the mean*. Without centering, the origin biases the covariance calculation.

---

## Step 3 — Covariance Matrix

The covariance matrix captures how much features vary together:

$$C = \frac{1}{n-1}\ X_c^T X_c$$

First, write out $X_c^T$:

$$X_c^T = \begin{bmatrix} -0.8 & -2.8 & 0.2 & 1.2 & 2.2 \\ 
-1.4 & -0.4 & -0.4 & 0.6 & 1.6 \end{bmatrix}$$

Now compute each entry of $X_c^T X_c$ explicitly:

**Entry (1,1) — variance of Feature 1:**

$$(-0.8)^2 + (-2.8)^2 + (0.2)^2 + (1.2)^2 + (2.2)^2$$
$$= 0.64 + 7.84 + 0.04 + 1.44 + 4.84 = 14.8$$

**Entry (1,2) = (2,1) — covariance between features:**

$$(-0.8)(-1.4) + (-2.8)(-0.4) + (0.2)(-0.4) + (1.2)(0.6) + (2.2)(1.6)$$
$$= 1.12 + 1.12 - 0.08 + 0.72 + 3.52 = 6.40$$

**Entry (2,2) — variance of Feature 2:**

$$(-1.4)^2 + (-0.4)^2 + (-0.4)^2 + (0.6)^2 + (1.6)^2$$
$$= 1.96 + 0.16 + 0.16 + 0.36 + 2.56 = 5.20$$

So:

$$X_c^T X_c = \begin{bmatrix} 14.8 & 6.4 \\ 
6.4 & 5.2 \end{bmatrix}$$

Divide by $n - 1 = 4$:

$$\boxed{C = \begin{bmatrix} 3.7 & 1.6 \\ 
1.6 & 1.3 \end{bmatrix}}$$

> **Interpretation:** The diagonal entries (3.7 and 1.3) are the individual variances of each feature. The off-diagonal entry (1.6) shows they are positively correlated — when one goes up, so does the other.

---

## Step 4 — Eigenvalues (Characteristic Equation)

Eigenvalues tell us **how much variance** lies along each principal direction. We solve:

$$\det(C - \lambda I) = 0$$

$$\begin{vmatrix} 3.7 - \lambda & 1.6 \\ 
1.6 & 1.3 - \lambda \end{vmatrix} = 0$$

Expand the determinant:

$$(3.7 - \lambda)(1.3 - \lambda) - (1.6)^2 = 0$$

$$(3.7 \cdot 1.3) - (3.7 + 1.3)\lambda + \lambda^2 - 2.56 = 0$$

$$4.81 - 5\lambda + \lambda^2 - 2.56 = 0$$

$$\lambda^2 - 5\lambda + 2.25 = 0$$

Apply the quadratic formula:

$$\lambda = \frac{5 \pm \sqrt{25 - 9}}{2} = \frac{5 \pm \sqrt{16}}{2} = \frac{5 \pm 4}{2}$$

$$\boxed{\lambda_1 = 4.5 \qquad \lambda_2 = 0.5}$$

**Variance explained:**

| Component | Eigenvalue | Variance Retained |
|-----------|------------|-------------------|
| PC1       | 4.5        | 4.5 / 5.0 = **90%** |
| PC2       | 0.5        | 0.5 / 5.0 = **10%** |
| Total     | 5.0        | **100%**          |

> PC1 alone captures 90% of the total variance — projecting to 1D loses only 10% of information.

---

## Step 5 — Eigenvector for Largest Eigenvalue (λ = 4.5)

The eigenvector for $\lambda_1 = 4.5$ gives the **direction of maximum variance**. Solve:

$$(C - 4.5I)\,\mathbf{v} = \mathbf{0}$$

$$\begin{bmatrix} 3.7 - 4.5 & 1.6 \\ 
1.6 & 1.3 - 4.5 \end{bmatrix} = \begin{bmatrix} -0.8 & 1.6 \\ 
1.6 & -3.2 \end{bmatrix}$$

Both rows encode the same equation (the matrix is rank-deficient at an eigenvalue by design). Take the first row:

$$-0.8x + 1.6y = 0$$

$$\Rightarrow\quad y = 0.5x$$

So the eigenvector points in the direction:

$$\mathbf{v} \propto \begin{bmatrix} 1 
\\ 0.5 \end{bmatrix}$$

> **Intuition:** For every 1 unit along Feature 1, you move 0.5 units along Feature 2 on this principal axis — matching the positive correlation seen in the covariance matrix.

---

## Step 6 — Normalize the Eigenvector

Eigenvectors must be unit length (norm = 1) so projections are true orthogonal distances:

$$\|\mathbf{v}\| = \sqrt{1^2 + 0.5^2} = \sqrt{1 + 0.25} = \sqrt{1.25} \approx 1.118$$

$$\mathbf{v} = \frac{1}{\sqrt{1.25}} \begin{bmatrix} 1 \\ 
0.5 \end{bmatrix} \approx \begin{bmatrix} 0.894 \\ 
0.447 \end{bmatrix}$$

**Verify unit length:** $0.894^2 + 0.447^2 \approx 0.799 + 0.200 = 0.999 \approx 1$ ✓

---

## Step 7 — Project the Data

Project every centered row onto the principal component $\mathbf{v}$ via dot product:

$$X_{\text{proj}} = X_c \cdot \mathbf{v}$$

**Row-by-row calculation:**

| Row | $X_c$ row         | Dot product                                    | Result   |
|-----|-------------------|------------------------------------------------|----------|
| 1   | $[-0.8,\ -1.4]$   | $(-0.8)(0.894) + (-1.4)(0.447)$               | $-1.341$ |
| 2   | $[-2.8,\ -0.4]$   | $(-2.8)(0.894) + (-0.4)(0.447)$               | $-2.682$ |
| 3   | $[\ 0.2,\ -0.4]$  | $(\ 0.2)(0.894) + (-0.4)(0.447)$              | $\ 0.000$ |
| 4   | $[\ 1.2,\ \ 0.6]$ | $(\ 1.2)(0.894) + (\ 0.6)(0.447)$             | $+1.341$ |
| 5   | $[\ 2.2,\ \ 1.6]$ | $(\ 2.2)(0.894) + (\ 1.6)(0.447)$             | $+2.682$ |

$$\boxed{X_{\text{proj}} = \begin{bmatrix} -1.341 \\ 
-2.682 \\ 
0.000 \\ 
1.341 \\ 
2.682 \end{bmatrix}}$$

The 5 original 2D points are now 5 scalar values along the most informative axis — **dimensionality reduced from 2D → 1D**, retaining 90% of the variance.

---

## Final Summary

| Step | Operation | Purpose |
|------|-----------|---------|
| **1. Mean**        | $\mu = \frac{1}{n}\sum x_i$            | Measure center of data              |
| **2. Center**      | $X_c = X - \mu$                        | Remove bias, zero-center            |
| **3. Covariance**  | $C = \frac{1}{n-1} X_c^T X_c$         | Capture feature relationships       |
| **4. Eigenvalues** | $\det(C - \lambda I) = 0$              | Measure variance per direction      |
| **5. Eigenvectors**| $(C - \lambda I)\mathbf{v} = 0$        | Find directions of maximum variance |
| **6. Normalize**   | $\mathbf{v} \leftarrow \mathbf{v} / \|\mathbf{v}\|$ | Make unit-length for projection |
| **7. Project**     | $X_{\text{proj}} = X_c \cdot W$        | Compress data along best axes       |

```
Mean center → Covariance → Eigenvalue  → Eigenvector → Projection
 remove bias    feature      variance      direction      compress
                relations    amount        of max var
```

---

## Full Implementation

```python
import numpy as np

def pca_projection(X, k):
    X = np.array(X, dtype=float)
    n, d = X.shape

    # Steps 1 & 2: Center the data
    X_centered = X - np.mean(X, axis=0)

    # Step 3: Sample covariance matrix (divide by n-1)
    C = (X_centered.T @ X_centered) / (n - 1)

    # Steps 4 & 5: Eigendecomposition of symmetric matrix
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort by descending eigenvalue, take top-k columns
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    W = eigenvectors[:, :k]           # shape: (d, k)

    # Step 6: Sign canonicalization — largest component always positive
    for i in range(k):
        max_idx = np.argmax(np.abs(W[:, i]))
        if W[max_idx, i] < 0:
            W[:, i] = -W[:, i]

    # Step 7: Project
    X_proj = X_centered @ W           # shape: (n, k)
    return X_proj.tolist()


# --- Reproduce the worked example above ---
X = [[2, 0], [0, 1], [3, 1], [4, 2], [5, 3]]
result = pca_projection(X, k=1)
print(result)
# → [[-1.341], [-2.682], [0.0], [1.341], [2.682]]
```

### Why `np.linalg.eigh` and not power iteration?

`eigh` is built for **symmetric matrices** — which covariance matrices always are. It uses LAPACK's divide-and-conquer algorithm and handles zero eigenvalues exactly. Power iteration with deflation accumulates floating-point noise and produces garbage directions when data lies in a lower-dimensional subspace:

```python
# Rank-deficient case — all points on the line y = x + 1
X = [[1,2],[3,4],[5,6],[7,8]]
pca_projection(X, k=2)

# eigh  → [[-4.243, 0.0], [-1.414, 0.0], [1.414, 0.0], [4.243, 0.0]]  ✓
# power → [[-4.243,-4.243], ...]  ← second column wrong                 ✗
```
