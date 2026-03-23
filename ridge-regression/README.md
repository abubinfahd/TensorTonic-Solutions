# Ridge Regression

A practical guide to Ridge Regression: what it is, why it works, and how to implement it.

---

## Overview

Ridge Regression is linear regression with an added penalty on the size of coefficients. It is designed to reduce overfitting and produce more stable, generalizable models.

---

## The Problem with Standard Linear Regression

Linear regression can fail when:

- Features are highly correlated (multicollinearity)
- The number of features is large relative to the number of samples
- The model starts fitting noise in the training data

In these cases, coefficients can grow very large and predictions become unstable.

---

## How Ridge Regression Works

### Standard Linear Regression

Minimizes the sum of squared errors:

```
Loss = sum((y_i - y_hat_i)^2)
```

### Ridge Regression

Adds a penalty term to the loss:

```
L(beta) = sum((y_i - y_hat_i)^2) + lambda * sum(beta_j^2)
```

Where:

- `y_i` is the actual value
- `y_hat_i` is the predicted value
- `beta_j` are the model coefficients
- `lambda` controls the strength of the regularization

The penalty discourages large coefficients, pushing them toward zero without forcing any of them to be exactly zero.

---

## Closed-Form Solution

Standard linear regression:

```
beta = (X^T X)^-1 X^T y
```

Ridge regression:

```
beta = (X^T X + lambda * I)^-1 X^T y
```

The addition of `lambda * I` stabilizes the matrix inversion and prevents issues caused by multicollinearity.

---

## Example

### Data

```
X = [[1, 2],
     [2, 3],
     [3, 5]]

y = [5, 8, 12]
```

### Step 1 - Compute X^T X

```
X^T X = [[14, 23],
          [23, 38]]
```

### Step 2 - Add lambda * I (lambda = 1)

```
X^T X + I = [[15, 23],
              [23, 39]]
```

### Step 3 - Compute X^T y

```
X^T y = [57, 94]
```

### Step 4 - Solve for beta

```
beta = [1.09, 1.77]
```

The coefficients are moderate and stable, not inflated by multicollinearity.

---

## Python Implementation

```python
import numpy as np


def ridge_regression(X, y, lam=1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_features = X.shape[1]
    identity = np.eye(n_features)
    beta = np.linalg.inv(X.T @ X + lam * identity) @ X.T @ y

    return beta


X = np.array([[1, 2],
              [2, 3],
              [3, 5]])

y = np.array([5, 8, 12])

beta = ridge_regression(X, y, lam=1.0)
print(beta)
```

---

## Choosing Lambda

| Lambda value | Effect |
|---|---|
| Small | Behaves close to standard linear regression |
| Large | Heavy shrinkage, risk of underfitting |

Use cross-validation to find the optimal lambda for your dataset.

---

## Feature Scaling

Always standardize features before applying Ridge Regression. Because the penalty targets coefficient size, features on larger scales will be penalized more heavily, which distorts the result.

Standardization steps:

1. Subtract the mean of each feature
2. Divide by the standard deviation

---

## Ridge vs Lasso

| Method | Penalty | Effect |
|---|---|---|
| Ridge | L2 (sum of squared coefficients) | Shrinks all coefficients toward zero |
| Lasso | L1 (sum of absolute coefficients) | Can set some coefficients to exactly zero |

Ridge is generally preferred when features are correlated, since Lasso tends to arbitrarily eliminate one of the correlated features.

---

## Summary

- Ridge Regression = Linear Regression + L2 regularization
- Adds a penalty on the size of coefficients
- Reduces overfitting and improves generalization
- Controlled by the lambda hyperparameter
- Requires feature standardization for correct behavior
- Works well when features are correlated

