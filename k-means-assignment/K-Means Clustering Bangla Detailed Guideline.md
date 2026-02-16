## K-Means Clustering: Bangla Detailed Guideline

Machine Learning-এর unsupervised learning section-এ সবচেয়ে ব্যবহারযোগ্য algorithmগুলোর একটি হলো **K-Means Clustering**। এটা data-কে label ছাড়া group করে । মানে আমরা আগে থেকে জানি না কোনটা কোন class, কিন্তু algorithm নিজে pattern খুঁজে বের করে।

আজকে আমরা একদম পরিষ্কারভাবে দেখবো:

- K-Means কীভাবে কাজ করে
- দুইটা core step: **Assignment** ও **Centroid Update**
- Example
- Mathematical intuition
- Efficient Python implementation

# 1 K-Means আসলে কী করে?

ধরো তোমার কাছে কিছু data point আছে। তুমি চাও এগুলোকে K সংখ্যক group-এ ভাগ করতে, যেন একই group-এর ভিতরে থাকা point গুলো একে অপরের কাছাকাছি থাকে।

Objective function:
$$
J = \min \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2
$$
মানে:
প্রতিটা cluster-এর centroid থেকে point-গুলোর squared distance minimize করা।

# 2 K-Means Algorithm-এর দুইটা Step

Algorithm alternately দুইটা কাজ করে:

### Step 1: Assignment Step

প্রতিটা point কে সবচেয়ে কাছের centroid-এ assign করা হয়।

### Step 2: Update Step

প্রতিটা cluster-এর নতুন centroid হিসাব করা হয় — mean of all assigned points।

এভাবে iteration চলতে থাকে যতক্ষণ না centroid আর পরিবর্তন হচ্ছে না।

# 3 Banglay Practical Example

## Problem Scenario

ধরো তুমি একটা দোকানের customer segmentation করছো।

তোমার কাছে দুইটা feature আছে:

- মাসিক খরচ (Monthly Spending)
- দোকানে আসার frequency

Data:

| Customer | Spending | Frequency |
| -------- | -------- | --------- |
| A        | 2000     | 2         |
| B        | 2500     | 3         |
| C        | 8000     | 8         |
| D        | 8500     | 9         |

আমরা K = 2 ধরলাম।

## Iteration 1

### Initial Centroid (Random)

ধরলাম:

- C1 = (2000, 2)
- C2 = (8000, 8)

## Assignment Step

Distance measure = Euclidean Distance

Customer B:

###### Distance to C1

$$
\sqrt{(2500 - 2000)^2 + (3 - 2)^2} = 500
$$

------

###### Distance to C2

এখন C2 = (8000, 8)
$$
\sqrt{(2500 - 8000)^2 + (3 - 8)^2} = 5500
$$

## Comparison

- Distance to C1 ≈ **500**
- Distance to C2 ≈ **5500**

অর্থাৎ C2 অনেক দূরে  প্রায় ১১ গুণ বেশি দূরে।

তাই Customer B অবশ্যই **Cluster 1**-এ যাবে।

Customer D যাবে Cluster 2 এ।

Result:

Cluster 1: A, B
Cluster 2: C, D

## Update Step

New centroid হিসাব করি।

Cluster 1:

Mean Spending = (2000 + 2500) / 2 = 2250   # For A and B
Mean Frequency = (2 + 3) / 2 = 2.5

New C1 = (2250, 2.5)

Cluster 2:

Mean Spending = (8000 + 8500) / 2 = 8250 # For C and D
Mean Frequency = (8 + 9) / 2 = 8.5

New C2 = (8250, 8.5)

## Iteration 2

আবার Assignment step হবে।

কিন্তু এবার আর cluster change হবে না।

Centroid stable → Converged.

# 4 Visualization Idea

Cluster 1 → Low spending customers
Cluster 2 → High spending customers

Business insight:
 তুমি high-value customers আলাদা marketing strategy দিতে পারো।

এটাই হলো practical power of K-Means.

# 5 Complexity Analysis

- Time Complexity: O(n * k * d * i)

Where:

- n = number of points
- k = number of clusters
- d = dimension
- i = number of iterations

# 6 Production-Level Efficient Python Implementation

```
import numpy as np


class KMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4,
                 random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def _initialize_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(X.shape[0],
                             self.n_clusters,
                             replace=False)
        return X[indices]

    def _assign_clusters(self, X):
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.centroids,
            axis=2
        )
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.array([
            X[labels == i].mean(axis=0)
            for i in range(self.n_clusters)
        ])
        return new_centroids

    def fit(self, X):
        X = np.array(X)
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)

            shift = np.linalg.norm(
                self.centroids - new_centroids
            )

            if shift < self.tol:
                break

            self.centroids = new_centroids

        return self

    def predict(self, X):
        X = np.array(X)
        return self._assign_clusters(X)
```

# 7 Real-World Use Cases

- Customer Segmentation
- Image Compression
- Document Clustering
- Market Basket Analysis
- Anomaly Preprocessing

