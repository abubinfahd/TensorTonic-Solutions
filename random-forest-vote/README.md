# Random Forest Explained (Bangla)

## What Is a Random Forest?

Random Forest হলো একটি **ensemble learning algorithm** যা অনেকগুলো Decision Tree একসাথে ব্যবহার করে।

Single decision tree সাধারণত overfit করে। কিন্তু Random Forest অনেকগুলো tree train করে এবং তাদের prediction aggregate করে। ফলে variance কমে এবং generalization improve হয়।

---

## How Random Forest Makes Predictions

### Classification

Final prediction = Majority Vote (Mode)

$$
\hat{y} = \text{mode}(\hat{y}_1, \hat{y}_2, ..., \hat{y}_T)
$$

Where:
- $\hat{y}_t$ = prediction of the $t$-th tree  
- $T$ = total number of trees

### Example

| Tree | Prediction |
| ---- | ---------- |
| 1    | A          |
| 2    | B          |
| 3    | A          |
| 4    | A          |
| 5    | B          |

Vote Count:

* A → 3
* B → 2
* C → 0

Final Prediction → **A**

---

### Regression

Final prediction = Average

$$
\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t
$$

Example:

250000 + 270000 + 245000 + 260000 + 255000 = 1,280,000
Final = 1,280,000 / 5 = **256,000**

---

## Why Voting Works (Wisdom of Crowds)

ধরি প্রতিটি tree এর accuracy = 60%

Single tree → 60%
5 trees majority vote → ~68.3%

Variance reduction principle:

$$
Var(\bar{X}) = \frac{Var(X)}{n}
$$

More independent trees → lower variance.

---

## Hard Voting vs Soft Voting

### Hard Voting

* Each tree gives one vote
* Final prediction = Mode

### Soft Voting

* Each tree outputs probabilities
* Average probabilities
* Pick highest average probability

Example:

| Tree | A   | B   | C   |
| ---- | --- | --- | --- |
| 1    | 0.7 | 0.2 | 0.1 |
| 2    | 0.4 | 0.5 | 0.1 |
| 3    | 0.6 | 0.3 | 0.1 |

Average:

* A → 0.567
* B → 0.333
* C → 0.1

Final Prediction → **A**

Soft voting সাধারণত বেশি accurate কারণ এটি confidence consider করে।

---

## Why Random Forest Works

### 1️ Bagging (Bootstrap Aggregating)

* Random sample (with replacement) দিয়ে প্রতিটি tree train হয়
* Trees বিভিন্ন data দেখে
* Diversity তৈরি হয়

### 2️ Feature Randomization

* প্রতিটি split এ random subset of features ব্যবহার করা হয়
* Trees decorrelated হয়

---

##Variance Reduction

Random Forest bias খুব বেশি বাড়ায় না কিন্তু variance কমায়।

Single deep tree → High variance
Random Forest → Lower variance + stable prediction

---

## Number of Trees

Typical range: **100 – 500 trees**

Rule of thumb:

> Add trees until Out-of-Bag (OOB) error stabilizes.

---

## Out-of-Bag (OOB) Prediction

Bootstrap sampling করলে প্রায় 37% samples প্রতিটি tree তে ব্যবহৃত হয় না।

এই samples ব্যবহার করে validation করা যায়।
এটি built-in validation mechanism।

---

## Feature Importance

### 1️ Mean Decrease in Impurity

* Gini / Entropy কত কমেছে তা measure করে

### 2️ Permutation Importance

* Feature shuffle করে accuracy drop measure করা হয়
* বেশি reliable

---

## Computational Considerations

Prediction complexity:

$$
O(T \cdot d)
$$

* T = number of trees
* d = average tree depth

Pros:

* Parallelizable
* Strong baseline model
* Works great for tabular data

Cons:

* Memory heavy
* Less interpretable than single tree

---

## When to Use Random Forest

Use when:

* Structured tabular data
* Non-linear relationships
* Medium size dataset
* Need strong baseline before boosting methods

---

## Summary

Random Forest:

* Reduces variance
* Improves generalization
* Uses bagging + feature randomness
* Works extremely well for real-world tabular ML problems

It is one of the most reliable baseline models in practical machine learning.
