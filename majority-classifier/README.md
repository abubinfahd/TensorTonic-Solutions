# Majority Classifier (Zero-Rule Classifier)

Majority Classifier হলো সবচেয়ে সহজ classification model।
এটি কোনো feature ব্যবহার করে না। শুধু training data-তে যে class সবচেয়ে বেশি আছে, সেটিই সব input-এর জন্য predict করে।

এটি baseline model হিসেবে ব্যবহৃত হয়।

---

## 1️ Mathematical Definition

$$
\hat{y} = \arg\max_{c} \sum_{i=1}^{n} \mathbf{1}(y_i = c)
$$

যেখানে,

* (1[.]) হলো indicator function
* যে class সবচেয়ে বেশি এসেছে, সেটাই prediction

---

## 2️ কেন Majority Classifier গুরুত্বপূর্ণ?

### Baseline Performance

যে কোনো ML model majority classifier-কে beat করতে না পারলে, সেই model কার্যকর নয়।

### Sanity Check

Evaluation pipeline সঠিকভাবে কাজ করছে কিনা তা যাচাই করার সহজ উপায়।

### Edge Case Handling

যদি feature না থাকে বা সব missing হয়, majority fallback ব্যবহার করা যায়।

---

## 3️ Step-by-Step Example

ধরি training labels:

```
[A, B, A, A, C, B, A, A, B, C]
```

### Step 1: Count Class Frequency

| Class | Count |
| ----- | ----- |
| A     | 5     |
| B     | 3     |
| C     | 2     |

### Step 2: Find Majority

Class **A** সবচেয়ে বেশি (5)

### Step 3: Prediction Rule

যে কোনো input → predict **A**

---

## 4️ Training Accuracy

মোট sample = 10
Majority count = 5

Accuracy = 5 / 10 = **50%**

এটাই majority classifier-এর expected training accuracy।

---

## 5️ Probability Output

Majority classifier class probability-ও দিতে পারে।

ধরি:

```
[A, A, A, B, B, C]
```

| Class | Probability |
| ----- | ----------- |
| A     | 3/6 = 0.5   |
| B     | 2/6 ≈ 0.33  |
| C     | 1/6 ≈ 0.17  |

Prediction → A
Probability vector → `[0.5, 0.33, 0.17]`

এটি আসলে empirical class prior:

$$
P(y=c) = \frac{\text{count}(c)}{n}
$$

---

## 6️ Tie Handling

ধরি:

```
[A, B, A, B, C, C]
```

সব class = 2

সম্ভাব্য strategy:

* Alphabetically first
* Smallest index
* Deterministic rule

Production system-এ deterministic behaviour রাখা উত্তম।

---

## 7️ Implementation (Pure Python)

```python
from collections import Counter


class MajorityClassifier:
    """
    Simple majority (most frequent) classifier.
    """

    def __init__(self):
        self.majority_class = None
        self.class_counts = None
        self.n_samples = 0

    def fit(self, y):
        if not y:
            raise ValueError("Input labels cannot be empty.")

        self.class_counts = Counter(y)
        self.n_samples = len(y)

        self.majority_class = max(
            sorted(self.class_counts),
            key=self.class_counts.get
        )

        return self

    def predict(self, X):
        if self.majority_class is None:
            raise ValueError("Model is not fitted yet.")

        return [self.majority_class] * len(X)

    def predict_proba(self):
        if self.class_counts is None:
            raise ValueError("Model is not fitted yet.")

        return {
            cls: count / self.n_samples
            for cls, count in self.class_counts.items()
        }
```

---

## 8️ Usage Example

```python
y_train = ["A", "B", "A", "A", "C", "B", "A", "A", "B", "C"]

model = MajorityClassifier()
model.fit(y_train)

predictions = model.predict([1, 2, 3])
probabilities = model.predict_proba()

print("Prediction:", predictions)
print("Probabilities:", probabilities)
```

Output:

```
Prediction: ['A', 'A', 'A']
Probabilities: {'A': 0.5, 'B': 0.3, 'C': 0.2}
```

---

## 9️ Computational Complexity

| Stage      | Complexity |
| ---------- | ---------- |
| Training   | O(n + C)   |
| Prediction | O(1)       |

যেখানে
n = number of samples
C = number of classes

এটি অত্যন্ত efficient।

---

## 10 Imbalanced Dataset Scenario

Binary classification:

* 95% Negative
* 5% Positive

Majority classifier accuracy = 95%

তোমার model যদি 96% দেয় → marginal improvement।
এখানে Precision, Recall, F1-score দেখা জরুরি।

---

## 1️1️ Balanced Dataset Scenario

3-class balanced dataset:

Accuracy = 1/3 ≈ 33%

এখানে majority classifier দুর্বল baseline।

---

## 1️2️ Theoretical Insight

Majority classifier শুধু class prior ব্যবহার করে:

[
P(y)
]

এটি likelihood:

[
P(x|y)
]

সম্পূর্ণ ignore করে।

Decision tree-তে split না হলে root node-ই majority classifier হয়ে যায়।

---

## 1️3️ Limitations

* Feature ব্যবহার করে না
* Minority class কখনো predict করে না
* Balanced problem-এ দুর্বল
* Insight প্রদান করে না

