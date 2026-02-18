# K-Means Centroid Update Step

K-Means হলো সবচেয়ে জনপ্রিয় clustering অ্যালগরিদমগুলোর একটি। এটি data-কে kটি cluster-এ ভাগ করে এমনভাবে, যাতে প্রতিটি cluster-এর ভেতরের variance সর্বনিম্ন হয়।

এই ব্লগে আমরা বিস্তারিত আলোচনা করবো:

- Centroid Update Step কী
- কেন Mean ব্যবহার করা হয়
- ধাপে ধাপে উদাহরণ
- Multi-dimensional ক্ষেত্রে গণনা
- Vectorized implementation
- Empty cluster সমস্যা
- Convergence guarantee
- Online ও Mini-batch update
- K-Means বনাম K-Medians বনাম K-Medoids

# ১. K-Means অ্যালগরিদমের দুটি ধাপ

প্রতিটি iteration-এ দুইটি ধাপ থাকে:

### Step 1: Assignment

প্রতিটি data point-কে সবচেয়ে কাছের centroid-এ assign করা হয়।

### Step 2: Centroid Update

প্রতিটি cluster-এর centroid নতুন করে হিসাব করা হয়।

এই লেখার মূল ফোকাস হলো দ্বিতীয় ধাপ।

# ২. Centroid Update Step কী?

ধরুন j নম্বর cluster-এর সব point-এর সেট হলো:
$$
C_j
$$
তাহলে centroid হবে:
$$
\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i
$$
অর্থাৎ:

> Cluster-এর সব point-এর গড় (mean) হলো centroid।

# ৩. কেন Mean ব্যবহার করি?

K-Means যে objective minimize করে:
$$
J = \sum_{j=1}^{k} \sum_{i \in C_j} ||x_i - \mu_j||^2
$$
এটা হলো within-cluster sum of squared distances।

এখন আমরা চাই:
$$
\mu_j^* = \arg\min_\mu \sum_{i \in C_j} ||x_i - \mu||^2
$$
Derivative নিয়ে zero করলে পাওয়া যায়:
$$
\sum_{i \in C_j} (x_i - \mu) = 0
$$
⇒
$$
\sum_{i \in C_j} x_i = |C_j| \cdot \mu
$$
⇒
$$
\mu = \frac{1}{|C_j|} \sum x_i
$$
অর্থাৎ Mean-ই squared distance মিনিমাইজ করে।

এটাই K-Means-এর mathematical backbone।

# ৪. ধাপে ধাপে উদাহরণ

ধরুন ৫টি 2D data point আছে:

```
x1 = (1,2)
x2 = (2,1)
x3 = (4,5)
x4 = (5,4)
x5 = (3,3)
```

বর্তমান cluster assignment:

Cluster 1: (1,2), (2,1), (3,3)
 Cluster 2: (4,5), (5,4)

## Cluster 1 centroid হিসাব

$$
\mu_1 = \frac{1}{3} [(1,2) + (2,1) + (3,3)]
$$

x-coordinate:
 1 + 2 + 3 = 6

y-coordinate:
 2 + 1 + 3 = 6
$$
\mu_1 = (2,2)
$$

## Cluster 2 centroid হিসাব

$$
\mu_2 = \frac{1}{2} [(4,5) + (5,4)]
$$

x-coordinate:
 4 + 5 = 9

y-coordinate:
 5 + 4 = 9
$$
\mu_2 = (4.5, 4.5)
$$

# ৫. Multi-Dimensional ক্ষেত্রে

d-dimensional হলে:
$$
\mu_j = (\mu_{j1}, \mu_{j2}, ..., \mu_{jd})
$$
যেখানে:
$$
\mu_{jk} = \frac{1}{|C_j|} \sum x_{ik}
$$
অর্থাৎ প্রতিটি feature আলাদাভাবে average নেওয়া হয়।

# ৬. Empty Cluster সমস্যা

যদি কোনো cluster-এ কোনো point না থাকে:
$$
|C_j| = 0
$$
Mean undefined হবে।

সমাধান:

1. আগের centroid রেখে দেওয়া
2. Random data point দিয়ে reinitialize
3. Farthest point দিয়ে replace করা
4. Largest cluster split করা

Robust implementation-এ এটা অবশ্যই handle করতে হবে।

# ৭. কেন Objective কখনো বাড়ে না?

Assignment fixed থাকলে, mean নেওয়া মানেই ঐ cluster-এর error minimum।

তাই centroid update step:

- Objective কমায়
- অথবা একই রাখে
- কখনো বাড়ায় না

যেহেতু:

- Objective ≥ 0
- Possible assignment finite

⇒ K-Means finite iteration-এ converge করে।

# ৮. Online Update (Streaming Data)

নতুন point এলে পুরো mean আবার হিসাব না করে:
$$
\mu_{new} = \mu_{old} + \frac{1}{n+1}(x_{new} - \mu_{old})
$$
এটা O(d) time নেয়।

Mini-batch K-Means এই ধারণা ব্যবহার করে।

# ৯. Weighted K-Means

যদি data point-এর weight থাকে:
$$
\mu_j = \frac{\sum w_i x_i}{\sum w_i}
$$
সব weight = 1 হলে সাধারণ mean।

# ১০. K-Means vs K-Medians vs K-Medoids

| Algorithm | Centroid          | Distance   | Robustness        |
| --------- | ----------------- | ---------- | ----------------- |
| K-Means   | Mean              | L2         | Outlier sensitive |
| K-Medians | Median            | L1         | More robust       |
| K-Medoids | Actual data point | Any metric | Highly robust     |

K-Medoids computationally expensive:
$$
O(|C_j|^2)
$$

# ১১. Computational Complexity

Centroid update:
$$
\sum_j O(|C_j| \cdot d) = O(n \cdot d)
$$
অর্থাৎ data size-এর সাথে linear growth।

# More

> **Assignment fixed থাকলে**, mean নেওয়া মানেই ঐ cluster-এর squared error minimum।

মানে কী?

- কোন point কোন cluster-এ যাবে — এটা ধরা আছে (fixed)
- এখন শুধু centroid কোথায় বসাবো — সেটা ঠিক করছি
- এমন জায়গায় বসাতে চাই যাতে total squared distance সবচেয়ে কম হয়

#  ছোট উদাহরণ দিয়ে শুরু করি (1D case)

ধরো একটা cluster-এ ৩টা সংখ্যা আছে:

```
2, 4, 6
```

আমরা centroid = μ কোথায় বসাবো?

Objective:
$$
J(\mu) = (2-\mu)^2 + (4-\mu)^2 + (6-\mu)^2
$$
আমরা এমন μ চাই যেটা J সবচেয়ে কম করবে।

## Step 1: Expand করি

$$
J(\mu) =
(4 - 4\mu + \mu^2)
+
(16 - 8\mu + \mu^2)
+
(36 - 12\mu + \mu^2)
$$

সব যোগ করলে:
$$
= 56 - 24\mu + 3\mu^2
$$

## Step 2: Derivative নেই

$$
\frac{dJ}{d\mu} = -24 + 6\mu
$$

Zero করি:
$$
-24 + 6\mu = 0
$$
এটাই mean।

# Intuition

Mean = 4
 এটাই মাঝখানে বসে।

যদি μ = 3 বসাও?

Error বাড়বে।

যদি μ = 5 বসাও?

Error বাড়বে।

Mean হলো সেই balance point যেখানে:
$$
\sum (x_i - \mu) = 0
$$
অর্থাৎ positive deviation = negative deviation

এটাই squared error-এর minimum।

# এখন 2D intuition

ধরো 3টা point:

```
(1,2)
(2,1)
(3,3)
```

Centroid কোথায়?

Mean:

```
(2,2)
```

এখন ভাবো তুমি centroid-টা একটু ডানে সরাও।

কি হবে?

সব point থেকে distance বাড়বে।

একটু উপরে সরাও?

আবার distance imbalance হবে।

Mean হলো geometric center of mass।

Physics analogy:

> সব point যদি equal weight-এর particle হয়
>  centroid হলো balance point।

# General Proof Idea (Vector Form)

Objective:
$$
J(\mu) = \sum ||x_i - \mu||^2
$$
Derivative নিলে পাই:
$$
\frac{\partial J}{\partial \mu}
=
-2 \sum (x_i - \mu)
$$
Zero করলে:
$$
\sum x_i = n \mu
$$
⇒
$$
\mu = \frac{1}{n} \sum x_i
$$
এটাই mean।