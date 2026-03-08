# Linear Regression Closed Form (Normal Equation) — Bengali Guide

## ভূমিকা

Linear Regression হলো সবচেয়ে ক্লাসিক এবং বেসিক Machine Learning অ্যালগরিদমগুলোর একটি। এর লক্ষ্য হলো এমন একটি সরলরেখা (line) বা hyperplane খুঁজে বের করা যা ডেটা পয়েন্টগুলোর সাথে সবচেয়ে ভালোভাবে ফিট করে।

আমরা এমন একটি weight vector **w** খুঁজতে চাই যাতে:

$$
Xw \approx y
$$

এখানে,

* **X** → Feature matrix (n × d)
* **y** → Target vector (n × 1)
* **w** → Weight vector (d × 1)

Closed form solution ব্যবহার করলে iterative optimization (যেমন gradient descent) ছাড়াই সরাসরি weight বের করা যায়।

---

# Linear Regression Normal Equation

Linear Regression এর closed form solution হলো:

$$
w = (X^T X)^{-1} X^T y
$$

এখানে,

* `X^T` = transpose of X
* `(X^T X)^{-1}` = inverse matrix

---

# Loss Function

Linear regression এর লক্ষ্য হলো **Sum of Squared Errors (SSE)** minimize করা।

Loss function:

$$
L(w) = ||Xw - y||^2
$$

Expanded form:

$$
L(w) = (Xw - y)^T (Xw - y)
$$

এটা expand করলে পাওয়া যায়:

$$
L(w) = w^T X^T X w - 2 w^T X^T y + y^T y
$$

---

# Gradient বের করা

Loss function কে w এর উপর differentiate করলে পাই:

$$
\nabla_w L = 2X^T X w - 2X^T y
$$

Minimum point পেতে gradient = 0 সেট করি:

$$
2X^T X w - 2X^T y = 0
$$

Simplify করলে:

$$
X^T X w = X^T y
$$

এখন উভয় পাশে `(X^T X)^{-1}` multiply করলে পাই:

$$
w = (X^T X)^{-1} X^T y
$$

এটাই হলো **Normal Equation**।

---

# প্রতিটি অংশের অর্থ

## 1. X^T X (Gram Matrix)

$$
X^T X
$$

Properties:

* shape → `d × d`
* symmetric
* positive semi-definite

এটা feature গুলোর correlation বোঝায়।

---

## 2. (X^T X)^{-1}

$$
(X^T X)^{-1}
$$

এই matrix inverse তখনই থাকবে যদি:

* features গুলো perfectly correlated না হয়
* matrix singular না হয়

---

## 3. X^T y

$$
X^T y
$$

এটা প্রতিটি feature এর সাথে target এর alignment বোঝায়।

---

# উদাহরণ ১

ধরি,

$$
X =
\begin{bmatrix}
1 \
2 \
3
\end{bmatrix}
$$

$$
y =
\begin{bmatrix}
2 \
4 \
6
\end{bmatrix}
$$

এখন,

$$
w = (X^T X)^{-1} X^T y
$$

Calculate করলে পাওয়া যাবে:

$$
w = [2]
$$

Model:

$$
y = 2x
$$

এই লাইনটি সব ডেটা পয়েন্টকে perfectly fit করে।

---

# উদাহরণ ২ (Bias সহ)

ধরি feature matrix:

$$
X =
\begin{bmatrix}
1 & 1 \
1 & 2 \
1 & 3
\end{bmatrix}
$$

Target:

$$
y =
\begin{bmatrix}
3 \
5 \
7
\end{bmatrix}
$$

প্রথম column হলো bias term।

Normal equation:

$$
w = (X^T X)^{-1} X^T y
$$

Result:

$$
w =
\begin{bmatrix}
1 \
2
\end{bmatrix}
$$

Model:

$$
y = 1 + 2x
$$

---

# Step by Step Calculation

ধরি:

$$
X =
\begin{bmatrix}
1 & 1 \
1 & 2 \
1 & 3
\end{bmatrix}
$$

$$
y =
\begin{bmatrix}
1 \
2 \
2
\end{bmatrix}
$$

## Step 1

$$
X^T X =
\begin{bmatrix}
3 & 6 \
6 & 14
\end{bmatrix}
$$

## Step 2

$$
X^T y =
\begin{bmatrix}
5 \
11
\end{bmatrix}
$$

## Step 3

$$
(X^T X)^{-1} =
\frac{1}{6}
\begin{bmatrix}
14 & -6 \
-6 & 3
\end{bmatrix}
$$

## Step 4

$$
w = (X^T X)^{-1} X^T y
$$

$$
w =
\begin{bmatrix}
2/3 \
1/2
\end{bmatrix}
$$

Final model:

$$
\hat{y} = \frac{2}{3} + \frac{1}{2}x
$$

---

# কখন এই মেথড কাজ করবে

Normal equation কাজ করার জন্য দরকার:

$$
n \ge d
$$

এখানে,

* `n` = number of samples
* `d` = number of features

এবং feature গুলো perfectly collinear হওয়া যাবে না।

---

# Computational Complexity

| Operation      | Complexity    |
| -------------- | ------------- |
| X^T X          | O(nd^2)       |
| Matrix inverse | O(d^3)        |
| Total          | O(nd^2 + d^3) |

যখন feature সংখ্যা কম থাকে তখন closed form খুব দ্রুত কাজ করে।

---

# Practical Problems

Direct inverse ব্যবহার করা অনেক সময় numerically unstable হয়।

Better approaches:

* QR Decomposition
* SVD
* Gradient Descent

---

# Regularization (Ridge Regression)

Overfitting বা multicollinearity সমস্যা হলে Ridge Regression ব্যবহার করা হয়।

Formula:

$$
w = (X^T X + \lambda I)^{-1} X^T y
$$

এখানে,

* `λ` = regularization parameter
* `I` = identity matrix

এটা matrix কে stable করে এবং weights ছোট রাখে।

---

# Geometric Interpretation

Linear regression আসলে target vector **y** কে feature space এর column space এ project করে।

Residual:

$$
y - Xw
$$

এই residual vector টি X এর সব column এর সাথে orthogonal হয়:

$$
X^T (y - Xw) = 0
$$

এটাই আসলে normal equation এর geometric অর্থ।

---

# উপসংহার

Linear Regression Closed Form Solution Machine Learning এর একটি fundamental concept।

মূল আইডিয়াগুলো:

* Loss minimize করা
* Normal equation ব্যবহার করা
* Matrix algebra দিয়ে সরাসরি solution পাওয়া

Formula:

$$
w = (X^T X)^{-1} X^T y
$$

এটা ছোট feature space এর জন্য খুব efficient, তবে large scale problem এ gradient descent বা SVD বেশি stable।
