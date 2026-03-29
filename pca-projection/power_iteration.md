# PCA in Production: Why Power Iteration Fails and `np.linalg.eigh` is the Right Choice

মেশিন লার্নিং মডেলে Dimensionality Reduction এর জন্য PCA আমাদের প্রথম পছন্দ। সাধারণত আমরা scikit-learn দিয়েই কাজ শেষ করি। কিন্তু প্রোডাকশনে যদি স্ক্র্যাচ থেকে PCA ইমপ্লিমেন্ট করতে হয়?

নতুন অবস্থায় অনেকেই eigenvector বের করতে **Power Iteration** অ্যালগরিদমের কথা ভাবেন। থিওরিটিক্যালি কাজ করলেও, প্রোডাকশন সিস্টেমে এটি একটি বড় ফাঁদ।

এই পোস্টে আমরা দেখব কেন Power Iteration ফেইল করে এবং কেন `numpy.linalg.eigh` হলো সঠিক পছন্দ।

---

## Power Iteration এবং Deflation এর ফাঁদ

Power Iteration একটি iterative প্রসেস। এটি randomly শুরু করে matrix-এর সবচেয়ে বড় eigenvector (প্রথম Principal Component) খুঁজে বের করে। এরপর **Deflation** নামক প্রক্রিয়ায় মূল matrix থেকে সেই component এর প্রভাব মুছে পরবর্তী eigenvector খোঁজা হয়।

Deflation এর সমীকরণ:

$$C_{\text{new}} = C - \lambda (v v^T)$$

শুনতে logical মনে হলেও, real-world ডেটায় এটি দুটি বড় সমস্যা তৈরি করে।

---

### সমস্যা ১: Floating-Point Noise এবং Zero Variance

ধরুন, আপনার ডেটাসেট 2D স্পেসে একদম $y = x$ লাইনের উপর অবস্থিত। এর মানে ডেটার সম্পূর্ণ variance একটিমাত্র dimension-এ। দ্বিতীয় dimension-এ কোনো variance নেই, অর্থাৎ দ্বিতীয় eigenvalue $0.0$ হওয়া উচিত।

Power Iteration যখন প্রথম component বের করে Deflation করে, তখন matrix টি গাণিতিকভাবে:

$$C_{\text{new}} = \begin{bmatrix} 0 & 0 \\ 
0 & 0 \end{bmatrix}$$

হওয়ার কথা। কিন্তু কম্পিউটারে floating-point এর সীমাবদ্ধতার কারণে সেখানে সামান্য noise থেকে যায়:

$$C_{\text{new}} \approx \begin{bmatrix} 1.11 \times 10^{-16} & -2.22 \times 10^{-16} \\ 
-2.22 \times 10^{-16} & 1.11 \times 10^{-16} \end{bmatrix}$$

Power Iteration শূন্য বুঝতে না পেরে এই $10^{-16}$ noise-কেই বারবার গুণ করে একটি সম্পূর্ণ অর্থহীন eigenvector তৈরি করে ফেলে।

---

### সমস্যা ২: Sign Ambiguity

Power Iteration random seed-এর উপর ভিত্তি করে $+v$ বা $-v$ যেকোনো দিকে converge করতে পারে। Deflation এর সময় এই চিহ্নের সামান্য পরিবর্তনে পরবর্তী eigenvector-গুলোর calculation-এ ভুলের মাত্রা জ্যামিতিক হারে বাড়তে থাকে। শেষের দিকের projection গুলো পুরোপুরি ভুল আসে।

---

## সমাধান: `np.linalg.eigh`

এই সমস্যাগুলো এড়াতে প্রোডাকশনে সবসময় `numpy.linalg.eigh` ব্যবহার করা উচিত।

**কেন `eigh`, সাধারণ `eig` নয়?**

PCA-তে covariance matrix সবসময় symmetric হয় ($C = C^T$)। `eigh` (শেষে `h` মানে Hermitian/Symmetric) বিশেষভাবে এই ধরনের matrix-এর জন্যই ডিজাইন করা।

| বৈশিষ্ট্য | Power Iteration + Deflation | `np.linalg.eigh` |
|---|---|---|
| Algorithm | Iterative, sequential | LAPACK `dsyevd` (Divide-and-conquer) |
| Deflation | প্রতিটি component এর পর | নেই — একবারেই decompose করে |
| Floating-point noise | তৈরি হয় | হয় না |
| Sign ambiguity | আছে | নেই |
| Zero eigenvalue | $10^{-16}$ noise আসে | নিখুঁত `0.0` রিটার্ন করে |
| Complex numbers | সম্ভব | গাণিতিকভাবে অসম্ভব |

**No Deflation Needed:** `eigh` পুরো matrix-কে একসাথে (simultaneously) decompose করে। ফলে floating-point noise বা sign ambiguity তৈরির কোনো সুযোগ নেই।

**Perfect Zeros:** কোনো dimension-এ variance না থাকলে এটি noise-এ বিভ্রান্ত না হয়ে নিখুঁত `0.0` রিটার্ন করে।

**Guaranteed Real Numbers:** Symmetric matrix থেকে সর্বদা real eigenvalue পাওয়া যায়, complex number আসার কোনো সম্ভাবনা নেই।

---

## গাণিতিক প্রমাণ: Rank-Deficient Dataset

একটি rank-deficient ডেটাসেট দিয়ে বিষয়টি দেখি। ধরুন, 2D ডেটা পয়েন্টগুলো $y = x$ লাইনের উপর:

$$X = \begin{bmatrix} 1 & 1 \\ 
2 & 2 \\ 
3 & 3 \end{bmatrix}$$

Mean subtraction এর পর covariance matrix:

$$C = \begin{bmatrix} 1 & 1 \\ 
1 & 1 \end{bmatrix}$$

এই matrix-এর মোট variance হলো $2$ (diagonal এর যোগফল: $1 + 1$)।

---

### ধাপ ১: প্রথম Principal Component

Power Iteration সহজেই বের করে:

$$\lambda_1 = 2, \qquad v_1 = \begin{bmatrix} 0.707 \\ 0.707 \end{bmatrix}$$

প্রথম component-এই পুরো variance ($2$) কভার হয়ে গেছে। তাই গাণিতিকভাবে দ্বিতীয় eigenvalue নিখুঁত $0.0$ হওয়া উচিত।

---

### ধাপ ২: Deflation (যেখানে ব্যর্থতা শুরু হয়)

$$C_{\text{new}} = C - \lambda_1 (v_1 v_1^T)$$

$$C_{\text{new}} = \begin{bmatrix} 1 & 1 \\ 
1 & 1 \end{bmatrix} - 2 \begin{bmatrix} 0.5 & 0.5 \\ 
0.5 & 0.5 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 
0 & 0 \end{bmatrix}$$

কাগজে-কলমে ফলাফল শূন্য। কিন্তু কম্পিউটারে $0.707$ (আসলে $\frac{1}{\sqrt{2}}$) অসীম decimal পর্যন্ত store করা যায় না। তাই Python-এ বিয়োগের পর:

$$C_{\text{new}} \approx \begin{bmatrix} 1.11 \times 10^{-16} & -2.22 \times 10^{-16} \\ 
-2.22 \times 10^{-16} & 1.11 \times 10^{-16} \end{bmatrix}$$

এটি কোনো আসল information নয় — স্রেফ **floating-point noise**। Power Iteration এই $C_{\text{new}}$ থেকে দ্বিতীয় eigenvector খুঁজতে গিয়ে একটি সম্পূর্ণ garbage vector output দেয়।

`np.linalg.eigh` কোনো Deflation ছাড়াই সরাসরি symmetric matrix solve করে বলে দেয়:

```
Eigenvalues: [0.0, 2.0]
```

কোনো noise নেই, কোনো ভুল নেই।

---

## Code

```python
"""
PCA: Power Iteration Deflation Trap vs np.linalg.eigh
"""

import numpy as np


def demonstrate_floating_point_noise() -> None:
    """
    Demonstrates how Power Iteration's deflation creates floating-point noise
    on a rank-deficient covariance matrix, and how eigh solves it elegantly.
    """
    print("Experiment: The Floating-Point Trap in PCA")
    print("=" * 50)

    # Rank-deficient covariance matrix (data lies on y = x line)
    C = np.array([[1.0, 1.0],
                  [1.0, 1.0]])

    # Analytically known: lambda_1 = 2, v_1 = [1/sqrt(2), 1/sqrt(2)]
    lambda_1 = 2.0
    v_1 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

    # Deflation: C_new = C - lambda_1 * (v_1 @ v_1.T)
    outer_product = np.outer(v_1, v_1)
    C_new = C - (lambda_1 * outer_product)

    print("\n1. Result of Manual Deflation (Power Iteration approach):")
    print("Notice the 10^-16 noise instead of absolute zeros.\n")
    print(repr(C_new))
    print("-" * 50)

    # The correct approach: eigh for symmetric matrices
    eigenvalues, _ = np.linalg.eigh(C)

    print("\n2. Result using np.linalg.eigh:")
    # Note: eigh returns eigenvalues in ascending order
    print(f"Eigenvalues: {repr(eigenvalues)}")
    print("=" * 50)


if __name__ == "__main__":
    demonstrate_floating_point_noise()
```

**Expected Output:**

```
Experiment: The Floating-Point Trap in PCA
==================================================

1. Result of Manual Deflation (Power Iteration approach):
Notice the 10^-16 noise instead of absolute zeros.

array([[ 1.11022302e-16, -2.22044605e-16],
       [-2.22044605e-16,  1.11022302e-16]])
--------------------------------------------------

2. Result using np.linalg.eigh:
Eigenvalues: array([0., 2.])
==================================================
```


