# PCA (Principal Component Analysis) - Worked Example

ধরি, আমাদের একটি ডেটাসেট আছে যেখানে ২টি ফিচার এবং ২টি স্যাম্পল আছে:

$$X = \begin{bmatrix} 1 & 2 \\ 
3 & 4 \end{bmatrix}$$

প্রতিটি সারি (row) একটি ডেটা পয়েন্ট, কলাম (column) হলো ফিচার। আমরা এই 2D ডেটাকে 1D-তে নামিয়ে আনব ($k = 1$).

---

## ধাপ ১: Data Centering (Mean Centering)

প্রথমে ডেটার কেন্দ্রবিন্দুকে $(0, 0)$-তে আনতে হবে। প্রতিটি ফিচার কলামের গড় বের করে বিয়োগ করি:

$$\mu_1 = \frac{1 + 3}{2} = 2, \qquad \mu_2 = \frac{2 + 4}{2} = 3$$

$$X_c = \begin{bmatrix} 1-2 & 2-3 \\ 
3-2 & 4-3 \end{bmatrix} = \begin{bmatrix} -1 & -1 \\ 
1 & 1 \end{bmatrix}$$

---

## ধাপ ২: Covariance Matrix ($C$)

$$C = \frac{X_c^T X_c}{n - 1}$$

যেহেতু $n = 2$, তাই $n - 1 = 1$।

$$X_c^T = \begin{bmatrix} -1 & 1 \\ 
-1 & 1 \end{bmatrix}$$

$$C = X_c^T X_c = \begin{bmatrix} -1 & 1 \\ 
-1 & 1 \end{bmatrix} \begin{bmatrix} -1 & -1 \\ 
1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 
2 & 2 \end{bmatrix}$$

---

## ধাপ ৩: Eigendecomposition

Covariance matrix থেকে eigenvectors (direction) এবং eigenvalues (variance) বের করি।

সমীকরণ: $Cv = \lambda v$

**Eigenvalues:**

$$\lambda_1 = 4, \qquad \lambda_2 = 0$$

**Eigenvectors:**

$$v_1 = \begin{bmatrix} 0.707 \\ 
0.707 \end{bmatrix}, \qquad v_2 = \begin{bmatrix} -0.707 \\ 
0.707 \end{bmatrix}$$

**Insight:** $\lambda_1 = 4$ এবং $\lambda_2 = 0$ হওয়ার মানে হলো ডেটার ১০০% variance শুধুমাত্র $v_1$ বরাবর আছে। $v_2$ দিকে কোনো তথ্য নেই।

---

## ধাপ ৪: Top-$k$ Component নির্বাচন

$k = 1$ হওয়ায় সবচেয়ে বড় eigenvalue $\lambda_1 = 4$-এর eigenvector $v_1$ বেছে নেওয়া হয়। এটিই আমাদের Principal Component।

$$W = \begin{bmatrix} 0.707 \\ 
0.707 \end{bmatrix}$$

---

## ধাপ ৫: Projection

Centered data $X_c$-কে projection matrix $W$ দিয়ে গুণ করি:

$$X_{\text{proj}} = X_c \cdot W = \begin{bmatrix} -1 & -1 \\ 
1 & 1 \end{bmatrix} \begin{bmatrix} 0.707 \\ 
0.707 \end{bmatrix}$$

$$= \begin{bmatrix} (-1)(0.707) + (-1)(0.707) \\ 
(1)(0.707) + (1)(0.707) \end{bmatrix} = \begin{bmatrix} -1.414 \\ 
1.414 \end{bmatrix}$$

---

## ফলাফল

| | আগে (2D) | পরে (1D) |
|---|---|---|
| Sample 1 | $[1, \ 2]$ | $-1.414$ |
| Sample 2 | $[3, \ 4]$ | $1.414$ |

$3 \times 2$ matrix থেকে $2 \times 1$ matrix-এ পরিণত হয়েছে। যেহেতু $\lambda_2 = 0$, দ্বিতীয় dimension বাদ দেওয়ায় ডেটার কোনো তথ্য হারায়নি।
