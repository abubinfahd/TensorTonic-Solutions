# PCA (Principal Component Analysis) - Step-by-Step Guide

ধরে নিই, আমরা ৩ জন ইউজারের ২টি ফিচারের (যেমন: ওয়েবসাইটে কাটানো সময় এবং ক্লিক সংখ্যা) ডেটা কালেক্ট করেছি।

---

## ধাপ ১: Raw Data Matrix ($X$)

আমাদের কাছে ৩টি স্যাম্পল ($n=3$) এবং ২টি ফিচার ($d=2$) আছে।

$$X = \begin{bmatrix} 2 & 4 \\ 
4 & 2 \\ 
6 & 6 \end{bmatrix}$$

- **প্রথম কলাম (Feature 1):** $2, 4, 6$
- **দ্বিতীয় কলাম (Feature 2):** $4, 2, 6$

---

## ধাপ ২: Mean বের করা ($\mu$)

প্রতিটি ফিচারের গড় বের করি:

$$\mu_1 = \frac{2 + 4 + 6}{3} = 4, \qquad \mu_2 = \frac{4 + 2 + 6}{3} = 4$$

$$\mu = \begin{bmatrix} 4 & 4 \end{bmatrix}$$

---

## ধাপ ৩: Data Centering ($X_c$)

প্রতিটি ভ্যালু থেকে তার কলামের গড় বিয়োগ করি:

$$X_c = X - \mu = \begin{bmatrix} 2-4 & 4-4 \\ 
4-4 & 2-4 \\ 
6-4 & 6-4 
\end{bmatrix} = \begin{bmatrix} -2 & 0 \\ 
0 & -2 \\ 
2 & 2 \end{bmatrix}$$

এখন ডেটার নতুন গড় $0$।

---

## ধাপ ৪: Covariance Matrix ($C$)

$$C = \frac{X_c^T X_c}{n - 1}$$

যেহেতু $n = 3$, তাই $n - 1 = 2$।

**Step 4a:** $X_c^T$ বের করি:

$$X_c^T = \begin{bmatrix} -2 & 0 & 2 \\ 
0 & -2 & 2 \end{bmatrix}$$

**Step 4b:** $X_c^T X_c$ ক্যালকুলেট করি:

$$X_c^T X_c = \begin{bmatrix} -2 & 0 & 2 \\ 
0 & -2 & 2 \end{bmatrix} \begin{bmatrix} -2 & 0 \\ 
0 & -2 \\ 
2 & 2 \end{bmatrix}$$

$$= \begin{bmatrix} (-2)(-2)+(0)(0)+(2)(2) & (-2)(0)+(0)(-2)+(2)(2) \\ 
(0)(-2)+(-2)(0)+(2)(2) & (0)(0)+(-2)(-2)+(2)(2) \end{bmatrix} = \begin{bmatrix} 8 & 4 \\ 
4 & 8 \end{bmatrix}$$

**Step 4c:** $n-1 = 2$ দিয়ে ভাগ করি:

$$C = \frac{1}{2} \begin{bmatrix} 8 & 4 \\ 
4 & 8 \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 
2 & 4 \end{bmatrix}$$

---

## ধাপ ৫: Eigenvalues ($\lambda$)

Characteristic equation: $\det(C - \lambda I) = 0$

$$\det \begin{bmatrix} 4 - \lambda & 2 \\ 
2 & 4 - \lambda \end{bmatrix} = 0$$

$$(4 - \lambda)^2 - 4 = 0$$

$$\lambda^2 - 8\lambda + 12 = 0$$

$$(\lambda - 6)(\lambda - 2) = 0$$

$$\lambda_1 = 6 \quad (\text{PC1}), \qquad \lambda_2 = 2$$

---

## ধাপ ৬: Eigenvector ($v_1$) for PC1

$\lambda_1 = 6$ ব্যবহার করে $(C - \lambda_1 I)v = 0$ সলভ করি:

$$\begin{bmatrix} -2 & 2 \\ 
2 & -2 \end{bmatrix} \begin{bmatrix} x \\ 
y \end{bmatrix} = \begin{bmatrix} 0 \\ 
0 \end{bmatrix}$$

এখান থেকে: $-2x + 2y = 0 \implies x = y$

তাহলে eigenvector: $\begin{bmatrix} 1 \\ 
1 \end{bmatrix}$

**Normalize** করি (magnitude $= \sqrt{1^2 + 1^2} = \sqrt{2}$):

$$W = \begin{bmatrix} \dfrac{1}{\sqrt{2}} \\ 
\dfrac{1}{\sqrt{2}} \end{bmatrix} \approx \begin{bmatrix} 0.707 \\ 
0.707 \end{bmatrix}$$

---

## ধাপ ৭: Projection (2D to 1D)

$$X_{\text{proj}} = X_c \cdot W = \begin{bmatrix} -2 & 0 \\ 
0 & -2 \\ 
2 & 2 \end{bmatrix} \begin{bmatrix} 0.707 \\ 
0.707 \end{bmatrix}$$

| Sample | Calculation | Result |
|--------|-------------|--------|
| Sample 1 | $(-2)(0.707) + (0)(0.707)$ | $-1.414$ |
| Sample 2 | $(0)(0.707) + (-2)(0.707)$ | $-1.414$ |
| Sample 3 | $(2)(0.707) + (2)(0.707)$ | $2.828$ |

$$X_{\text{proj}} = \begin{bmatrix} -1.414 \\ 
-1.414 \\ 
2.828 \end{bmatrix}$$

**ফলাফল:** $3 \times 2$ matrix থেকে $3 \times 1$ matrix — dimensionality সফলভাবে কমানো হয়েছে।
