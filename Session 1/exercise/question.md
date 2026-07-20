# The Oracle's Password

The attached zip file contains your core resources. To unlock it, you must perform exactly one iteration of gradient descent over the entire dataset provided below.

## Dataset

$(x_1 = 1, y_1 = 3)$

$(x_2 = 2, y_2 = 5)$

$(x_3 = 3, y_3 = 7)$

$(x_4 = 4, y_4 = 9)$

## Initial Parameters

$$w = 0$$

$$b = 0$$

$$\alpha = 0.02$$

Use the full formula of gradient descent to calculate this:

### 1. Loss Function

$$
\begin{aligned}
\mathcal{L}[\phi]
&= \sum_{i=1}^{I} \Big( \big(b + w x_i\big) - y_i \Big)^2
\end{aligned}
$$

### 2. Gradient with Respect to $w$

$$
\frac{\partial \mathcal{L}[\phi]}{\partial w}
= 2 \sum_{i=1}^{I} x_i \Big(b + w x_i - y_i\Big)
$$

### 3. Gradient with Respect to $b$

$$
\frac{\partial \mathcal{L}[\phi]}{\partial b}
= 2 \sum_{i=1}^{I} \Big(b + w x_i - y_i\Big)
$$

### 4. Update Rule for $w$

$$
w \leftarrow w - \alpha \frac{\partial \mathcal{L}[\phi]}{\partial w}
$$

### 5. Update Rule for $b$

$$
b \leftarrow b - \alpha \frac{\partial \mathcal{L}[\phi]}{\partial b}
$$

## Password Format

Write your updated $w$ and $b$ to exactly two decimal places, separated by a comma and no spaces (e.g., 1.23,4.56).