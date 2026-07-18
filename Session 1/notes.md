# Supervised Machine Learning

## The Model

$$y = f[x, \phi]$$

where **y** is the label (output), **x** is the feature (input), and **φ** is the set of model parameters.

Training a model means finding the set of parameters φ that best predicts y.

How do we know what "best" means? We use a **cost function** (or loss function) to calculate how wrong a prediction is. Loss is a quantifiable number telling us how far off the model's prediction is — our goal is to minimize it. We write the loss function as L[φ].

$$\hat{\phi} = \arg\min_{\phi} \, L[\phi]$$

(find the set of φ that minimizes L[φ])

We test how well our model is doing on a separate set of **test examples**, checking how well the model generalizes to unseen data. Loss is calculated during the training run, but the real goal is a minimum loss on the test run — this indicates the model has learned the underlying patterns in the data, rather than simply memorizing the training examples.

## Linear Regression

Formalizing y = f[x, φ], linear regression is represented as:

$$y = \phi_0 + \phi_1 x$$

For simplicity, we'll use the more common notation:

$$y = b + wx$$

where **w** is the weight and **b** is the bias.

## Loss

We want $\hat{\phi} = \arg\min_{\phi} L[\phi]$ — the best w and b — but how do we actually measure whether one set of parameters is better than another? We use the loss function L[φ]:

$$\text{individual loss}_i = f[x_i, \phi] - y_i$$

where *i* indexes each individual data point.

Summing over all *I* data points:

$$L[\phi] = \sum_{i=1}^{I} \big(f[x_i, \phi] - y_i\big)^2$$

We square each individual loss to make it positive and to penalize larger errors more heavily, then sum them up. This is commonly called **Mean Squared Error (MSE)**.

Substituting our linear regression model, the loss becomes:

$$L[\phi] = \sum_{i=1}^{I} \big(b + w x_i - y_i\big)^2$$

The goal is to find the φ that gives the minimum loss for a given w and b:

$$\hat{\phi} = \arg\min_{\phi} \, L[\phi] = \arg\min_{\phi} \, \sum_{i=1}^{I} \big(b + w x_i - y_i\big)^2$$

## Gradient Descent

We now have a way to quantify how good a set of parameters (w, b) is — but how do we actually calculate w and b, rather than simply guessing? For that, we use **gradient descent**.

Let α (alpha) be the learning rate:

$$w \leftarrow w - \alpha \frac{\partial L[\phi]}{\partial w}$$

$$b \leftarrow b - \alpha \frac{\partial L[\phi]}{\partial b}$$

Breaking this down, recall:

$$L[\phi] = \sum_{i=1}^{I} \big(b + w x_i - y_i\big)^2$$

Taking the partial derivatives:

$$\frac{\partial L[\phi]}{\partial w} = \sum_{i=1}^{I} 2\big(b + w x_i - y_i\big)\, x_i$$

$$\frac{\partial L[\phi]}{\partial b} = \sum_{i=1}^{I} 2\big(b + w x_i - y_i\big)$$