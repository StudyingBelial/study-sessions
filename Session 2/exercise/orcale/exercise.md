# From Lines to Folds: Shallow Networks on the Diabetes Dataset

Last week you fit a **linear regression** model to the diabetes dataset and recorded its test-set performance. This week you'll fit a **shallow neural network** — a Multi-Layer Perceptron (MLP) with one hidden layer of ReLU units — to the *same* dataset, using the *same* train/test split, and compare it head-to-head against your linear baseline.

The question you're really answering: does letting the model bend (via ReLU folds) actually help here — and if so, how much bending is the right amount?

---

## Setup

Use the same data-loading and train/test split code you used for last week's linear regression exercise, so your comparison is apples-to-apples. If you saved your linear regression test MSE and R² from last week, have them on hand — you'll need them below.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # use the same random_state you used last week
)
```

---

## Part 1: Fit a Single Shallow Network

Fit an `MLPRegressor` with **one hidden layer of 10 ReLU units** on the training data.

```python
mlp = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='relu',
    max_iter=5000,
    random_state=42
)

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Tasks

1. Report the test MSE and R² for this model.
2. Place these side by side with your linear regression numbers from last week in a small table (in a markdown cell or comment):

| Model | Test MSE | Test R² |
|---|---|---|
| Linear Regression | | |
| MLP (10 hidden units) | | |

3. In 2–3 sentences: did the shallow network beat linear regression? Why might it **not** obviously win, given that the diabetes dataset only has 442 samples and 10 features?

---

## Part 2: How Many Hidden Units Is Enough?

The universal approximation idea says more hidden units → more flexibility → closer fit. Let's test that directly, and find out whether "closer fit" on training data actually means "better" on test data.

```python
hidden_sizes = [1, 3, 10, 50, 100]
train_errors = []
test_errors = []

for h in hidden_sizes:
    model = MLPRegressor(
        hidden_layer_sizes=(h,),
        activation='relu',
        max_iter=5000,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

plt.plot(hidden_sizes, train_errors, marker='o', label='Train MSE')
plt.plot(hidden_sizes, test_errors, marker='o', label='Test MSE')
plt.xlabel('Number of Hidden Units')
plt.ylabel('MSE')
plt.xscale('log')
plt.legend()
plt.title('Train vs. Test Error as Hidden Units Increase')
plt.show()
```

### Tasks

1. Run the loop and produce the plot.
2. Identify the hidden-unit count that gives the **lowest test MSE**. Is it the largest value (100), or something smaller?
3. Describe what happens to the **training** error as hidden units increase. Does it keep going down?
4. Describe what happens to the **gap** between train and test error as hidden units increase. What is this gap usually called, and what does a widening gap tell you?
5. In 3–4 sentences: reconcile this result with the universal approximation theorem. The theorem guarantees that a large-enough network *can* represent almost any function closely — so why doesn't "more hidden units" straightforwardly translate into "better test performance" here?

---

## Submission Format

Submit a single notebook (`.ipynb`) or script (`.py`) containing:
- Your code for Part 1 and Part 2
- The comparison table from Part 1
- The plot from Part 2
- Your written answers to all numbered questions above, as markdown cells or comments
