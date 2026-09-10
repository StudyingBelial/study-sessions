# Session 7: Theoretical Foundations — Computation Graphs, The Chain Rule & Backpropagation

A student reference guide for understanding how neural networks learn mathematically, how gradients flow backward through mathematical operations, and how individual neurons and multi-layer perceptrons (MLPs) are trained.

---

## 1. The Core Problem: How Does a Network Learn?

In supervised learning, a neural network is given inputs $\vec{x}$ and makes a prediction $\hat{y}$. We compare that prediction against the true target $y$ using a **Loss function** $L(\hat{y}, y)$, which outputs a single scalar number representing the error.

$$\text{Goal: Find the parameters (weights } w \text{ and biases } b \text{) that make } L \text{ as small as possible.}$$

To minimize $L$, we need to know:
> *"If I make a tiny adjustment to a specific weight $w$ inside the network, how will that change the final Loss $L$?"*

The mathematical tool that answers this question is the **derivative** (or **partial derivative**), denoted as:

$$\frac{\partial L}{\partial w}$$

* If $\frac{\partial L}{\partial w} > 0$: Increasing $w$ increases the loss (so we should decrease $w$).
* If $\frac{\partial L}{\partial w} < 0$: Increasing $w$ decreases the loss (so we should increase $w$).
* The magnitude $\left|\frac{\partial L}{\partial w}\right|$ tells us how sensitive the loss is to changes in $w$.

Once we calculate $\frac{\partial L}{\partial w}$ for every parameter, we update them using **Gradient Descent**:

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$

$$b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}$$

where $\eta$ is the **learning rate** (step size).

The central challenge of deep learning is: **How do we compute $\frac{\partial L}{\partial w}$ for a weight buried deep inside a network with hundreds or thousands of intermediate steps?**

The answer is **Backpropagation** using the **Chain Rule** on a **Computation Graph**.

---

## 2. What is a Computation Graph?

Every complex mathematical model—including an entire neural network—can be broken down into a sequence of elementary arithmetic operations (such as addition and multiplication).

A **Computation Graph** is a directed graph where:
* **Nodes** represent numerical values (inputs, parameters, intermediate variables, and output loss).
* **Directed Edges (Arrows)** represent the flow of values through mathematical operations ($+$, $\times$).

```
Inputs / Parameters ---> [ Operation ] ---> Output / Intermediate Value
```

### The Two Passes of Learning

1. **Forward Pass (Left to Right):**
   * Raw numbers and current parameter values are fed into the graph.
   * Operations compute values step-by-step from inputs toward the final output.
   * Produces the final prediction and the scalar **Loss** $L$.

2. **Backward Pass (Right to Left):**
   * Begins at the final Loss $L$, where the gradient of the loss with respect to itself is always $1$:
     $$\frac{\partial L}{\partial L} = 1.0$$
   * Gradients flow backward along the edges in reverse order.
   * At each node, we compute the partial derivative of the Loss with respect to that node using the **Chain Rule**.

---

## 3. The Chain Rule: The Heart of Backpropagation

### 3.1 Single-Path Chain Rule

Suppose variable $x$ affects variable $y$, and $y$ affects the final result $z$:

$$x \longrightarrow y \longrightarrow z$$

If we know:
1. How sensitive $z$ is to changes in $y$: $\frac{\partial z}{\partial y}$ (the *incoming gradient* from upstream)
2. How sensitive $y$ is to changes in $x$: $\frac{\partial y}{\partial x}$ (the *local gradient* of the operation)

Then the sensitivity of $z$ with respect to $x$ is simply their product:

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$$

$$\text{Total Gradient} = (\text{Upstream Gradient}) \times (\text{Local Gradient})$$

#### Intuitive Analogy:
* If $y$ changes twice as fast as $x$ ($\frac{\partial y}{\partial x} = 2$),
* and $z$ changes three times as fast as $y$ ($\frac{\partial z}{\partial y} = 3$),
* then $z$ changes $3 \times 2 = 6$ times as fast as $x$ ($\frac{\partial z}{\partial x} = 6$).

---

### 3.2 Multi-Path Chain Rule (Branching & Gradient Accumulation)

What happens if a single variable $x$ is used in multiple operations downstream?

```
         ↗  y₁  ↘
       x          Loss (L)
         ↘  y₂  ↗
```

Here, changing $x$ affects $L$ through **two distinct paths**: through $y_1$ and through $y_2$.

According to multivariable calculus, the total effect of $x$ on $L$ is the **sum** of the effects along each individual path:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y_1} \cdot \frac{\partial y_1}{\partial x} + \frac{\partial L}{\partial y_2} \cdot \frac{\partial y_2}{\partial x}$$

> [!IMPORTANT]
> **Gradient Accumulation Rule:**
> Whenever a node distributes its value to more than one downstream operation, the gradients flowing back from all branches must be **added together**.
>
> $$\frac{\partial L}{\partial x} = \sum_{\text{all paths } k} \frac{\partial L}{\partial y_k} \cdot \frac{\partial y_k}{\partial x}$$

---

## 4. Local Derivatives of Fundamental Operations

A neural network is built primarily from two elementary operations: **Addition** and **Multiplication**. Let's examine how gradients propagate backward through each of them.

---

### 4.1 The Addition Node ($+$)

Let $z = x + y$, where $z$ flows downstream into a final Loss $L$.

```
x ──┐
    (+) ──> z ──> ... ──> Loss (L)
y ──┘
```

#### Step 1: Compute Local Derivatives
* How does $z$ change when $x$ changes?
  $$\frac{\partial z}{\partial x} = \frac{\partial}{\partial x}(x + y) = 1$$
* How does $z$ change when $y$ changes?
  $$\frac{\partial z}{\partial y} = \frac{\partial}{\partial y}(x + y) = 1$$

#### Step 2: Apply the Chain Rule
Assume we already received the upstream gradient $\frac{\partial L}{\partial z}$ from the rest of the network:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x} = \frac{\partial L}{\partial z} \cdot 1 = \frac{\partial L}{\partial z}$$

$$\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y} = \frac{\partial L}{\partial z} \cdot 1 = \frac{\partial L}{\partial z}$$

> [!TIP]
> **Addition is a Gradient Distributor (Pass-Through):**
> An addition node takes the incoming upstream gradient $\frac{\partial L}{\partial z}$ and distributes it **unchanged** to both inputs $x$ and $y$.

---

### 4.2 The Multiplication Node ($\times$)

Let $z = x \cdot y$, where $z$ flows downstream into a final Loss $L$.

```
x ──┐
    (×) ──> z ──> ... ──> Loss (L)
y ──┘
```

#### Step 1: Compute Local Derivatives
* How does $z$ change when $x$ changes?
  $$\frac{\partial z}{\partial x} = \frac{\partial}{\partial x}(x \cdot y) = y$$
* How does $z$ change when $y$ changes?
  $$\frac{\partial z}{\partial y} = \frac{\partial}{\partial y}(x \cdot y) = x$$

#### Step 2: Apply the Chain Rule
Assume upstream gradient $\frac{\partial L}{\partial z}$ is known:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x} = \frac{\partial L}{\partial z} \cdot y$$

$$\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y} = \frac{\partial L}{\partial z} \cdot x$$

> [!TIP]
> **Multiplication is a Gradient Switcher and Scaler:**
> The gradient for input $x$ is the upstream gradient scaled by the value of the **other input** $y$.
> Similarly, the gradient for input $y$ is the upstream gradient scaled by the value of the **other input** $x$.

---

## 5. Fully Worked Example by Hand

Let's trace a complete computation graph by hand with real numbers.

### Forward Equations
Let the inputs and parameters be:
* $a = 2.0$
* $b = -3.0$
* $c = 10.0$

Operations:
1. $d = a \cdot b = 2.0 \cdot (-3.0) = -6.0$
2. $L = d + c = -6.0 + 10.0 = 4.0$

Here, $L$ is our final output value.

```
a (2.0)  ──┐
           (×) ──> d (-6.0) ──┐
b (-3.0) ──┘                  (+) ──> L (4.0)
c (10.0) ─────────────────────┘
```

---

### Backward Pass (Step-by-Step)

#### Step 1: Gradient at the Output ($L$)
Every backward pass starts with the output with respect to itself:
$$\frac{\partial L}{\partial L} = 1.0$$

#### Step 2: Gradients into the Addition Node ($L = d + c$)
Using the addition rule:
$$\frac{\partial L}{\partial d} = \frac{\partial L}{\partial L} \cdot \frac{\partial L}{\partial d} = 1.0 \cdot 1 = 1.0$$

$$\frac{\partial L}{\partial c} = \frac{\partial L}{\partial L} \cdot \frac{\partial L}{\partial c} = 1.0 \cdot 1 = 1.0$$

#### Step 3: Gradients into the Multiplication Node ($d = a \cdot b$)
Using the multiplication rule:
* The upstream gradient arriving at $d$ is $\frac{\partial L}{\partial d} = 1.0$.
* For $a$:
  $$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial a} = \frac{\partial L}{\partial d} \cdot b = 1.0 \cdot (-3.0) = -3.0$$
* For $b$:
  $$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial b} = \frac{\partial L}{\partial d} \cdot a = 1.0 \cdot 2.0 = 2.0$$

---

### Summary of Forward and Backward Values

| Node | Forward Value | Upstream Gradient ($\frac{\partial L}{\partial \cdot}$) | Interpretation |
| :--- | :--- | :--- | :--- |
| **$L$** | $4.0$ | $1.0$ | Base sensitivity |
| **$c$** | $10.0$ | $+1.0$ | Increasing $c$ by $\epsilon$ increases $L$ by $1\epsilon$ |
| **$d$** | $-6.0$ | $+1.0$ | Increasing $d$ by $\epsilon$ increases $L$ by $1\epsilon$ |
| **$a$** | $2.0$ | $-3.0$ | Increasing $a$ by $\epsilon$ **decreases** $L$ by $3\epsilon$ |
| **$b$** | $-3.0$ | $+2.0$ | Increasing $b$ by $\epsilon$ **increases** $L$ by $2\epsilon$ |

#### Verification:
If we nudge $a$ slightly from $2.0$ to $2.001$ ($\Delta a = +0.001$):
* New $d = 2.001 \cdot (-3.0) = -6.003$
* New $L = -6.003 + 10.0 = 3.997$
* Change in $L$: $\Delta L = 3.997 - 4.0 = -0.003$
* Ratio: $\frac{\Delta L}{\Delta a} = \frac{-0.003}{0.001} = -3.0$ (Matches our gradient $\frac{\partial L}{\partial a} = -3.0$ exactly!)

---

## 6. Building Neural Networks from Scratch

Now that we understand how $+$, $\times$, and the chain rule operate on individual numbers, we can scale this concept up to complete neural networks.

---

### 6.1 The Single Neuron

An artificial neuron takes an input vector $\vec{x} = [x_1, x_2, \dots, x_n]$, pairs each input with a learnable weight $\vec{w} = [w_1, w_2, \dots, w_n]$, and adds a learnable bias $b$.

The output of a linear neuron is:

$$z = \sum_{i=1}^{n} (w_i \cdot x_i) + b = (w_1 x_1 + w_2 x_2 + \dots + w_n x_n) + b$$

```
x₁ ──(× w₁)──┐
x₂ ──(× w₂)──┼──(+)──(+) ──> z (Output)
              │    │
xₙ ──(× wₙ)──┘    b
```

Notice that a neuron is simply a collection of **multiplication** nodes followed by a series of **addition** nodes.

#### Backward Pass for a Single Neuron
When an upstream gradient $\frac{\partial L}{\partial z}$ arrives at neuron output $z$:

1. **Gradient with respect to weights ($w_i$):**
   $$\frac{\partial z}{\partial w_i} = x_i \implies \frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial z} \cdot x_i$$
   *Meaning: The gradient of a weight is the incoming error signal multiplied by the corresponding input value.*

2. **Gradient with respect to bias ($b$):**
   $$\frac{\partial z}{\partial b} = 1 \implies \frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot 1 = \frac{\partial L}{\partial z}$$
   *Meaning: The gradient of the bias directly receives the incoming error signal.*

3. **Gradient passed back to input ($x_i$):**
   $$\frac{\partial z}{\partial x_i} = w_i \implies \frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial z} \cdot w_i$$
   *Meaning: The gradient passed back to previous layers is scaled by the weight connecting them.*

---

### 6.2 The Neural Network Layer

A **Layer** is simply a collection of multiple neurons operating in parallel on the **same input vector** $\vec{x}$.

If an input has $n_{in}$ dimensions and the layer has $n_{out}$ neurons:
* Neuron 1 computes: $y_1 = \sum_{i=1}^{n_{in}} w_{1, i} x_i + b_1$
* Neuron 2 computes: $y_2 = \sum_{i=1}^{n_{in}} w_{2, i} x_i + b_2$
* ...
* Neuron $j$ computes: $y_j = \sum_{i=1}^{n_{in}} w_{j, i} x_i + b_j$

The output of the layer is the vector $\vec{y} = [y_1, y_2, \dots, y_{n_{out}}]$.

```
Inputs (x)          Layer (Neurons)              Outputs (y)
   x₁  ─────────┬────────> [ Neuron 1 ] ─────────> y₁
                │ 
   x₂  ─────────┼────────> [ Neuron 2 ] ─────────> y₂
                │ 
   ... ─────────┴────────> [ Neuron n_out ] ─────> y_nout
```

#### Backward Pass Across a Layer
In the backward pass:
1. Each neuron $j$ calculates the gradients for its own weights $w_{j, i}$ and its own bias $b_j$.
2. Notice that input $x_i$ is distributed to **every single neuron** in the layer!
3. Therefore, by the **Multi-Path Chain Rule**, the total gradient for $x_i$ is the sum of gradients contributed by all neurons in that layer:

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{n_{out}} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i} = \sum_{j=1}^{n_{out}} \frac{\partial L}{\partial y_j} \cdot w_{j, i}$$

This combined gradient vector $\frac{\partial L}{\partial \vec{x}}$ is what gets handed backward to the preceding layer.

---

### 6.3 Multi-Layer Perceptron (MLP)

An **MLP** is formed by stacking multiple layers sequentially:

$$\vec{x} \longrightarrow \text{Layer 1} \longrightarrow \vec{h}_1 \longrightarrow \text{Layer 2} \longrightarrow \vec{h}_2 \longrightarrow \dots \longrightarrow \text{Layer } K \longrightarrow \hat{y} \longrightarrow \text{Loss } L$$

* The output of Layer 1 becomes the input to Layer 2.
* The output of Layer 2 becomes the input to Layer 3, and so on.
* The final layer produces predictions $\hat{y}$ that are compared with true targets $y$ to compute the scalar Loss $L$.

```
Input Layer         Hidden Layer(s)          Output Layer          Loss
  [ x₁ ] ───\       /─── [ h₁ ] ───\        /─── [ ŷ ] ─────────> [ Loss L ]
             ======                 ======
  [ x₂ ] ───/       \─── [ h₂ ] ───/        \
```

#### The Full Backpropagation Cycle Through an MLP

```
   FORWARD PASS: Computing predictions & error (Left-to-Right)
══════════════════════════════════════════════════════════════════>
   Inputs ──> Layer 1 ──> Layer 2 ──> ... ──> Output ──> Loss (L)
<══════════════════════════════════════════════════════════════════
   BACKWARD PASS: Propagating gradients via Chain Rule (Right-to-Left)
```

1. **Initialize:** Set $\frac{\partial L}{\partial L} = 1.0$.
2. **Propagate back to Output Layer:** Compute gradients for weights and biases of the output layer, and compute $\frac{\partial L}{\partial \vec{h}_{last}}$.
3. **Propagate through Hidden Layers:** Flow backwards layer by layer, computing gradients for each layer's parameters and passing gradients to previous layer activations.
4. **Update Parameters:** Apply gradient descent to all parameters across the entire network:
   $$w \leftarrow w - \eta \frac{\partial L}{\partial w}, \quad b \leftarrow b - \eta \frac{\partial L}{\partial b}$$
5. **Zero Gradients:** Clear accumulated gradients before the next forward pass so gradients do not bleed into the next training step.

---

## 7. Execution Order: Why Topological Sort is Required

In a complex computation graph, a node may depend on values that were computed several steps earlier.

When computing gradients backward:
> [!CAUTION]
> **Order Dependency:**
> You **cannot** compute the gradient of a node $x$ until you have already computed the gradients of **all** nodes that consume $x$ downstream!

To guarantee that we never evaluate a node's gradient prematurely, we use **Topological Ordering** of a Directed Acyclic Graph (DAG):

1. **Topological Order:** An ordering of nodes such that for every directed edge from node $A$ to node $B$, $A$ comes before $B$.
2. **Forward Evaluation Order:** Traverse nodes in **topological order** (inputs first, final loss last).
3. **Backward Evaluation Order:** Traverse nodes in **reverse topological order** (final loss first, inputs/parameters last).

This ensures every node receives its full upstream gradient from all its consumers before it calculates and passes its own local gradients backward.

---

## 8. Summary Reference Sheet

| Concept | Mathematical Notation | Meaning in Plain English |
| :--- | :--- | :--- |
| **Scalar Value** | $v$ | A single number stored at a node in the graph. |
| **Forward Pass** | $z = f(x, y)$ | Calculating outputs by flowing left-to-right through operations. |
| **Loss** | $L$ | A single number measuring how wrong the network's prediction is. |
| **Gradient** | $\frac{\partial L}{\partial v}$ | Sensitivity: "If $v$ increases by a tiny amount, how much does $L$ change?" |
| **Base Condition** | $\frac{\partial L}{\partial L} = 1.0$ | The starting point of backpropagation at the final output. |
| **Chain Rule (Single)** | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}$ | Total sensitivity = (Upstream gradient) $\times$ (Local derivative). |
| **Addition ($z = x + y$)** | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z}$, $\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z}$ | Gradient pass-through: distributes incoming gradient equally. |
| **Multiplication ($z = x \cdot y$)** | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot y$, $\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \cdot x$ | Gradient switcher: multiplies incoming gradient by the *other* operand. |
| **Branching / Multi-path** | $\frac{\partial L}{\partial x} = \sum_k \frac{\partial L}{\partial y_k} \frac{\partial y_k}{\partial x}$ | Gradients from multiple downstream consumers **add together**. |
| **Neuron Weights** | $\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial z} \cdot x_i$ | Weight gradient is incoming error times the input feature. |
| **Neuron Bias** | $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}$ | Bias gradient is directly the incoming error. |
| **Reverse Topological Order** | Reverse DAG sequence | The exact order to visit nodes so gradients are computed correctly without missing dependencies. |
| **Parameter Update** | $w \leftarrow w - \eta \frac{\partial L}{\partial w}$ | Move parameters opposite to the gradient to decrease the Loss. |
