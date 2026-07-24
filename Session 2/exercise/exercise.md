# The Oracle's Hidden Layer

The vault behind this message will only open for someone who can think the way the Oracle thinks — in bent lines and folded planes. Perform exactly one forward pass through the shallow network described below to reveal the unlock code.

## Network Architecture

- **1 input**, $x$
- **1 hidden layer** with **3 neurons**, each using the ReLU activation function
- **1 output**, $y$, formed from a weighted sum of the hidden activations

## Given Values

**Input:**

$$x = 2$$

**Parameters:**

$$\phi_0 = 0.25, \quad \phi_1 = 0.8, \quad \phi_2 = -0.3, \quad \phi_3 = 1.2$$

$$\theta_{10} = 0.5, \quad \theta_{11} = 1.0$$

$$\theta_{20} = -1.0, \quad \theta_{21} = 1.5$$

$$\theta_{30} = 2.0, \quad \theta_{31} = -0.5$$

Use the full forward-pass calculation to solve this:

### 1. Pre-activations

$$z_1 = \theta_{10} + \theta_{11} x$$

$$z_2 = \theta_{20} + \theta_{21} x$$

$$z_3 = \theta_{30} + \theta_{31} x$$

### 2. Activation Function (ReLU)

$$a[z] = \text{ReLU}[z] = \begin{cases} 0 & \text{if } z < 0 \\ z & \text{if } z \ge 0 \end{cases}$$

### 3. Hidden Activations

$$h_1 = a[z_1], \quad h_2 = a[z_2], \quad h_3 = a[z_3]$$

### 4. Output Equation

$$y = \phi_0 + \phi_1 h_1 + \phi_2 h_2 + \phi_3 h_3$$

### 5. Final Value

Compute the numerical value of $y$ using the steps above.

## Password Format

Write your computed $h_1$, $h_2$, $h_3$, and $y$ values to exactly two decimal places, separated by commas and no spaces (e.g., 1.00,2.00,3.00,4.00).
