# Statistics

## 1. Introduction & Foundations

### What is Statistics?
When referring to statistics, people generally mean one of two things:
1. **The Field of Statistics:** The study and practice of collecting, organizing, analyzing, and interpreting data.
2. **Statistical Data / Summaries:** Specific numerical facts, metrics, or summary values derived from a data set.

---

### What Can Statistics Do?

#### 1. The Workflow: Question → Data → Validation
* **Formulating the Question:** First and foremost, we start with a question.
* **Collecting Data:** To answer that question, we collect data. This can involve compiling existing data sources, designing and conducting surveys, or gathering field observations. In practical scenarios, we typically leverage pre-existing datasets or combine multiple data sources.
* **Validating the Question against Data:** Once we have both the question and the data, our true work begins: determining whether statistics can accurately answer our question. We must evaluate how the data was collected and understand how it can best serve us to address our core inquiry.

#### 2. Measurability & Quantifiable Metrics
> [!IMPORTANT]
> Statistics is not just about asking and answering interesting questions; it is about answering **measurable and quantifiable** questions.

* **Example Scenario:**
  * **Data available:** Number of parks in various cities and the number of park attendees.
  * **Proposed Question:** *Does the number of parks and park attendance make city residents happy?*
  * **The Problem:** While intuitive, how do we quantify "happiness"? How do we measure it without conflating it with unaccounted confounding factors?

#### 3. Working with Proxies
* **Definition:** A **proxy** is a measurable variable used in place of a concept that cannot be directly measured.
* **Refining the Measure:** Instead of trying to measure abstract "happiness" directly, we might measure proxies such as:
  * Mental well-being metrics
  * Self-reported mood scores of park attendees vs. non-attendees
* **Inferring Insights:** A dataset might include fields like *self-satisfaction*, *mental wellness*, and *quality of life*. Combining these metrics allows us to approximate a reliable proxy for our primary question.

---

## 2. Core Branches of Statistics

### I. Descriptive Statistics
Descriptive statistics are used to summarize, organize, and describe the features of a dataset.

* **Central Tendency & Spread:** Calculating the mean, median, mode, variance, and standard deviation to understand the center and distribution of the data.
* **Observing Relationships:** Techniques such as correlation studies fall under descriptive statistics, enabling us to observe patterns and relationships within the dataset.
* **Trade-off:** Makes large amounts of data digestible and highlights the bigger picture, though at the expense of individual data point granularity.

### II. Inferential Statistics
Inferential statistics allow us to make predictions, generalize, or draw conclusions about a larger population based on a sample of data.

* **Hypothesis Testing:** Enables us to test ideas, validate hypotheses, and answer questions that extend beyond the immediate data at hand.
* **Handling Uncertainty:** Unlike descriptive statistics, inferential statistics always involve a degree of uncertainty. It tells us how *likely* an outcome is within acceptable confidence thresholds.
* **Trade-off:** Helps us make decisions and draw conclusions about data beyond our sample, at the cost of managing statistical uncertainty.

---

## 3. Summary Overview

| Branch | Primary Purpose | Key Focus | Trade-off |
| :--- | :--- | :--- | :--- |
| **Descriptive Statistics** | Summarize & Describe | Central tendency, spread, correlation | Provides a high-level view, loses individual detail |
| **Inferential Statistics** | Predict & Generalize | Hypothesis testing, estimation, decision-making | Extends findings to broader contexts, introduces uncertainty |

> **Key Takeaway:** Statistics helps us make sense of complex data and simplify patterns in the world. Descriptive statistics make data digestible, while inferential statistics empower data-driven decision-making under uncertainty.

---

## 4. Central Tendency

Central tendency is represented by summary metrics such as the **mean**, **median**, and **mode**. It provides a single representative value that describes the center of a data distribution, summarizing the overall dataset at the cost of losing detail about individual data points.

### Mean
The **mean** (arithmetic average or expected value) is the sum of all values divided by the total number of data points. It is best suited for measuring data that is symmetrically or normally distributed.

#### Formulas:
* **Population Mean ($\mu$):**
  $$\mu = \frac{\sum_{i=1}^{N} x_i}{N}$$
* **Sample Mean ($\bar{x}$):**
  $$\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$$

> [!NOTE]
> **Normal Distribution:** A symmetric, bell-shaped data distribution where most observations cluster around the central mean with equal tails on either side.
> 
> **Limitations:** In real-world datasets, the mean can be highly misleading because its value is heavily sensitive to extreme outliers.

---

### Median
The **median** is the middle value in a sorted dataset (arranged in ascending or descending order). It divides the data into two equal halves.

#### Formulas:
For a sorted dataset $x_1 \le x_2 \le \dots \le x_n$:
* **Odd number of observations ($n$ is odd):**
  $$\text{Median} = x_{\left(\frac{n+1}{2}\right)}$$
* **Even number of observations ($n$ is even):**
  $$\text{Median} = \frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2} + 1\right)}}{2}$$

> [!TIP]
> Unlike the mean, the median is **robust to outliers**. Regardless of how extreme the values at the tails are, the median always reflects the true geometric middle of the sorted dataset.

---

### Mode
The **mode** is the value (or values) that appears most frequently in a dataset.

#### Formula:
$$\text{Mode} = \arg\max_{x} f(x)$$
*(The value $x$ corresponding to the maximum frequency $f(x)$ in the sample)*

* **Usage:** Mode is particularly useful for categorical or non-numeric data, as well as large sample sizes where identifying the most common value is beneficial without distortion from extreme values.

---

### Relationship Between Mean, Median, and Mode
Comparing the positional relationship between the mean, median, and mode reveals the **skewness** (asymmetry) of the data distribution:

* **Symmetric (Normal):** $\text{Mean} = \text{Median} = \text{Mode}$
* **Right-Skewed (Positive Skew):** $\text{Mean} > \text{Median} > \text{Mode}$ (pulled right by high outliers)
* **Left-Skewed (Negative Skew):** $\text{Mean} < \text{Median} < \text{Mode}$ (pulled left by low outliers)

> [!IMPORTANT]
> For a single dataset, these three measures can yield vastly different results. Statistics can simultaneously inform and mislead depending on the context, the research question, and the choice of measure.

---

## 5. Measures of Spread

While measures of central tendency describe the center of a dataset, **measures of spread (dispersion)** quantify how spread out or clustered the data points are relative to the center. Gauging spread helps determine how accurately central values represent the overall dataset.

### Why Does Spread Matter?
Beyond machine learning and formal statistics, measures of spread are essential in real-world contexts such as income distribution, test scores, financial risk assessment, and quality control.

* **Scenario Example:** Consider two organizations—a face mask manufacturing factory and an accounting software firm.
  * Both companies employ low-wage entry roles (e.g., janitorial staff) and high-paid executives (CEOs).
  * Consequently, both companies might share similar extreme ranges, but the distribution of employee salaries between them is fundamentally different (low-wage labor vs. high-wage engineering roles).

---

### 1. Range
The **range** measures the total span of the dataset—the difference between the maximum and minimum values.

#### Formula:
$$\text{Range} = x_{\text{max}} - x_{\text{min}}$$

* **Pros:** Quick and intuitive to calculate.
* **Cons:** Highly sensitive to extreme values/outliers and fails to describe how data is distributed between the extremes.

---

### 2. Interquartile Range (IQR)
The **Interquartile Range (IQR)** measures the spread of the middle 50% of the dataset, ignoring the extreme 25% on both upper and lower ends.

#### Formulas:
$$\text{IQR} = Q_3 - Q_1$$

Where:
* $Q_1$ (First Quartile / 25th percentile) $= \text{Median of the lower half of the data}$
* $Q_3$ (Third Quartile / 75th percentile) $= \text{Median of the upper half of the data}$

> [!TIP]
> The IQR focuses purely on the central 50% of observations, making it resistant to extreme outliers. In our example, the IQR of the software firm will sit in a significantly higher salary bracket than that of the manufacturing plant.

---

## 6. Variance

Unlike the IQR, which only considers the middle 50%, **variance** measures the dispersion of all data points relative to the mean. It calculates the average of the squared deviations from the mean.

#### Formulas:
* **Population Variance ($\sigma^2$):**
  $$\sigma^2 = \frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}$$

* **Sample Variance ($s^2$ - Unbiased Estimator):**
  $$s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1}$$

> [!NOTE]
> Dividing by $n - 1$ (Bessel's correction) instead of $n$ for sample variance compensates for sampling bias, providing an unbiased estimate of the population variance.

---

## 7. Standard Deviation

Because variance squares the differences from the mean, its units are squared (e.g., $\text{dollars}^2$), making direct interpretation difficult. **Standard deviation** takes the square root of the variance, restoring the metric back to the original unit of measurement.

#### Formulas:
* **Population Standard Deviation ($\sigma$):**
  $$\sigma = \sqrt{\sigma^2} = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}}$$

* **Sample Standard Deviation ($s$):**
  $$s = \sqrt{s^2} = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1}}$$

> [!IMPORTANT]
> **Key Takeaway:** If the **mean** represents the average value of the data, the **standard deviation** represents the average distance of data points from that mean—a standard metric for spread on the original scale.


---

## 8. Data Types: Quantitative vs. Categorical Data

While numerical calculations form the bedrock of statistics, data visualization bridges the gap between raw numbers and intuitive human understanding. Visualizations make distribution shapes, clustering, and anomalies immediately perceptible.

Before selecting a visualization, we must classify data into its two fundamental types:

### I. Quantitative Data (Numerical)
Quantitative data consists of numerical measurements with an intrinsic mathematical order and **consistent spacing**.
* **Consistent Spacing:** The interval between 1 and 2 represents the exact same physical quantity/difference as the interval between 100 and 101.
* **Subtypes:**
  * **Continuous:** Measurable on an unbroken continuum (e.g., precise age like $24.7$ years, height, salary, lap times).
  * **Discrete:** Countable whole units (e.g., number of hospital visits, number of parks).
* **Applicable Metrics:** Mean, median, mode, variance, standard deviation, IQR, and range.

### II. Categorical Data (Qualitative)
Categorical data consists of values that represent distinct labels, qualities, or groups without a continuous metric scale.
* **Subtypes:**
  * **Nominal:** Unordered categories (e.g., blood type: A, B, AB, O; department: Engineering, HR, Sales).
  * **Ordinal:** Categories with a natural order, but **inconsistent spacing** between levels (e.g., survey ratings: *Poor, Fair, Good, Excellent*; education level: *High School, Bachelor's, Master's, PhD*).
* **Handling Categorical Data:** Because mathematical arithmetic (like mean or variance) cannot be performed directly on nominal labels, we analyze categorical data by aggregating occurrences into **Frequency Tables**.

---

### Quantitative vs. Categorical Comparison

| Property | Quantitative Data | Categorical Data |
| :--- | :--- | :--- |
| **Nature of Data** | Numerical measurements & counts | Qualitative labels, groups, or classifications |
| **Spacing** | Consistent interval spacing | Arbitrary or ordinal (inconsistent intervals) |
| **Mathematical Operations** | Arithmetic (Sum, Average, Variance) | Frequency counts, Proportions, Mode |
| **Primary Visualizations** | Histograms, Box Plots, Scatter Plots | Bar Charts, Frequency Tables, Pie Charts |

---

## 9. Frequency Tables, Relative Frequency & Bar Charts

When working with categorical data or discretized groups, we summarize occurrences using frequency tables.

### Frequency ($f_i$) vs. Relative Frequency ($p_i$)
* **Absolute Frequency ($f_i$):** The raw count of observations falling into category $i$.
* **Relative Frequency ($p_i$):** The proportion or percentage of the total sample size ($n$) represented by category $i$:
  $$p_i = \frac{f_i}{n} \times 100\%$$
* **Cumulative Frequency:** The running sum of frequencies up through a given class interval.

> [!TIP]
> **Bar Charts:** The standard visual representation of frequency tables. Each discrete category has its own separated bar, with bar height proportional to frequency or relative frequency (%). Unlike histograms, bars in a bar chart have spaces between them to indicate distinct, non-continuous categories.

---

## 10. Binning Quantitative Data: The Age Case Study

**Binning (Discretization)** is the process of converting a continuous quantitative variable into discrete intervals (bins).

### The Danger of Arbitrary / "Bogus" Bins
When binning continuous variables (such as age), choosing arbitrary or poorly justified intervals can severely distort the data, mask underlying variance, and lead to flawed statistical conclusions.

#### Case Study: Arbitrary Made-Up Age Bins
Consider an arbitrary binning scheme:
* `[0 – 11]`: Pre-teen
* `[12 – 14]`: Tween (Span: 3 years)
* `[15 – 49]`: "General Adults" (**Span: 35 years!**)
* `[50 – 54]`: Early Senior (Span: 5 years)
* `[55 +]`: Older

**Why Bogus Bins Are Statistically Flawed:**
1. **Artificial Mode Spikes:** The `[15–49]` bin encompasses 35 years of human life, creating a gigantic, artificial frequency peak simply because its width is 10x wider than adjacent bins.
2. **Obscured Variance:** Lumping university students (19), young professionals (28), and established parents (45) into one category obliterates crucial biological and economic differences.
3. **Distorted Hypothesis Testing:** Non-uniform and arbitrary intervals introduce severe classification bias, rendering statistical significance and tests invalid.

### Principled Binning Strategies
To avoid distortion, use established demographic standards or mathematical rules:
1. **Domain-Principled Demographic Cohorts:** Group by clear developmental/societal milestones with meaningful definitions (e.g., Child: 0–12, Teen: 13–19, Young Adult: 20–39, Middle Age: 40–59, Senior: 60+).
2. **Equal-Width Bins:** Split the range into equal numeric intervals:
   $$\text{Bin Width} = \frac{x_{\max} - x_{\min}}{k}$$
3. **Data-Driven Rules for Number of Bins ($k$):**
   * **Sturges' Rule:** $k = 1 + \log_2(n)$ (ideal for normally distributed data).
   * **Freedman-Diaconis Rule:** $\text{Bin Width} = 2 \times \frac{\text{IQR}}{\sqrt[3]{n}}$ (robust to outliers and non-normal data).

---

## 11. Histograms: Continuous Distributions vs. Discrete Bins

A **histogram** is a graphical display of data using bars of different heights, where each bar groups continuous quantitative data into intervals along an unbroken continuous number line.

### Histograms vs. Bar Charts

| Feature | Histogram | Bar Chart |
| :--- | :--- | :--- |
| **Data Type** | Continuous quantitative data | Discrete categorical / qualitative data |
| **X-Axis** | Continuous quantitative scale (unbroken intervals) | Distinct qualitative categories (no numerical continuity) |
| **Bar Spacing** | **No gaps** between bars (indicates continuous range) | **Gaps** between bars (indicates separate categories) |
| **Bar Width** | Represents interval / bin width ($[a, b)$) | Arbitrary aesthetic width |
| **Bar Height / Area** | Area represents frequency or probability density | Height represents count / frequency directly |

> [!NOTE]
> **Bin Width Impact:**
> * **Too few bins (Oversmoothing):** Hides multi-modal distributions (e.g., dual peaks in college vs. retirement communities).
> * **Too many bins (Undersmoothing / Noise):** Breaks continuous trends into jagged, noisy single-count spikes.

---

## 12. Box and Whisker Plots (Anatomy & Outlier Fences)

A **Box and Whisker Plot (Box Plot)** is a standardized visual representation of a quantitative dataset based on its **Five-Number Summary**:
1. **Minimum ($x_{\min}$):** Smallest data point within the non-outlier range.
2. **First Quartile ($Q_1$):** 25th percentile (splits lower 25% of data).
3. **Median ($Q_2$):** 50th percentile (the central dividing line).
4. **Third Quartile ($Q_3$):** 75th percentile (splits upper 25% of data).
5. **Maximum ($x_{\max}$):** Largest data point within the non-outlier range.

---

### Mathematical Definitions & Fences (John Tukey's Rule)
* **Interquartile Range (IQR):**
  $$\text{IQR} = Q_3 - Q_1$$
* **Inner Fences (Whiskers Span):**
  $$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}$$
  $$\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$
* **Whiskers:** Extend from $Q_1$ down to the smallest data point $\ge \text{Lower Fence}$, and from $Q_3$ up to the largest data point $\le \text{Upper Fence}$.
* **Outliers:** Any individual observation falling outside the inner fences ($< \text{Lower Fence}$ or $> \text{Upper Fence}$) is plotted as an individual point $(\bullet)$.

---

### Diagnosing Skewness via Box Plots
* **Symmetric:** Median line sits in the exact middle of the box; whiskers are of equal length.
* **Right-Skewed:** Median is shifted toward $Q_1$ (lower end); upper whisker is notably longer than lower whisker; outliers cluster to the right.
* **Left-Skewed:** Median is shifted toward $Q_3$ (upper end); lower whisker is notably longer than upper whisker; outliers cluster to the left.

> [!IMPORTANT]
> **Complementary Roles of Visualizations:**
> * **Histograms** reveal the detailed shape, modality (unimodal vs. bimodal), and probability density.
> * **Box Plots** provide a clean five-number summary, facilitate rapid side-by-side subgroup comparisons, and objectively isolate statistical outliers.