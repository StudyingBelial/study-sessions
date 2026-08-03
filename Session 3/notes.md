# Session 3: Core Statistical Distributions & Summary Metrics

## 1. Normal (Gaussian) Distribution

A symmetrical, bell-shaped continuous probability distribution.

- The y-axis represents the **probability density** for a given x-value — not the probability itself. For continuous distributions, probability comes from the *area* under a section of the curve, not the height at one point.
- Why the shape of a distribution matters:
  1. **Central tendency** — where does the center of the curve sit (the average)?
  2. **Spread** — how wide is the curve? Narrow = tall and tightly clustered. Wide = short and spread out.

The normal distribution shows up constantly in nature, and there's a specific reason for that: the **Central Limit Theorem** (see below). That's what makes it so central to statistics.

### Empirical Rule (68-95-99.7 Rule) & Parameter Effects
- **68.27%** of data falls within $\mu \pm 1\sigma$
- **95.45%** of data falls within $\mu \pm 2\sigma$
- **99.73%** of data falls within $\mu \pm 3\sigma$

![Normal Distribution and Empirical Rule](images/normal_distribution.png)

---

## 2. Mean, Variance, and Standard Deviation

To describe a dataset — or draw its histogram / fit a normal curve to it — we need the mean, the variance, and the standard deviation.

### Population Mean ($\mu$)

The average of every value in the entire population. This is the value the normal curve gets centered on.

$$\mu = \frac{\sum x}{N}$$

In reality this rarely gets calculated — populations are usually far too large to measure completely. Instead, we estimate the mean from a smaller sample.

**$\bar{x}$** ("x-bar") is the *sample mean* — an estimate of $\mu$. $\bar{x}$ isn't the same as $\mu$, but the larger the sample, the closer $\bar{x}$ gets to it.

### Population Variance and Standard Deviation

To describe how wide the curve is, we use the variance and standard deviation — both measure spread around the mean.

- **Population variance**: 
  $$\sigma^2 = \frac{\sum (x - \mu)^2}{N}$$
  Variance is in *squared* units (e.g. $\text{cm}^2$ if $x$ is in cm), so it can't be plotted directly on the same axis as the data — the units don't match. Taking the square root brings it back to the original units:

- **Population standard deviation**: 
  $$\sigma = \sqrt{\frac{\sum (x - \mu)^2}{N}}$$

### Sample (Estimated) Variance and Standard Deviation

We almost never have the true population mean, so in practice we use estimates from a sample instead:

- **Sample variance**: 
  $$s^2 = \frac{\sum (x - \bar{x})^2}{n - 1}$$

Notice the denominator is **$(n - 1)$**, not $n$ — this is **Bessel's correction**. Here's the intuition: $\bar{x}$ is calculated from the very same sample being measured, and it's mathematically the value that *minimizes* the sum of squared deviations for that sample. So the squared deviations from $\bar{x}$ are, on average, a little smaller than the squared deviations from the true (unknown) population mean $\mu$ would be. Dividing by $n$ would systematically underestimate the true variance — dividing by $(n - 1)$ corrects for that.

- **Sample standard deviation**: 
  $$s = \sqrt{s^2}$$

![Bessel's Correction Simulation & Mean Convergence](images/variance_sd_bessel.png)

---

## 3. Mean Absolute Deviation (MAD)

The average distance of each point from the mean, without squaring:

$$\text{MAD} = \frac{\sum |x - \mu|}{n}$$

MAD gives a linear measure of dispersion that does not overweight extreme values as heavily as variance and standard deviation do.

![MAD vs Standard Deviation Visualization](images/mad_vs_sd.png)

---

## 4. Coefficient of Variation (CV)

$$\text{CV} = \left(\frac{\text{SD}}{\text{Mean}}\right) \times 100\%$$

CV expresses spread *relative to* the size of the mean — which lets you compare variability across datasets that have different units or different scales, something SD alone can't do.

Why isn't standard deviation enough on its own? Suppose you have two datasets:

| Dataset | Mean | SD | Relative Calculation | CV (%) |
|---|---|---|---|---|
| **A** | 10 | 2 | $2 / 10 = 0.20$ | **20%** |
| **B** | 1000 | 2 | $2 / 1000 = 0.002$ | **0.2%** |

A spread of 2 is huge relative to a mean of 10, but almost nothing relative to a mean of 1000. Same SD, very different real-world variability — that's exactly what CV captures.

![Coefficient of Variation Comparison](images/coefficient_of_variation.png)

---

## 5. Range

$$\text{Range} = \text{Max} - \text{Min}$$

The simplest possible measure of spread, measuring the total distance between the largest and smallest values.

![Range of a Dataset](images/range_visualization.png)

---

## 6. Central Limit Theorem (CLT)

The sampling distribution of the mean approaches a normal distribution as sample size grows — regardless of the shape of the original population's distribution. This holds as long as the sample is reasonably large (a common rule of thumb is **$n \ge 30$**) and the population has finite variance.

Why does this matter? We usually can't know the true underlying distribution our data comes from — the CLT makes that irrelevant. Because the sample mean behaves approximately normally regardless, we can apply normal-theory tools to it: **confidence intervals, t-tests, ANOVA**, and more.

*(Note: technically this describes the theoretical sampling distribution — what you'd see if you repeated the sampling process many times. In practice we usually only draw one sample, but the CLT is what justifies treating that single sample's mean as approximately normal.)*

![Central Limit Theorem Demonstration](images/central_limit_theorem.png)

---

## 7. Median, Quantiles, and Interquartile Range (IQR)

### Median
The value where 50% of the data sits above and 50% sits below — the center of the ordered list of values. It's the quantile that splits the data into two equal-sized halves: the **0.5 quantile**.

Quantiles are just cut points that divide data into equally sized groups. Percentiles are quantiles that divide the data into 100 equally sized groups.

### Interquartile Range (IQR)
The IQR uses three quantiles spaced 0.25 apart — 0.25, 0.5, and 0.75 — dividing the data into quarters.

One common way to find the **rank/position** of Q1 and Q3 in the ordered dataset:
- **Position of Q1** = $1 \times (N + 1) / 4$
- **Position of Q3** = $3 \times (N + 1) / 4$

*(Important: this gives the position in the sorted list, not the value itself — if the position isn't a whole number, interpolate between the two nearest ranked values.)*

$$\text{IQR} = Q3 - Q1$$

### Outlier Fences
To flag outliers, extend **$1.5 \times \text{IQR}$** out from each quartile to set the "fences":

- **Lower fence** = $Q1 - 1.5 \times \text{IQR}$
- **Upper fence** = $Q3 + 1.5 \times \text{IQR}$

Any value beyond these fences is a potential outlier.

The IQR measures the spread of the central 50% of the data and, unlike range or SD, is resistant to extreme values. Extending $1.5 \times \text{IQR}$ beyond Q1 and Q3 defines a reasonable range for "normal" observations — values beyond it are unusually far from the bulk of the distribution relative to its typical spread.

This is the standard tool for **data preprocessing and anomaly detection**: finding outliers, catching data-entry errors, and avoiding misleading patterns caused by extreme values.

![IQR, Quartiles, and Outlier Box Plot](images/iqr_box_outliers.png)

---

## 8. Mode

The value (or values) that appear most frequently in the data.

| Type | Definition | Example Data Set | Mode(s) |
|---|---|---|---|
| **Unimodal** | Exactly one mode | {1, 2, 2, 3, 6, 7, 7, 7, 8, 9} | 7 |
| **Bimodal** | Two modes | {1, 1, 1, 3, 4, 4, 6, 6, 6} | 1 and 6 |
| **Trimodal** | Three modes | {2, 2, 2, 3, 4, 4, 6, 6, 6, 7, 9, 9, 9} | 2, 6, and 9 |
| **Multimodal** | Four or more modes | {1, 1, 1, 3, 4, 4, 6, 6, 6, 7, 9, 9, 9, 11, 11, 11} | 1, 6, 9, and 11 |

![Distribution Modes Visual Comparison](images/mode_types.png)

---

## 9. Skewness

In a symmetrical/normal distribution: **mean = median = mode**.

Skewness measures asymmetry — how far the data deviates from that symmetric ideal.

- **Negative skew**: tail stretches left, bulk of the data sits on the right.
  $$\text{Mean} < \text{Median} < \text{Mode}$$

- **Positive skew**: tail stretches right, bulk of the data sits on the left.
  $$\text{Mode} < \text{Median} < \text{Mean}$$

**Pearson's First Coefficient of Skewness**:

$$\text{Coefficient} = \frac{\text{Mean} - \text{Mode}}{\text{SD}}$$

Skewness tells you whether the mean can be trusted as a "typical" value, or whether it's being pulled away from where most of the data actually sits — Pearson's coefficient quantifies exactly how far.

### Example — Two classes with similar average scores:
- **Class A:** 50, 51, 49, 52, 48 — almost everyone scores around 50.
- **Class B:** 5, 7, 8, 10, 70 — most students scored poorly, but one scored extremely high.

Both classes could report a similar mean, but they tell very different stories. Are the values balanced around the center, or are a few extreme values dragging the average in one direction? That's what skewness — and Pearson's coefficient specifically — is measuring.

![Skewness Distributions and Central Tendency Alignment](images/skewness_types.png)

---

## 10. Kurtosis

Measures how peaked (or flat) a distribution's peak is.

- **Mesokurtic** — normal distribution, kurtosis = 3 (excess kurtosis = 0)
- **Platykurtic** — kurtosis < 3 (flatter peak, thinner tails, excess kurtosis < 0)
- **Leptokurtic** — kurtosis > 3 (sharper peak, fatter tails, excess kurtosis > 0)

![Kurtosis Types: Mesokurtic vs Leptokurtic vs Platykurtic](images/kurtosis_types.png)

---

### Quick Summary of Corrections Applied
- **IQR fences**: Corrected formulas to $Q1 - 1.5 \times \text{IQR}$ and $Q3 + 1.5 \times \text{IQR}$.
- **Spelling**: Fixed "Platokurtic" $\rightarrow$ **Platykurtic**.
- **Quartile positions**: Clarified that $\frac{N+1}{4}$ and $\frac{3(N+1)}{4}$ give the **rank/position** in the sorted list.
- **Symbols**: Added proper LaTeX notation for sample mean ($\bar{x}$) and population mean ($\mu$).
- **CLT Qualifiers**: Included standard qualifiers (sufficient sample size $n \ge 30$, finite variance).
