tags: [[computer science]] [[machine learning]]
#computer_science #machine_learning 

# Overview
Correlation Analysis is a technique used in Data Science that deals with understanding the relationship between different variables. This is in order to understand the underlying patters in the data, that may give us valuable insight.
The core is of this concept is determining the strength of the link between two variables.

## What is correlation?
Correlation is simply defining the relationship between two variables in a dataset, it is the representation whether or not two variable are related to each other and by how much. This quantifiable number of how much two variables are related to each other is represented using Correlation Coefficient. Pearson Correlation coefficient is the most commonly used correlation metric and is expressed as 'r'.  It represents the linear relationship between two variables.
## Types of correlation
There are three types of correlation:
1. Positive Correlation: This indicates that two variables have a direct relationship and that when one increase so does the other.
2. Negative Correlation: This indicates that two variables have an inverse relationship and that when one increase the other decreases.
3. Zero Correlation: This indicates that there is no relationship between the two variables.

## Correlation Coefficients
| Correlation Coefficient                   | Type of Relation | Levels of Measurement                            | Data Distribution   |
| ----------------------------------------- | ---------------- | ------------------------------------------------ | ------------------- |
| **Pearson Correlation Coefficient**       | Linear           | Interval / Ratio                                 | Normal distribution |
| **Spearman Rank Correlation Coefficient** | Non-linear       | Ordinal                                          | Any distribution    |
| **Kendall Tau Coefficient**               | Non-linear       | Ordinal                                          | Any distribution    |
| **Phi Coefficient**                       | Non-linear       | Nominal vs. Nominal (2 categories / dichotomous) | Any distribution    |
| **Cramer's V**                            | Non-linear       | Two nominal variables                            | Any distribution    |
## Considerations
Correlation does not always mean causation, sometimes two variables can just be correlated and but not have any relationship at all. One of the ways you can see if it is a coincidence is to check if the variables that are being compared are hyper specific.

Only have positive or negative correlation is not the end all be all in a correlation analysis, the strength of the relationship also matters, the strength that is valid enough to be stated as an explicit relationship changes slightly in different use cases.
# Sources
## Internal Sources

## External Sources
1. https://www.geeksforgeeks.org/data-analysis/what-is-correlation-analysis/
2. https://researchmethod.net/correlation-analysis/