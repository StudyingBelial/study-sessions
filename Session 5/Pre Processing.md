tags: [[computer science]], [[machine learning]]
#computer_science #machine_learning

# Overview
Data preprocessing is the first process in ML or statistical pipeline, it involves cleaning, transforming and organization of raw data to ensure it is accurate, consistent and ready for modeling. The preprocessing stage of training an ML model can take up to 80% of the entire project timeline.
## Steps of Pre Processing
1. [[Data Cleaning]]
2. [[Data Transformation]]
3. [[Dimensionality Reduction]]
4. [[Feature Engineering]]
5. [[Handling Imbalanced Data]]
While there is a flow, these steps are not necessary to be followed in order. You can skip any step if not required. On top of which these steps have overlap so the conceptual steps of the Machine Learning pipeline is more important than hard and fast step by step rules.
## Considerations for Machine Learning
Before spending resources on doing anything, it is important to learn and remember a few things.
1. How much data is enough for the model to generalize well?
2. What would constitute as a good model? What metrics? What capability or business requirement?
3. The type of model to be used?
4. What are your sources of data and how reliable are they?
5. Are you combining different datasets? If so then what are the standards that are to be used?
## Data Cleaning
This process mainly involves correcting the errors in a dataset, most specifically handling the missing and inconsistent values. For example, there may be missing values which were valid input as they were optional, so you must choose on how you want to handle the missing data? Is the missing data something you are able to add numerical values like 0 or 1 to or are they errors in the input and have to be removed?
The other common point of correction is bring the data to the same scale or unit, as different data sources may have different units for the same data or the data may be so raw that during the collection of data, the values may have been put in a format that is not consistent with the rest of the data.
And lastly handling the duplicates and irrelevant data. While duplicates are usually easy to deal with, irrelevant data is something that one has to think hard about as it depends on what you are trying to achieve. For example if you are trying to predict the prices of houses, then the color of the house is irrelevant, but if you are trying to predict the type of house then the color of the house is relevant.
The main point of Data Cleaning is to remove noise from the data, values which are obviously wrong or inconsistent. This is does for one thing to reduce the possible number of values however small that reduction may be and not to have system breaking values like the aforementioned NaN values.
## Data Transformation
Transformation is where specifically standardization comes in, this is more helpful for combined datasets, adjusting the units, numeric values, encoding and such.
Scaling is used to adjust the numeric values to a common scale, this is often necessary for Machine Learning, and generally algorithms that rely on distance metrics.
One of the most common forms of data transformation is encoding into numerical values, statistical and machine learning models fail to operate on non numerical or categorical data, so the data is encoded. Even though Data Transformation is not the last part of the Machine Learning pipeline, this part also traditionally deals with the train, test split.

## Dimensionality Reduction
Dimensionality Reduction or more simply data reduction simplifies the dataset by reducing the number of features or records while preserving the essential information. This helps us speed up analysis and model training without sacrificing the accuracy.
Some popular methods of Data reduction is Feature Selection, PCA (Principal Component Analysis) and implementing sampling methods.
Feature Selection is simply choosing what we deem to be the most relevant features contributing the analysis or model performance. While PCA is a technique to transform the data into a lower dimensional space, pretty similar but the principled approach for these two is different. Sample while less common is also a method of data reduction, by only calling on the input features that are representative of the data, useful for smaller datasets.

## Feature Engineering
Feature engineering is a more advanced preprocessing step in the Machine Learning pipeline, where one selects, creates and modifies the features in a dataset. It involves transforming raw data into meaningful inputs that improve model accuracy and performance.
Feature Engineering is an umbrella category that involves a lot of methods that can overlap with other steps of the pre processing pipeline. But the main point to remember about this step is performing thing like PCA or correlation analysis to learn about what features may need, removal or combination.

## Handling Imbalanced Data
Data Imbalance or Class Imbalance is a common occurrence in Machine Learning and learning how to handle this may be the most impactful step that will be seen later in the training step. Some of the common ways of handling this is Over Sampling of the under represented class or Under Sampling of the over represented class, SMOTE which creates synthetic points for the under represented class or Cost Sensitive Training, where mis predictions for the under represented class are heavily penalized

# Sources
# Internal Sources
1. [[Data Cleaning]]
2. [[Data Transformation]]
3. [[Dimensionality Reduction]]
4. [[Principal Component Analysis]]
5. [[Feature Engineering]]
6. [[Handling Imbalanced Data]]
7. [[Synthetic Minority Over Sampling]]
8. [[Cost Sensitive Training]]

# External Sources
1. https://www.datacamp.com/blog/data-preprocessing
2. https://www.geeksforgeeks.org/data-analysis/data-preprocessing-machine-learning-python/