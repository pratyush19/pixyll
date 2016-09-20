---
layout:     post
title:      Algorithm Parameteres Tuning 
date:       2016-09-19
summary:    Machine learning, Parameters tuning, Features engineering, Evaluation metrices, Data visualization
categories: blog
---
In this blog, I will cover how to find best parameters for a Machine Learning algorithm using Sklearn.

## Feature Scaling
Feature scaling is an important step in pre-processing the features for some types of machine learning algorithms. 

#### Standardization
It is used to rescaled the features to behave like standard normal distribution with mean equal to 0 and standard deviation equal to 1. The standard scores of the samples are calculated using z = (X - mean)/s.d. The features are now centered around o with a standard deviation of 1.

```python
from sklearn.preprocessing import StandardScaler()
scaler = StandardScaler()
rescaled_features = scaler.fit_transform(features)
```
#### MinMax Scaling
The min-max rescaler (or normalization) transform the features to have range [0, 1]. A min-max scaling is done using the following equation: (X - X.min)/(X.max - X.min)

```python
from sklearn.preprocessing import MinMaxScaler()
scaler = MinMaxScaler()
rescaled_features = scaler.fit_transform(features)
```

## Features Selection

*make everything as simple as possible, but no simpler* -- Albert Einstein<br/>
The minimal number of features a machine learning algorithm takes to really capture the trends and patterns in the data.
There are several go-to methods of automatically selecting features in sklearn. Many of them fall under the umbrella of univariate feature selection, which treats each feature independently and asks how much power it gives you in classifying or regressing.<br/>
There are two big univariate feature selection tools in sklearn: ```SelectPercentile``` and ```SelectKBest```. The difference is pretty apparent by the names: SelectPercentile selects the X% of features that are most powerful (where X is a parameter) and SelectKBest selects the K features that are most powerful (where K is a parameter).

* Classification: [f_classif](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif), [chi2](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2)
* Regression: [f_regression](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression)

#### SelectPercentile

Select top 10 percentile features for the classification data

```python
from sklearn.feature_selection import SelectPercentile, f_classif
x_best = SelectPercentile(f_classif, percentile=10)  
x_best.fit_transform(features, labels)
```

#### SelectKBest

Select k best features for the classification data

```python
from sklearn.feature_selection import SelectKBest, f_classif
k_best = SelectkBest(f_classif, k=k)  
k_best.fit_transform(features, labels)
```

After selecting the k or x best features, we now implement **dimensionality reduction**  used to reduce the dimensions of the features. I'm not going into detail of [PCA](http://scikit-learn.org/stable/modules/decomposition.html#pca) but just giving a brief introduction and its implementation using sklearn.
## Principal Component Analysis (PCA)
PCA is a systematized way to transform input features into principal components (PCs) or new features. PCs are directions in data that maximizes variance or minimizes information loss when you perform projection or compression down onto those PCs. Here information loss is the distance between old data point to its new transformed value and variance means variability or uniqueness of the dataset.

* ```n_components``` (int): number of components to keep with default value is ```min(n_samples, n_features)```
* ```whiten``` (bool): transform data to unit variance and zero mean
* ```random_state``` (int): pseudo random number generator seed control
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=3, whiten=TRUE, random_state=42) #transform features to 3 new features
pca.fit(features)
```
Other PCA:
* [RandomizedPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.RandomizedPCA.html#sklearn.decomposition.RandomizedPCA)
* [IncrementalPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA)
* [KernelPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA)

## Validation
Training and testing on the same dataset would fail to predict anything useful on yet-unseen data, this is a common problem called [overfitting](https://en.wikipedia.org/wiki/Overfitting). To avoid overfitting, it is common practice in machine learning experiment to hold out part of the available data as a **test set** ```features_test, labels_test```.<br/>
In sklearn, a random split into training and test set cab be done using [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html#sklearn.cross_validation.train_test_split).

```python
# Splitting data into 70% training and 30% testing set

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=60)
```

## Cross-Validation

But there are problems splitting data into training and testing sets is that you want to maximize both of the sets. You want to maximize the training sets to get best learning result and the the testing sets to get best validation. The solution to this problem is [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) (or CV). In sklearn, we can use [KFold](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html) to do cross-validation. In kfold cross-validation, we randomly partition the data set into k bins of equal size. Each fold have n%n_folds (where n is number of original data points and n_fold is number of bins) data points.<br/>
Run k seperate learning experiments:
* pick testing set
* train (k -1) sets
* test on testing set

Average test results from those k experiments. In this way, we have used all our data points for both training and testing.

```python
from sklearn.cross_validation import KFold
kf = KFold(len(labels), n_folds=5)  #partition the data set into 5 bins
for train_index, test_index in kf:
  features_train, features_test = features[train_index], features[test_index]
  labels_train, labels_test = labels[train_index], labels[test_index]
```

## Parameters Tuning
