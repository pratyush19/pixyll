---
layout:     post
title:      Algorithm Parameteres Tuning 
date:       2016-09-19
summary:    Machine learning, Parameters tuning, Features engineering, Evaluation metrices, Data visualization
categories: blog
---
In this blog, I will cover how to find best parameters for a Machine Learning algorithm using Sklearn.

> ## Features Selection

*make everything as simple as possible, but no simpler* -- Albert Einstein<br/>
There are several go-to methods of automatically selecting features in sklearn. Many of them fall under the umbrella of univariate feature selection, which treats each feature independently and asks how much power it gives you in classifying or regressing.<br/>
There are two big univariate feature selection tools in sklearn: ```SelectPercentile``` and ```SelectKBest```. The difference is pretty apparent by the names: SelectPercentile selects the X% of features that are most powerful (where X is a parameter) and SelectKBest selects the K features that are most powerful (where K is a parameter).

* Classification: [f_classif](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif), [chi2](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2)
* Regression: [f_regression](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression)

### SelectPercentile

Select top 10 percentile features for the classification data

```python
from sklearn.feature_selection import SelectPercentile, f_classif
x_best = SelectPercentile(f_classif, percentile=10)  
x_best.fit_transform(features, labels)
```

### SelectKBest

Select k best features for the classification data

```python
from sklearn.feature_selection import SelectKBest, f_classif
k_best = SelectkBest(f_classif, percentile=10)  
k_best.fit_transform(features, labels)
```
