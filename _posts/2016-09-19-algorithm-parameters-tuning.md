---
layout:     post
title:      Algorithm Parameters Tuning 
date:       2016-09-19
summary:    Machine learning, Parameters tuning, Features engineering, Evaluation metrices, Data visualization
categories: blog
---
In this post, I will detail cover how to perform parameters tuning of an algorithm. Parameters tuning is a process of optimizing algorithm parameters to give the best result on an unseen data set. Before jumping into the parameters tuning, I will first give you the overview of feature scaling, feature selection, dimensionality reduction and validation. These all needs to be performed to apply parameters tuning more efficiently.  I will use Python [scikit-learn](http://scikit-learn.org/stable/) library.

## Feature Scaling
Feature scaling is a major step in pre-processing the features for some types of machine learning algorithms. Some algorithms like Support Vector Machine, K-means calculate the distance between points, in that case, feature scaling becomes significantly important. If one of the features has a broad range of values, the distance will be dominated by this particular feature. Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance. 

#### Standardization
Feature standardization or, Z-score normalization rescaled the values of each feature to behave like standard normal distribution with mean equal to 0 and standard deviation equal to 1. The Z-score normalization is calculated by [standard scores](https://en.wikipedia.org/wiki/Standard_score) using equation: z = (X-mean)/s.d.. It transforms the data to center it by removing the mean value of each feature, then scale it by dividing by their standard deviation. Below is its implementation in scikit-learn.

```python
from sklearn.preprocessing import StandardScaler()
scaler = StandardScaler()
rescaled_features = scaler.fit_transform(features)
```

#### MinMax Scaling
Min-max scaling (or, normalization) transforms each feature to a range [0, 1]. The general formula to transform each feature given by (X-X.min)/(X.max-X.min). 

The use of standardization or normalization depends on the application you choose. For most applications, standardization is preferred.

```python
from sklearn.preprocessing import MinMaxScaler()
scaler = MinMaxScaler()
rescaled_features = scaler.fit_transform(features)
```

## Features Selection

The minimal number of features a machine learning algorithm takes to capture the trends and patterns in the data. There are several go-to methods of automatically selecting features in sklearn. Many of them fall under the umbrella of univariate feature selection, which treats each feature independently and asks how much power it gives you in classifying or regressing.<br/>
There are two big univariate feature selection tools in sklearn: ```SelectPercentile``` and ```SelectKBest```. The difference is pretty apparent by the names: SelectPercentile selects the X% of features that are most powerful (where X is a parameter) and SelectKBest selects the K features that are most powerful (where K is a parameter).

Read more about various parameters used in feature selection from sklearn [documentation](http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection).

#### SelectPercentile

Select top x percentile features for the classification data

```python
from sklearn.feature_selection import SelectPercentile, f_classif
x_best = SelectPercentile(f_classif, percentile=x)  
x_best.fit_transform(features, labels)
```

#### SelectKBest

Select k best features for the classification data

```python
from sklearn.feature_selection import SelectKBest, f_classif
k_best = SelectkBest(f_classif, k=k)  
k_best.fit_transform(features, labels)
```
 

## Principal Component Analysis (PCA)
We now implement **dimensionality reduction** (or, [PCA](http://scikit-learn.org/stable/modules/decomposition.html#pca)  used to reduce the dimensions of the features. I'm not going into detail of PCA but just giving a brief introduction and its implementation using sklearn.
PCA is a systematized way to transform input features into principal components (PCs) or new features. PCs are directions in data that maximizes variance or minimizes information loss when you perform projection or compression down onto those PCs. Here information loss is the distance between old data point to its new transformed value and variance means variability or uniqueness of the dataset.

So, PCA is different from feature selection is that in PCA combines similar (correlated) features and creates new ones while feature selection doesn't combine features, it just evaluates their quality, predictive power and selects the best set.  

* ```n_components``` (int): number of components to keep with default value is min(n_samples, n_features)
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
Training and testing on the same dataset would fail to predict anything useful on yet-unseen data, this is a common problem called [overfitting](https://en.wikipedia.org/wiki/Overfitting). To avoid overfitting, it is common practice in machine learning experiment to hold out part of the available data as a test set (features_test, labels_test).<br/>
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
Parameters tuning is the final step in the process of applied machine learning. It is also [Hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization) where the algorithm parameters are referred to as hyperparameters. In this process, we choose diffferent values of the hyperparameters with the goal of optimize the algorithm's performance. It is really important to first understand the available parameters and their roles in the performance of the algorithm's before performing any parameters tuning.<br/>
Sklearn provides two different methods for parameters tuning, [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV) and [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html#sklearn.grid_search.RandomizedSearchCV). I discuss GridSearchCv as it is the most widely used method for parameter optimization.

#### GridSearchCV
GridSearchCV is a way of systematically working through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. It exhaustively generates candidates from a grid of parameter values specified with the ```param_grid``` parameter. The beauty is that it can work through many combinations in only a couple extra lines of code.<br/>
Let's do the parameters tuning for [decision tree](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)

```python
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#dictionary of the parameters, and the possible values they may take
#you can more values and parameters (more computation time)
parameters = {'max_depth':[None, 5, 10], 
              'min_samples_split':[2, 4, 6, 8, 10],
              'min_samples_leaf': [2, 4, 6, 8, 10],
              'criterion': ["entropy", "gini"],
              'random_state': [42, 46, 60]}
#decision tree algorithm for classification
dt = DecisionTreeClassifier()  
#pass the algorithm and the dictionary of parameters to generate a grid of parameter combinations to try
clf = GridSearchCV(dt, parameters) 
#fit function tries all the parameter combinations, and returns an optimal parameters value
clf.fit(features, labels)
#dictionary of optimal parameters value 
clf.best_params_.
```


