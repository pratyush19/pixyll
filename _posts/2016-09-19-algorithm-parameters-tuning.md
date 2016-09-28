---
layout:     post
title:      Algorithm Parameters Tuning 
date:       2016-09-19
summary:    Machine learning, Parameters tuning, Features engineering, Evaluation metrices, Data visualization
categories: blog
---
In this post, I will detail cover how to perform parameters tuning of an algorithm. Parameters tuning is a process of optimizing algorithm parameters to give the best result on an unseen data set. Before jumping into the parameters tuning, I will first give you the overview of feature scaling, feature selection, dimensionality reduction and cross-validation. These all needs to be performed to apply parameters tuning more efficiently.  I will use Python [scikit-learn](http://scikit-learn.org/stable/) library.

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

## Feature Selection

The minimal number of features a machine learning algorithm takes to capture the trends and patterns in the data. There are several go-to methods of automatically selecting features in sklearn. Many of them fall under the umbrella of univariate feature selection, which treats each feature independently and asks how much power it gives you in classifying or regressing.<br/>
There are two big univariate feature selection tools in sklearn: *SelectPercentile* and *SelectKBest*. The difference is pretty apparent by the names: SelectPercentile selects the X% of features that are most powerful (where X is a parameter) and SelectKBest selects the K features that are most powerful (where K is a parameter).

Read more about various parameters used in feature selection from [sklearn documentation](http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection).

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
PCA or, linear [dimenionality reduction](https://www.analyticsvidhya.com/blog/2015/07/dimension-reduction-methods/) used to reduce high dimensions of the features into lesser dimensions ensuring that it conveys similar information concisely. I'm not going into detail of PCA but just giving a brief introduction and its implementation using sklearn.
PCA is a systematized way to transform input features into principal components (PCs) or new features. PCs are directions in data that maximizes variance or minimizes information loss when you perform projection or compression down onto those PCs. Here information loss is the distance between old data point to its new transformed value and variance means variability or uniqueness of the dataset.

So, PCA is different from feature selection is that in PCA combines similar (correlated) features and creates new ones while feature selection doesn't combine features, it just evaluates their quality, predictive power and selects the best set.  

* *n_components* (int): number of components to keep with default value is min(n_samples, n_features)
* *whiten* (bool): transform data to unit variance and zero mean
* *random_state* (int): pseudo random number generator seed control

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=3, whiten=TRUE, random_state=42) #transform features to 3 new features
pca.fit(features)
```

Other dimensionality reduction:

* [RandomizedPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.RandomizedPCA.html#sklearn.decomposition.RandomizedPCA)
* [IncrementalPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA)
* [KernelPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA)

## Validation
Training and testing on the same dataset would fail to predict anything useful on yet-unseen data; this is a common problem called [overfitting](https://en.wikipedia.org/wiki/Overfitting). To avoid overfitting, it is common practice in machine learning experiment to hold out part of the available data as a test set (features_test, labels_test).<br/>
In sklearn, a random split into training and test set can be done using [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html#sklearn.cross_validation.train_test_split).

```python
# Splitting data into 70% training and 30% testing set

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=60)
```

## Cross-Validation

But there are problems splitting data into training and testing sets is that you want to maximize both of the sets. You want to maximize the training sets to get best learning result, and the testing sets to get the best validation. The solution to this problem is [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) (or CV). In sklearn, we can use [KFold](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html) to do cross-validation. In kfold cross-validation, we randomly partition the data set into k bins of equal size. Each fold has n%n_folds (where n is number of original data points and n_fold is number of bins) data points.<br/>
Run k separate learning experiments:

* pick testing set
* train (k -1) sets
* test on testing set

We then average test results from those k experiments. In this way, we have used all our data points for both training and testing.

```python
from sklearn.cross_validation import KFold
kf = KFold(len(labels), n_folds=5)  #partition the data set into 5 bins
for train_index, test_index in kf:
  features_train, features_test = features[train_index], features[test_index]
  labels_train, labels_test = labels[train_index], labels[test_index]
```

## Parameters Tuning
Parameters tuning is the final step in the process of applied machine learning. It is also called [Hyperparameter Optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization) where algorithm parameters referred to as hyperparameters. In this process, we choose different values of the hyperparameters with the goal of optimizing the algorithm's performance. It is critical first to understand the available parameters role on the algorithm performance before applying any parameters tuning.<br/>
Sklearn provides two different methods for parameters tuning, [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV) and [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html#sklearn.grid_search.RandomizedSearchCV). I discuss GridSearchCv as it is the most widely used method for parameter optimization.

#### GridSearchCV
GridSearchCV is a way of systematically working through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. It exhaustively generates candidates from a grid of parameter values specified with the *param_grid* parameter. The beauty is that it can work through many combinations in only a couple extra lines of code.<br/>
Let's do the parameters tuning for [decision tree](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier).

```python
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#dictionary of the parameters, and the possible values they may take
#you can use more values and parameters (more computation time)
parameters = {'max_depth':[None, 5, 10], 
              'min_samples_split':[2, 4, 6, 8, 10],
              'min_samples_leaf': [2, 4, 6, 8, 10],
              'criterion': ["entropy", "gini"]}
#decision tree algorithm for classification
dt = DecisionTreeClassifier()  
#pass the algorithm and the dictionary of parameters to generate a grid of parameter combinations to try
clf = GridSearchCV(dt, parameters) 
#fit function tries all the parameter combinations, and returns an optimal parameters value
clf.fit(features, labels)
#dictionary of optimal parameters value 
clf.best_params_.
```

We should try all the combinations of parameters, and not just vary them independently. In the above code, I  tried 3 different values of *max_depth*,  5 different values of each *min_samples_split* and *min_samples_leaf*, and 2 different values of *criterion*, that means 3 x 5 x 5 x 2 = 150 different combinations. *GridSearchCV* allows me to construct a grid of all the combinations of parameters, tries each combination, and then reports back the best combination.
 
 
Great, you have implemented parameters tuning for the decision tree with only a few extra lines of code. You can similarly tune parameters for any machine learning algorithm. But, there is one thing is that we have only used one estimator in the above code, i.e., the classifier in the parameters tuning. What if we want to use multiple estimators like *StandardScaler*, *SelectKBest*, *PCA* and grid search over parameters of all the estimators, the answer is *Pipeline* followed by grid search.

### Pipeline 
The pipeline module of sklearn allows you to chain transformers and estimators together in such a way that you can use them as a single unit. One thing to note that all estimators in a pipeline, except the last one, must be transformers (i.e. must have a transform method). The final estimator may be any type (transformer, classifier, etc.).

```python
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

#pipeline steps follow in the respective order:
#SelectKBest -> PCA -> DecisionTreeClassifier
pipe = Pipeline(steps=[("feature_selection", SelectKBest()), ("pca", PCA()), \
                       ("decision_tree", DecisionTreeClassifier())])

parameters = {"feature_selection__k": range(8, 10)}
dt_pca = {"pca__n_components": range(4, 7), "PCA__whiten": [True, False]}
dt_clf = {"decision_tree__min_samples_leaf": [2, 6, 10, 12],
          "decision_tree__min_samples_split": [2, 6, 10, 12],
          "decision_tree__criterion": ["entropy", "gini"],
          "decision_tree__max_depth": [None, 5]}

#update pca and classifier to the parameters dictionary
parameters.update(dt_pca)  
parameters.update(dt_clf)
```

In the above code, we first apply feature selection, then principal component analysis and then, finally decision tree classifier. There’s a particular convention you need to follow to name the parameters in the parameters dictionary. You need to have the name of the Pipeline step (e.g. decision_tree), followed by two underscores, followed by the name of the parameter (e.g., max_depth) that you want to vary.

```python
from sklearn.grid_search import GridSearchCV

grid_search = GridSearchCV(pipeline, param_grid=parameters)
grid_search.fit(features_train, labels_train)
prediction = grid_search.predict(features_test)

best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])
```

Once you have got the parameter grid set up correctly, then you apply GridSearchCV that multiplies out all the combinations of parameters and tries each one. Then you can ask for predictions from my GridSearchCV object, and it will automatically return to me the best set of predictions. Of course, trying tons of models can be time-consuming, but the outcome is a much better understanding of how my model performance depends on parameters.


