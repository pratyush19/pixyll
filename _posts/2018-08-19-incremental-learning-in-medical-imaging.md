---
layout:     post
title:      Incremental Learning in Medical Imaging
date:       2018-08-19
summary:    Incremental Learning, Lifelong Learning, Hard Example Mining, Convolutional Neural Network
categories: blog
---


Recently my paper with [Muktabh Mayank](https://www.quora.com/profile/Muktabh-Mayank) *Example Mining for Incremental Learning in Medical Imaging* is published at 2018 IEEE Symposium Series on Computational Intelligence (SSCI).

[Incremental Learning](https://en.wikipedia.org/wiki/Incremental_learning) is well known machine learning approach wherein the weights of the learned model are dynamically and gradually updated to generalize on new unseen data without forgetting the existing knowledge. Incremental learning proves to be time as well as resource-efficient solution for deployment of deep learning algorithms in real world as the model can automatically and dynamically adapt to new data as and when annotated data becomes available. The development and deployment of Computer Aided Diagnosis (CAD) tools in medical domain is another scenario, where incremental learning becomes very crucial as collection and annotation of a comprehensive dataset spanning over multiple pathologies and imaging machines might take years. However, not much has so far been explored in this direction. In the current work, we propose a robust and efficient method for incremental learning in medical imaging domain. Our approach makes use of Hard Example Mining technique (which is commonly used as a solution to heavy class imbalance) to automatically select a subset of dataset to fine-tune the existing network weights such that it adapts to new data while retaining existing knowledge. We develop our approach for incremental learning of our already under test model for detecting dental caries. Further, we apply our approach to one publicly available dataset and demonstrate that our approach reaches the accuracy of training on entire dataset at once, while availing the benefits of incremental learning scenario.

**[Read Paper](https://arxiv.org/abs/1807.08942?context=cs).**
