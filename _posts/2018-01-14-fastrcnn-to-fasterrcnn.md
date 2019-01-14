---
layout:     post
title:      Fast R-CNN to Faster R-CNN 
date:       2018-01-14
summary:    Convolution Neural Network, Object Detection, Region Proposal
categories: blog
---

Earlier state-of-the-art object detection networks relies on region proposal methods and region-based convolutional neural network (R-CNN) to approximate object locations. [Selective Search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf), one of the popular region proposal method looks at the image through windows of different sizes, and for each size tries to group together adjacent pixels by texture, color, or intensity to identify objects. It generates many regions, each of which belongs to at most one object, recursively combine similar regions into larger ones and then use the generated regions to produce object locations. [R-CNN](https://arxiv.org/pdf/1311.2524.pdf) take an image, and correctly identify where the main objects (via a bounding box) in the image.
R-CNN creates these bounding boxes, or region proposals, using Selective Search. It passes these bounding boxes to a CNN (AlexNet) and creates features maps for each bounding boxes. After passing through the CNN, it adds a Support Vector Machine (SVM) that simply classifies whether it contains an object.

## Fast R-CNN

A [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf) network takes as input an entire image and set different size of object proposals using Selective Search method. The network first processes the whole image with deep VGG16 network to produce a conv feature map (last conv layer). These object proposals projection on conv feature map are then sent to a Roi Pooling layer that resize all proposals to a fixed size. This step is needed because the fully connected layer expect that all the vectors will have same size. Each feature vector is fed into a sequence of fully connected layers that finally branch into two sibling output layers: one that produces softmax probability estimates over K object classes plus “background” class and another layer that outputs four real-valued numbers for each of the K object classes.

<br />
<p align="center">
<img src="/images/fastrcnn.png"/>
</p>
<br />

### RoI Pooling Layer

It is a type of pooling layer which performs max pooling on inputs of non-uniform sizes and produces a small feature map of fixed size (say 7x7), so that the output always has the same size. The inputs of the RoI layer are the region proposals and the last convolution layer activations. The region proposals are N x 5 matrix representing a list of regions of interest, where N is a number of RoIs. The first column represents the image index and the remaining four are the coordinates of the top left and bottom right corners of the region. For every region of interest from the input list, RoI pooling layer takes a section of the input feature map that corresponds to it and scales it to some pre-defined size (e.g., 7×7). If there are multiple different size region proposals, we can still use the same input feature map for all of them, this increases processing speed and can save a lot of time. The RoI layer is simply the special-case of the spatial pyramid pooling layer used in [SPPnets](https://arxiv.org/pdf/1406.4729.pdf) in which there is only one pyramid level.

<br />
<p align="center">
<img src="/images/roi.png"/>
</p>
<br />

So, using RoI pooling, Fast R-CNN jointly train the CNN, classifier, and bounding box regressor in a single model. Where earlier in R-CNN, it requires a forward pass of the CNN (AlexNet) for every single region proposal for every single image, it had different models to extract image features (CNN), classify (SVM), and tighten bounding boxes (regressor), Fast R-CNN instead used a single network to compute all three.


## Faster R-CNN

RoI pooling in Fast R-CNN significantly improves the processing time, there is still one remaining bottleneck in the Fast R-CNN process — the region proposer. As we see, the very first step to detecting the locations of objects is generating a bunch of potential bounding boxes or regions of interest to test. In Fast R-CNN, these proposals were created using Selective Search, a fairly slow process that is found to be the bottleneck of the overall process.
The insight of Faster R-CNN is that region proposals depended on features of the image that are already calculated with the forward pass of the CNN (first step of classification). So why not reuse those same CNN results for region proposals instead of running a separate selective search algorithm.

[Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) has two networks: region proposal network (RPN) for generating region proposals and a network using these proposals to detect objects (Fast R-CNN). The main different here with Fast R-CNN is that the later uses selective search to generate region proposals. The time cost of generating region proposals is much smaller in RPN than selective search, when RPN shares the most computation with the object detection network (Fast R-CNN). RPN ranks region boxes (called anchors) and proposes the ones most likely containing objects.


<br />
<p align="center">
<img src="/images/fasterrcnn.png"/>
</p>
<br />

In the image above, we can see how a single CNN is used to both carry out region proposals and classification. This way, only one CNN needs to be trained and we get region proposals almost for free.

### Anchors

The Region Proposal Network works by passing a sliding window over the CNN feature map and at each window, outputting k potential bounding boxes and scores for how good each of those boxes is expected to be. The bounding boxes here are called anchors.  In the default configuration of Faster R-CNN, there are 9 anchors at a position of an image. By default we use 3 scales and 3 aspect ratios, yielding k = 9 anchors at each sliding position. Three scales sizes are 128x128, 256x256, 512x512 and three aspect ratios are 1:1, 1:2, 2:1. We can tune these parameters (scales, aspect ratios) according to our dataset configuration.

### Region Proposal Network

An image goes through a CNN which output a set of convlutional feature maps (HxW) on the last convolutional layer. Then a sliding window of size 3x3 is run spatially on these feature maps. A set of 9 anchors are generated with 3 different aspect ratios and 3 different scales as discussed above. All these anchors coordinates are computed with respect to the original image. For each of these anchors, an IoU is computed which indicated how much these anchors overlap with the ground-truth bounding boxes.

The output of a region proposal network (RPN) is a set of proposals (9 * H * W) that is examined by a classifier and regressor to find the occurrence of objects. 

<br />
<p align="center">
<img src="/images/rpn.png"/>
</p>
<br />

We then pass each proposed regions that is likely to be an object into Fast R-CNN to generate a classification and tightened bounding boxes.  For the very deep VGG-16 model, Faster R-CNN has a frame rate of 5fps on a GPU, which is about 10X faster than Fast R-CNN.
