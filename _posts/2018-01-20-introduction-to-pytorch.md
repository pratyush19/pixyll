---
layout:     post
title:      Introduction to PyTorch
date:       2018-01-20
summary:    Deep Learning Framework, Neural Network
categories: blog
---

Before dive into the coding of [PyTorch](http://pytorch.org/), let's first know the motivation behind it, as there are lots of deep learning frameworks with great performance and documentation support like [TensorFlow](https://www.tensorflow.org/). 
PyTorch is inspired by the popular deep learning framework Torch, which is written in Lua programming language. Some AI researchers at Facebook who were inspired by Torch programming style decided to implement in Python calling it Py-Torch.

The main advantage of PyTorch is the adoption of Dynamic Computation Graph different from Static Computation Graph which most popular deep learning frameworks like Theano, Caffe, TensorFlow follow. You can read about Computation Graph in this [blog](http://colah.github.io/posts/2015-08-Backprop/) post. Deep learning frameworks maintain a computational graph that defines the order of computations that are required to be performed. In dynamic computation graph, we can define, change and execute nodes as we go. For example, input data of our neural network can be a set of data with different shapes, which means the computation graphs for each distinct data can be different. There are several dynamic neural network architectures like RNN that can benefit from the dynamic approach. With static graphs, the input sequence length in RNN will stay constant. This means that if we develop a sentiment analysis model for English sentences we must fix the sentence length to some maximum value and pad all smaller sequences with zeros. There are lots more in PyTorch which allow you to focus exclusively on your experiment and iterate very quickly, give you better developing and debugging experience.
PyTorch may feel more “pythonic” as it work just like Python and has an object-oriented approach.


## PyTorch Tensors

PyTorch tensors are very similar to NumPy, like both are a generic tool for scientific computing and do not know anything about deep learning or computationl graphs or gradients. Tensors support a lot of the same numpy API, so sometimes you may use PyTorch just as a drop-in replacement of the NumPy. And it’s very easy to convert tensors from NumPy to PyTorch and vice versa. However unlike numpy, PyTorch Tensors can utilize GPUs to accelerate their numeric computations. To run a PyTorch Tensor on GPU, you simply need to cast it to a new datatype.


```python
import torch
import numpy as np

numpy_array = np.random.randn(10,10)

# convert numpy array to pytorch tensor
pytorch_tensor = torch.from_numpy(numpy_array)
# or
pytorch_tensor = torch.Tensor(numpy_array)

# define pytorch tensor
dytpe = torch.FloatTensor
cpu_tensor = torch.randn(10, 20).type(dtype)
# or, FloatTensor by default
cpu_tensor = torch.randn(10, 20)

# call 'cuda()' method to convert cpu tensor to gpu tensor
gpu_tensor = cpu_tensor.cuda()

# back to cpu tensor
cpu_tensor = gpu_tensor.cpu()

# get the shape
cpu_tensor.size()		# torch.Size([10, 20])
```

## Variables and Autograd

In any neural network, we have to implement both forward and backward passes. The forward pass of your neural network define a computational graph and the backward pass or backpropagation allows you to easily compute gradients through this graph. If we use tensors, we have to manually implement both the forward and backward passes of our neural network. Manually implementing the backward pass is simple for a small two-layer network, but can quickly get very hairy for large complex networks.

PyTorch provides automatic differentiation system "autograd" to automate the computation of backward passes in neural networks. We can do forward pass using operation on PyTorch Variables, and uses PyTorch autograd to compute gradients. So, a PyTorch Variable is a wrapper around a PyTorch Tensor, and represents a node in a computational graph. If `x` is a Variable then `x.data` is a Tensor giving its value, and `x.grad` is another Variable holding the gradient of `x` with respect to some scalar value. PyTorch Variables have the same API as PyTorch Tensors, almost any operation that you can perform on a Tensor also works on Variables; the difference is that using Variables defines a computational graph, allowing you to automatically compute gradients.

Every Variable has two flags: `requires_grad` and `volatile`, which are used for exclusion of subgraphs from gradient computing and increase efficiency. If there’s a single input to an operation that requires gradient, its output will also require gradient. Conversely, only if all inputs don’t require gradient, the output also won’t require it. Backward computation is never performed in the subgraphs, where all Variables didn’t require gradients.

```python
x = Variable(torch.randn(5, 5))
y = Variable(torch.randn(5, 5))
z = Variable(torch.randn(5, 5), requires_grad=True)

a = x + y		# won't require gradient
b = x + z		# requires gradient
```

This is especially useful when you want to freeze part of your model, or you know in advance that you’re not going to use gradients w.r.t. some parameters.

In case of inference (neural network is trained and ready for testing), it’s better to provide volatile flag during variable creation. It can be provided only in case if you exactly sure that there will be no any gradients computing.

```python
x = Variable(torch.randn(5, 5), volatile=True)
```

Let's use PyTorch Variables and autograd to implement a two-layer neural network.

```python
import torch
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

dtype = torch.FloatTensor

# Create random Tensors to hold input and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y using operations on Variables
  y_pred = x.mm(w1).clamp(min=0).mm(w2)

  # Compute and print loss using operations on Variables.
  # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
  # (1,); loss.data[0] is a scalar value holding the loss.
  loss = (y_pred - y).pow(2).sum()
  print(t, loss.data[0])

  # Use autograd to compute the backward pass. 
  # After this call w1.grad and w2.grad will be Variables holding the gradient
  # of the loss with respect to w1 and w2 respectively.
  loss.backward()

  # Update weights using gradient descent; w1.data and w2.data are Tensors,
  # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
  # Tensors.
  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data

  # Manually zero the gradients after updating weights
  w1.grad.data.zero_()
  w2.grad.data.zero_()
```

## PyTorch nn Module

PyTorch autograd makes it easy to define computational graphs and take gradients, but raw autograd can be a bit too low-level for defining complex neural networks. When building neural networks we frequently think of arranging the computation into layers, some of which have learnable parameters which will be optimized during learning.

The *nn* package defines a set of Modules, which are roughly equivalent to neural network layers. A Module receives input Variables and computes output Variables, but may also hold internal state such as Variables containing learnable parameters. The *nn* package also defines a set of useful loss functions that are commonly used when training neural networks.

```python
import torch

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )

# Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)
```

If we want to build more complex models, we can define our own Modules by subclassing `nn.Module` and defining a forward which receives input Variables and produces output Variables using other modules or other autograd operations on Variables.

```python
import torch.nn as nn

class TwoLayerNN(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(TwoLayerNN, self).__init__()
    self.linear1 = nn.Linear(D_in, H)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(H, D_out)

  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must return
    a Variable of output data. We can use Modules defined in the constructor as
    well as arbitrary operators on Variables.
    """
    h_in = self.linear1(x)
    h_relu = self.relu(h_in)
    y_pred = self.linear2(h_relu)
    return y_pred
```

## Optimization

As of now, we have manually updating weights of our models:

```python
w1.data -= learning_rate * w1.grad.data
w2.data -= learning_rate * w2.grad.data
```    

This should be fine for simple optimization algorithm like Stochastic Gradient Descent (SGD) but can be very tedious for more advanced and sophisticated algorithms like RMSprop, AdaGrad, Adam etc. The *optim* package in PyTorch provides implementations of commonly used optimization algorithms.


Implementing the two-layer neural network with different packages in the PyTorch end-to-end on GPU. 

```python
import torch
import torch.nn as nn
from torch.autograd import Variable

class TwoLayerNN(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(TwoLayerNN, self).__init__()
    self.linear1 = nn.Linear(D_in, H)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(H, D_out)

  def forward(self, x):
    h_in = self.linear1(x)
    h_relu = self.relu(h_in)
    y_pred = self.linear2(h_relu)
    return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in)).cuda()
y = Variable(torch.randn(N, D_out), requires_grad=False).cuda()

model = TwoLayerNN(D_in, H, D_out)
model.cuda()

criterion = torch.nn.MSELoss(size_average=False)

# model.parameters() in the SGD constructor will contain the 
# learnable parameters of the model.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
  y_pred = model(x)
  loss = criterion(y_pred, y)
  optimizer.zero_grad()
  loss.backward()

# update the weights
optimizer.step()
```

## PyTorch DataLoader

A lot of effort in solving any machine learning problem goes into preparing the data. PyTorch provides `torch.utils` package to make data loading easy and hopefully, to make your code more readable. 

```python
import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64000, 1000, 100, 10
 
# Suppose you have training data x_train and y_train loaded in numpy
# x_train shape is (64000, 1000) and y_train shape is (64000, 10)
# Convert x_train and y_train into tensor
x = torch.from_numpy(x_train)
y = torch.from_numpy(y_train)

# build dataloader
training_dataset = torch.utils.data.TensorDataset(x, y)
data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64)

# calling model 
model = TwoLayerNN(D_in, H, D_out)
model.cuda()

for t in range(500):
  for i, data in enumerate(data_loader):
    # Get each batch
    inputs, labels = data

    inputs, labels = Variable(inputs), Variable(labels)
    y_pred = model(inputs)

    loss = criterion(y_pred, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    

```

I hope this blog helps you to understand the main points of PyTorch. PyTorch is a great framework for researchers, fast prototyping, maximum flexibility, easy to debug, which is getting momentum fast.
