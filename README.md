<div align="center">
  <h1>TensorFlow in Practice Specialization</h1>
</div>

<div align="center"> 
  <img src="readme_img/Tensorflow_logo.svg.png" width="250">
</div>

# Table of Content
- [Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](#Introduction-to-TensorFlow-for-Artificial-Intelligence,-Machine-Learning,-and-Deep-Learning)
    - [A new programming paradigm](#A-new-programming-paradigm)
        - [A primer in machine learning](#A-primer-in-machine-learning)
        - [The Hello World of neural networks](#The-Hello-World-of-neural-networks)
        - [From rules to data](#From-rules-to-data)
    - [Introduction to Computer Vision](#Introduction-to-Computer-Vision)
        - [An introduction to computer vision](#An-introduction-to-computer-vision)


# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

## A new programming paradigm

### A primer in machine learning

In _traditional programming_ we establish the rules. In _machine learning_ is all about a computer learning the patterns that distinguish things.

<div align="center"> 
  <img src="readme_img/paradigmas.png" width="60%">
</div>

An example of this is made an "Hello World" of neural networks to see this. But first see this 2 vectors:

```
X = -1, 0, 1, 2, 3, 4
Y = -3, -1, 1, 3, 5, 7
```

With you mind you can figure out the solution for this.

```
y = 2x - 1
```

But in machine learning when try figure out this rule, the machine will use probabilities to solve this. So if we try to predict `x = 10`, with our function `y = 2x - 1` we get `y = 90`, but with machine learning we going to be close to `90`.

### The Hello World of neural networks

A neural network is basically a set of functions wich can learn patterns. The simpliest neural network is one that has only one neuron in it.

```py
# This line of code usings TensorFlow and the API Keras.
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
```

- **Dense:** define a layer.
- **units:** define the numbers of neurons in the layer.
- **input_shape:** is the shape of inputs for the layer.

For compile our neural network we use the compile method.

```py
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
```

- **compile:** is a method from keras objetct, to compile the Sequential layers.
- **optimizer:** make corrections in model.
- **loss:** measure the error.

Our model dosn't have an idea to how solve the problem, so it will be start with a guest, so we get an wrong answer, but with the **loss function** we can **measure that error**, and this **measure** will goint to the **optimizer** wich **figures out** the next guess.

The next step is represent the know data and train the model.

```py
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# The know data.
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Training the model.
model.fit(xs, ys, epochs=500)
```

- **epochs:** determine how many times loop the training model with the process describe earlier.

### From rules to data

You saw that the traditional paradigm of expressing rules in a coding language may not always work to solve a problem. As such, scenarios such as Computer Vision are very difficult to solve with rules-based programming. Instead, if we feed a computer with enough data that we describe (or label) as what we want it to recognize, given that computers are really good at processing data and finding patterns that match, then we could potentially ‘train’ a system to solve a problem. We saw a super simple example of that -- fitting numbers to a line.

## Introduction to Computer Vision

### An introduction to computer vision

Computer vision is the field of having a computer understand and label what is present in an image. Consider this slide. When you look at it, you can interpret what a shirt is or what a shoe is, but how would you program for that? 

<div align="center"> 
  <img src="readme_img/slide-vision.png" width="60%">
</div>

So one way to solve that is to use lots of pictures of clothing and tell the computer what that's a picture of and then have the computer figure out the patterns that give you the difference between a shoe, and a shirt, and a handbag, and a coat.