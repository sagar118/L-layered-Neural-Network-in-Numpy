# L-layered-Neural-Network-in-Numpy
Build a complete neural network using Numpy. All the steps required to build a network - feedforward, loss computation, backpropagation, weight updates etc.

Divided into the following sections:

    1. Data preparation
    2. Feedforward
    3. Loss computation
    4. Backpropagation
    5. Parameter updates
    6. Model training and predictions
    
Data Preparation:<br>
Firstly, we load the data using the function load_data(). The function data_wrapper() is then applied to the data to get the train and test data in the desired shape. Please note that the code needs to take a batch of data points as the input.
You already know that we have 28x28 greyscale images in the MNIST dataset. Hence, each input image is a vector of length 784. The ground truth labels of a batch are stored in a matrix which is converted to a one-hot matrix. Also, the output of the model is a softmax output of length 10.

Hence, we have the following:

    train_set_x shape: (784, 50000)
    train_set_y shape: (10, 50000)
    test_set_x shape: (784, 10000)
    test_set_y shape: (10, 10000)

Feedforward: <br>
The feedforward algorithm.

    H0=B
    for l in [1,2,.......,L−1]:
        Zl=Wl.Hl−1+bl
        Hl=σ(Zl)
    HL = softmax(WL.HL−1+bL)

Now, let's summarize all the functions we have defined in feedforward:

| Function | Arguments | Returns | Explanation |
| --- | --- | --- | --- | --- |
| sigmoid | Z | H, sigmoid_memory | Applies sigmoid activation on Z to calculate H. Returns H, sigmoid_memory = Z  (Step 2.2) |
