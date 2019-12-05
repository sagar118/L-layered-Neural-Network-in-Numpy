# L-layered-Neural-Network-in-Numpy
Build a complete neural network using Numpy. All the steps required to build a network - feedforward, loss computation, backpropagation, weight updates etc.

Divided into the following sections:

1. Data preparation
2. Feedforward
3. Loss computation
4. Backpropagation
5. Parameter updates
6. Model training and predictions
    
### Data Preparation:<br>
Firstly, we load the data using the function load_data(). The function data_wrapper() is then applied to the data to get the train and test data in the desired shape. Please note that the code needs to take a batch of data points as the input.
You already know that we have 28x28 greyscale images in the MNIST dataset. Hence, each input image is a vector of length 784. The ground truth labels of a batch are stored in a matrix which is converted to a one-hot matrix. Also, the output of the model is a softmax output of length 10.

Hence, we have the following:
```
train_set_x shape: (784, 50000)
train_set_y shape: (10, 50000)
test_set_x shape: (784, 10000)
test_set_y shape: (10, 10000)
```

### Feedforward: <br>
The feedforward algorithm.

```
H0=B
for l in [1,2,.......,L−1]:
    Zl=Wl.Hl−1+bl
    Hl=σ(Zl)
HL = softmax(WL.HL−1+bL)
```

Now, let's summarize all the functions we have defined in feedforward:


| Function | Arguments | Returns | Explanation |
| :---:    | :---:     | :---:   | :---:       |
|sigmoid| Z | H, sigmoid_memory | Applies sigmoid activation on Z to calculate H. Returns H, sigmoid_memory = Z  (Step 2.2) |
| relu | Z | H, relu_memory | 	Applies relu activation on Z to calculate H. Returns H, relu_memory = Z  (Step 2.2) |
| softmax | Z | H, softmax_memory | Applies softmax activation on Z, i.e raise to an exponent and then normalize to get H. Returns H, softmax_memory = Z (Step 2.2) |
| initialize_parameters | dimensions | W and b for all the layers | Runs a for loop to initialize the W and b of all the layers stored as dictionary parameters |
| layer_forward | H_prev, W, b, activation | H, memory | Performs Step 2. Computes Z  using step 2.1 and stores Z as linear_memory and then computes H, activation_memory  using Z as input into the activation functions defined above. Check the arguments and returns of relu, sigmoid, softmax to define this function. Returns H & memory = (linear_memory, activation_memory) |
| L_layer_forward | X, parameters | HL, memories | Uses parameters initialized by initialize_parameters and performs feedforward using layer_forward in a 'for loop' for all layers. Appends memory for each layer to memories. layer_forward for the last layer is performed outside the loop since the activation function for all layers is relu except for the last layer (which is softmax). Returns the network output HL and memories(Steps 1 to 3) |

### Loss Calculation: 

The loss used for multiclass classification is the cross-entropy loss.

| Function | Arguments | Returns | Explanation |
| :---:    | :---:     | :---:   | :---:       |
| compute_loss | HL, Y | loss | It returns loss as the average of the loss of all data points. HL and Y are matrices. Hence, loss  = -1.\*average of the sum of all the elements of the matrix Ylog(HL) multiplied element-wise.|

### Backpropagation:

The backpropagation algorithm pseudocode is as follows:

```
DZL =   P−Y
∂L/∂WL =(1/m)*DWL =  (1/m)*DZL.(HL−1)T
∂L/∂bL = (1/m)*DbL  = (1/m)*DZL
 dHL−1 = (WL)T.DZL
for l in [L−1,........1]:
     DZl = dHl⊗σ′(Zl)
     ∂L/∂Wl = (1/m)*DWl = (1/m)*DZl.(Hl−1)T
     ∂L/∂bl = (1/m)*Dbl  = (1/m)*DZl
    dHl−1 = (Wl)T.DZl
```

| Function | Arguments | Returns | Explanation |
| :---:    | :---:     | :---:   | :---:       |
| sigmoid_backward | dH, sigmoid_memory | dZ | Z = sigmoid_memory (stored during feedforward); uses this Z and dH to calculate the gradient dZ (Step 5.1) |
| relu_backward | dH, relu_memory | dZ | Z = relu_memory (stored during feedforward); uses this Z and dH to calculate the gradient dZ (Step 5.1) |
| layer_backward | dH, memory, activation | dH_prev, dW, db | Performs Steps 5.1 to 5.4. Calculates dZ using activation_backward and then uses dZ to calculate dH_prev, dW, db using equations in steps 5.2 to 5.4. |
| L_layer_backward | HL, Y, memories | gradients | Uses layer_backward in a 'for loop' using relu activation to get gradients  dH_prev, dW, db stored in a dictionary gradients. Note that the backprop for the last layer(Steps 1 to 4) is calculated outside and before the for loop. Returns gradients which has dH, dW, db for all the layers. |

### Parameter Updates

In this section, you have to define the function that updates the weights and biases for all the layers using a 'for' loop.

| Function | Arguments | Returns | Explanation |
| :---:    | :---:     | :---:   | :---:       |
| update_parameters | parameters, gradients, learning_rate | parameters | 	parameters has the parameters which is updated using the learning_rate and the gradients stored in gradients. |

### Model Training

This is the final step in which you combine all the functions created above to define an 'L_layer_model'. 

| Function | Arguments | Returns | Explanation |
| :---:    | :---:     | :---:   | :---:       |
| L_layer_model | 	X, Y, dimensions, learning_rate , num_iterations, print_loss | parameters | Initializes parameters with initialize_parameters. In a for loop uses functions L_layer_forward, compute_loss, L_layer_backward, update_parameters in the specified order. |

After this, start the training by specifying the learning rate and the number of iterations.
