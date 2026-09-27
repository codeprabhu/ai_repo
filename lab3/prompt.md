Write a complete, well-documented PyTorch program that learns the XOR function.

The XOR dataset is:

Input -> Target
(0,0) -> 0
(0,1) -> 1
(1,0) -> 1
(1,1) -> 0

Requirements:

1. Create the XOR dataset using PyTorch tensors.

2. Implement a linear baseline model:
   - Input size = 2
   - Output size = 1
   - Single Linear layer followed by Sigmoid
   - Use BCELoss
   - Use SGD optimizer
   - Train for several thousand epochs
   - Print training loss periodically
   - Print final predictions for all four inputs

3. Explain why this model is expected to fail on XOR.

4. Implement a second neural network with:
   - Input layer size 2
   - Hidden layer size 2
   - Nonlinear hidden activation (Tanh or ReLU)
   - Output layer size 1
   - Sigmoid output
   - BCELoss
   - SGD optimizer

5. During training of the nonlinear model:
   - Print the loss periodically
   - Print the gradients of the first layer weights after backpropagation
   - Print final predictions

6. Compare the performance of the linear and nonlinear models and explain the role of the hidden nonlinear layer.

7. Use clear variable names and include comments explaining each major step.

The program must be executable as a standalone Python file.

Modify the XOR network into a three-class classifier.

Class 0: (0,0)
Class 1: (0,1) and (1,0)
Class 2: (1,1)

Requirements:
- Keep the hidden layer.
- Replace the binary output with three logits.
- Use CrossEntropyLoss.
- Train the network.
- Print logits, softmax probabilities, and predicted classes.
- Verify that each probability vector sums to approximately 1.
- Explain why softmax outputs form a probability distribution and why the gradient of softmax cross-entropy is p - y.