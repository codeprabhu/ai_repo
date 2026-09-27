Prediction and Analysis
Task 1 – Problem Specification

The task is a binary classification problem in which two binary sensor readings are used to predict whether the sensors disagree. The desired function is XOR.

Input Space
$$ X = \{(0,0), (0,1), (1,0), (1,1)\} $$
Output Space
$$ Y = \{0,1\} $$
Training Examples
x₁	x₂	y
0	0	0
0	1	1
1	0	1
1	1	0
Linear Separability Analysis

XOR is not linearly separable because the positive examples (0,1) and (1,0) lie on opposite corners of the input space while the negative examples (0,0) and (1,1) occupy the remaining corners. No single straight decision boundary can separate the two classes.

Initial Prediction

I predicted that a model consisting of a single affine transformation followed by a sigmoid activation would be unable to learn XOR perfectly because it can only represent a linear decision boundary.

Task 2 – Model Design
Model A: Linear Baseline
Input(2)
   ↓
Linear(2→1)
   ↓
Sigmoid

Expected Result:

Loss may decrease slightly.
Gradients should exist.
The model should fail to classify XOR perfectly.
Model B: Nonlinear Neural Network
Input(2)
   ↓
Linear(2→H)
   ↓
Tanh/ReLU
   ↓
Linear(H→1)
   ↓
Sigmoid

Expected Result:

Nonlinear hidden representations should allow the network to learn XOR.
Training loss should decrease significantly.
Predictions should move closer to the target values.
Validation Criteria

A model is considered successful if:

Training loss decreases during learning.
Predictions move toward the correct target values.
Classification accuracy improves.
Gradients are non-zero during backpropagation.
The network correctly classifies all four XOR examples.
Experimental Analysis
Linear Model Results

The linear model converged to a loss of approximately 0.693 and produced predictions close to 0.5 for all inputs. This confirmed the prediction that a single affine transformation cannot represent XOR.

Observation:

The model learned a trivial solution corresponding to chance-level performance and was unable to separate the XOR classes.

Nonlinear Model Results

After introducing a hidden layer with a nonlinear activation function, the loss decreased significantly and the predictions improved. This demonstrated that nonlinear hidden representations provide greater expressive power than a purely linear model.

Observation:

The nonlinear model learned meaningful structure in the data and achieved better performance than the linear baseline.

Activation Function Experiment

The hidden activation function influenced both gradient propagation and training behaviour.

Sigmoid activations may produce small gradients when saturated.
Tanh activations often provide stronger gradients near zero.
ReLU activations maintain gradients in their active region but produce zero gradients for negative inputs.

Observation:

Different activations produced different optimization behaviour even though the underlying task remained unchanged.

Symmetry Experiment

When all hidden-layer weights were initialized to identical values, hidden neurons received identical gradients and evolved identically throughout training.

Observation:

Symmetric initialization prevented hidden units from learning distinct features, demonstrating the importance of random initialization in neural networks.

Three-Class Extension

The binary XOR task was extended into a three-class classification problem:

Input	Class
(0,0)	0
(0,1)	1
(1,0)	1
(1,1)	2

The output layer was modified to produce three logits and the loss function was changed to CrossEntropyLoss.

Observation:

Softmax converted the logits into a valid probability distribution whose components summed to one. Cross-entropy provided a suitable learning signal for multiclass classification.