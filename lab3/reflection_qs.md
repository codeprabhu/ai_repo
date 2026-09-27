### 1. What did the XOR experiment demonstrate about the difference between depth and nonlinearity?

The XOR experiment demonstrated that depth alone is not sufficient to represent nonlinear decision boundaries. A network consisting only of affine transformations is mathematically equivalent to a single affine transformation, regardless of how many layers are stacked together. The introduction of a nonlinear activation function such as ReLU or Tanh allows the network to represent nonlinear relationships. Therefore, it is the combination of depth and nonlinearity that enables a neural network to learn XOR.

---

### 2. In your successful run, what evidence showed that backpropagation supplied a useful learning signal rather than merely a nonzero gradient?

Useful learning was demonstrated by several observations occurring simultaneously. The training loss decreased over time, predictions moved closer to the target values, and classification performance improved. Although nonzero gradients indicate that derivatives are being computed, useful learning requires those gradients to consistently guide parameter updates toward a lower loss. The reduction in loss and improvement in predictions provided evidence that backpropagation was producing a meaningful learning signal.

---

### 3. Why did identical/zero weight initialisation prevent the two hidden units from learning distinct features?

When hidden units start with identical weights and biases, they receive identical inputs, produce identical outputs, and obtain identical gradients during backpropagation. As a result, both units undergo exactly the same updates throughout training. This symmetry prevents the hidden units from specializing and learning different features of the data. Random initialization breaks this symmetry and allows different neurons to learn distinct representations.

---

### 4. How did changing the hidden activation affect the gradient you observed? Distinguish the scientific explanation from the engineering observation.

Scientifically, the activation function influences the derivatives propagated through the network during backpropagation. Sigmoid activations can produce small derivatives when neurons saturate, which may reduce gradient magnitudes. ReLU activations have a derivative of 1 in their active region, allowing gradients to propagate more effectively. Tanh often produces larger gradients near zero while remaining nonlinear.

From an engineering perspective, changing the activation function affected training behaviour. Some activations converged more reliably, reduced the loss faster, or learned XOR more successfully than others. These practical observations reflected the underlying mathematical differences in gradient propagation.

---

### 5. Why must the output layer and loss be selected together according to the task?

The output activation determines the interpretation of the model's predictions, while the loss function determines how prediction errors are measured. These components must be compatible. For binary classification, a sigmoid output combined with binary cross-entropy produces probabilistic predictions and an appropriate learning signal. For multiclass classification, softmax combined with cross-entropy provides a probability distribution across classes and suitable gradients. Choosing incompatible combinations can result in incorrect probability interpretations and poor optimization behaviour.

---

### 6. Give one example where the LLM improved your engineering productivity and one example where human verification was essential.

The LLM improved productivity by rapidly generating PyTorch model definitions, training loops, and loss-function configurations, reducing implementation time. Human verification remained essential when evaluating the generated code and interpreting the results. For example, the linear model failed to learn XOR despite running correctly, and human reasoning was required to determine that the limitation arose from model capacity rather than a programming error.

---

### 7. Which tests in this laboratory would you keep if the model were scaled up, and which would become too expensive?

I would continue monitoring training and validation loss, prediction accuracy, gradient statistics, and parameter updates because these checks scale well to larger models and remain useful for diagnosing learning behaviour.

However, exhaustive finite-difference gradient checking would become computationally expensive because every parameter must be perturbed individually. Similarly, manually inspecting predictions for every training example becomes impractical on large datasets. For larger models, these detailed checks would typically be replaced by sampling-based validation, automated testing, and statistical monitoring.
