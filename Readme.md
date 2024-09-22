# EE5907 Pattern Recognition Continuous Assessment 1

## Overview
This report explores the implementation of two classification techniques from scratch: a **Multi-Layer Perceptron (MLP)** using backpropagation and **Radial Basis Function (RBF) Networks** with randomly selected centers. Both methods are applied to a binary classification problem, and their performances are compared in terms of accuracy, decision boundaries, and generalization capabilities.

## Contents

### Part 1: Classification with Multi-Layer Perceptron (MLP)
- **Data Generation and Visualization:** 
  - Binary class data was generated and visualized as a scatter plot, distinguishing Class 1 and Class 2.
  
- **MLP Architecture:**
  - The MLP consists of 2 input neurons, 3 hidden neurons (with ReLU activation), and 1 output neuron (with Sigmoid activation). 
  - Initial random weights resulted in a starting classification accuracy of around **63%**.

- **Backpropagation and Training:**
  - **Loss Function:** Binary Cross-Entropy was used to guide the backpropagation process.
  - **Training:** The weights and biases were updated iteratively using gradient descent over multiple epochs. 
  - **Final Accuracy:** After training, the accuracy improved to **84%**, highlighting the importance of backpropagation in refining model performance.

- **Decision Boundary Visualization:**
  - Decision boundaries before and after backpropagation were plotted, showcasing how the updated weights resulted in better separation between the classes.
  - This visually demonstrates the improvement in classification accuracy after training.

- **Comparison:** 
  - A table summarizes the performance of the MLP before and after backpropagation, showing how the model adapts to the data through weight and bias updates.

### Part 2: Classification with Radial Basis Function (RBF) Networks
- **RBF with 3 Neurons:**
  - An RBF network was implemented using 3 randomly selected centers (neurons). 
  - **Gaussian RBF function:** The activation function was based on the distance between the input points and the centers, with sigma controlling the spread.
  - **Least Squares Estimation:** Weights were calculated using the pseudo-inverse of the RBF matrix.
  - **Classification Accuracy:** The RBF network achieved an accuracy of **71%**.

- **RBF with 6 Neurons:**
  - The number of neurons (RBF centers) was increased to 6 to provide more flexibility in modeling the data.
  - **Improved Accuracy:** With 6 centers, the accuracy increased to **80%**, demonstrating the benefit of additional hidden neurons in capturing more complex patterns in the data.

- **Decision Boundary Visualization:**
  - The decision boundaries for both the 3-neuron and 6-neuron networks were plotted, showing improved classification with additional neurons.

- **Comparison of 3 vs. 6 Neurons:**
  - A detailed comparison between 3 and 6 neurons highlighted the trade-off between model complexity and generalization. While more neurons allowed for more precise decision boundaries, care was taken to avoid overfitting.

### Conclusions
- **MLP vs RBF:** 
  - The MLP, after training with backpropagation, outperformed the RBF network, achieving an accuracy of **84%** compared to **80%** with 6 RBF centers.
  
- **Effect of Neurons in RBF:** 
  - The RBF network’s performance improved as more centers were added, but a balance is needed to prevent overfitting.
  
- **Visual Insights:** 
  - The decision boundaries and accuracy comparisons provide visual evidence of how both models separate the two classes and how they adapt with training and increased complexity.

### Final Thoughts
This report demonstrates how neural networks can be implemented from scratch and tuned to improve classification performance. The comparison between MLPs and RBF networks provides insights into the strengths and limitations of each approach in handling nonlinear separations in data.
