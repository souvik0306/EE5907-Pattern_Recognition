# EE5907 Pattern Recognition Continuous Assessment 1 - NUS

## Overview

This report explores the implementation of two classification techniques from scratch: a **Multi-Layer Perceptron (MLP)** using backpropagation and **Radial Basis Function (RBF) Networks** with randomly selected centers. Both methods are applied to a binary classification problem, and their performances are compared.

## Contents

### Part 1: Classification with Multi-Layer Perceptron (MLP)

- **Data Generation and Visualization:** Binary class data was generated and visualized.
- **MLP Architecture:** The MLP consists of 2 input neurons, 3 hidden neurons (ReLU), and 1 output neuron (Sigmoid). Initial accuracy: **63%**.
- **Backpropagation and Training:** Binary Cross-Entropy loss, gradient descent over multiple epochs. Final accuracy: **84%**.
- **Decision Boundary Visualization:** Plots before and after backpropagation show improved separation.
- **Comparison:** Performance before and after backpropagation.

### Part 2: Classification with Radial Basis Function (RBF) Networks

- **RBF with 3 Neurons:** 3 randomly selected centers, Gaussian RBF function, Least Squares Estimation. Accuracy: **71%**.
- **RBF with 6 Neurons:** Increased to 6 centers. Improved accuracy: **80%**.
- **Decision Boundary Visualization:** Plots for 3 and 6 neurons.
- **Comparison of 3 vs. 6 Neurons:** Trade-off between complexity and generalization.

### Conclusions

- **MLP vs RBF:** MLP (84%) outperformed RBF (80% with 6 centers).
- **Effect of Neurons in RBF:** Performance improved with more centers, but balance needed to avoid overfitting.
- **Visual Insights:** Decision boundaries and accuracy comparisons.

### Final Thoughts

This report demonstrates neural networks implemented from scratch and tuned for improved classification. The comparison provides insights into the strengths and limitations of MLPs and RBF networks.

## How to Run the Scripts

### Prerequisites

Ensure you have Python installed along with the necessary libraries:

- `numpy`
- `matplotlib`

Install the required libraries using:

```sh
pip install numpy matplotlib 
```

### Dataset

- `class1.npy` and `class2.npy` are dataset arrays. You can generate your own using `generate_data.py`.

### Running the Scripts

1. **Data Generation and Visualization:**

- Script: `view_data.py`
- Command:
  ```sh
  python view_data.py
  ```

2. **Multi-Layer Perceptron (MLP):**

- Script: `MLP_random.py`
- Command:
  ```sh
  python MLP_random.py
  ```

3. **Backpropagation Training:**

- Script under `Back_Prop/`: `grad_loop_final.py`
- Command:
  ```sh
  python grad_loop.py
  ```

4. **Radial Basis Function (RBF) Networks:**

- Script under `RBF/`: `RBF_3.py` or `RBF_6.py`
- Command:

  ```sh
  python RBF_3.py
  ```
  or

  ```sh
  python RBF_6.py
  ```
