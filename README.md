# Pattern Recognition with Single Layer Neural Network (SLNN)
### Noa Mediavilla & Rebeca Torrecilla

![MATLAB](https://img.shields.io/badge/MATLAB-R2023b%2B-orange)
![Optimization](https://img.shields.io/badge/Optimization-Unconstrained-blue)
![Status](https://img.shields.io/badge/Status-Educational-green)

The main objective is to develop an application capable of recognizing blurred digits (0-9) from a sequence of images. To achieve this, we implemented and trained a **Single Layer Neural Network (SLNN)** using various first-derivative optimization methods.

The project involves defining the network architecture, formulating the loss function with **L2 Regularization**, computing the analytical gradient, and implementing custom optimization algorithms from scratch.


## Methodology

### Architecture
*   **Input:** 7x5 pixel matrix (vectorized to $n=35$) representing blurred digits.
*   **Activation Function:** Sigmoid function $\sigma(x) = 1/(1+e^{-x})$.
*   **Output:** Binary classification (1 if target digit, 0 otherwise).

### Optimization Problem
The training process minimizes a **Mean Squared Error (MSE)** loss function combined with **L2 Regularization** (Weight Decay) to prevent overfitting.

$$ \min_{w \in \mathbb{R}^n} \tilde{L}(w) = \frac{1}{p} \sum_{j=1}^{p} (y(x_j^{TR}, w) - y_j^{TR})^2 + \frac{\lambda}{2} \|w\|^2 $$

### Algorithms Implemented
The following unconstrained optimization algorithms were used to minimize the loss function:

1.  **Gradient Method (GM):** Uses a backtracking line search (BLS) with cubic interpolation to determine step size.
2.  **Quasi-Newton Method (QNM):** Approximates the Hessian to improve convergence speed.
3.  **Stochastic Gradient Method (SGM):** Implements mini-batches, learning rate decay, and **Early Stopping** based on validation accuracy to handle large datasets efficiently.


## Usage
### Running experiments
To reproduce computational experiments:
1. Open MATLAB and navigate to the project directory.
2. Run the batch script:
```
uo_nn_batch_st
```
3. This will generate a detailed logs of the execution (`uo_nn_batch.log`) and a summary table with iterations, time and accuracy metrics (`uo_nn_batch.csv`).

### Single Instance Training
To train the network for a specific digit with a specific algorithm inside `uo_nn_solve`:
```
num_target = [4];
tr_freq = 0.5;
lambda = 0.05;
[w, tr_acc, te_acc] = uo_nn_solve(num_target, tr_freq, lambda, ...);
```

## Experimental Results
This project analyzes the performance of Gradient Method, Quasi-Newton Method and Stochastic Gradient Method based on:
1. **Global Convergence**: Ability to reach the optimal solution.
2. **Local Convergence**: Speed in terms of time and iterations (only if we reached the global convergence).
3. **Accuracy**: Recognition rate on Test vs. Training sets.
4. **Regularization Sensity**: Impact of $\lambda \in {0.0, 0.05, 0.1}$.

Detailed analysis and plots can be found in the report document. 

## License 
This work is based on the assignment by F.-Javier Heredia (UPC).
Creative Commons Attribution NonCommercial-NoDerivs 3.0 Unported License.














